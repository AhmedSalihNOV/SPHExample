module SPHCellList

export NeighborLoop!, ComputeInteractions!, RunSimulation

using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SPHNeighborList: BuildNeighborCellLists!, ComputeCellNeighborCounts, ComputeCellParticleCounts, ConstructStencil, ExtractCells!, FindCellIndex, MapFloor, NeighborParticleRanges, NeighborSortScratch, PackedNeighborCellLists, UpdateNeighbors!, UpdateΔx!

using Base.Threads: @threads, nthreads, Atomic, atomic_add!
using Bumper: @alloc, @no_escape
using FastPow: @fastpow
using LinearAlgebra: det, dot, norm
using Parameters: @unpack
using StaticArrays: SMatrix, SVector
using StructArrays: StructArray
using TimerOutputs: @timeit, flatten

    # Single-neighbor stepping reuses the derivative evaluated at the previous
    # half state. Re-anchor that carried derivative periodically at an accepted
    # full state to suppress its long-time parasitic mode. Twenty steps is the
    # coarsest tested interval that retained the stable StillWedge hydrostatic
    # result; 40 and 80 steps left progressively more drift. This cadence belongs
    # to the integrator and must remain independent of output scheduling.
    const SingleNeighborCorrectionInterval = 20

    const ParticleBatchSize = 64

    # Spatially sorted particles can give contiguous thread partitions very
    # different amounts of work, especially for MDBC's sparse ghost points.
    # Workers take small contiguous batches while each particle retains its
    # original neighbor order and owns all of its accumulator writes.
    # @threads joins every worker before returning or propagating exceptions,
    # keeping the caller's Bumper buffers alive for the entire computation.
    @inline function ForEachParticle!(Body::F, Indices::AbstractUnitRange) where {F}
        if nthreads(:default) == 1 || length(Indices) <= ParticleBatchSize
            for Index in Indices
                Body(Index)
            end
        else
            NextBatch = Atomic{Int}(first(Indices))
            @threads for Worker in 1:nthreads(:default)
                BatchStart = atomic_add!(NextBatch, ParticleBatchSize)
                while BatchStart <= last(Indices)
                    for Index in BatchStart:min(BatchStart + ParticleBatchSize - 1, last(Indices))
                        Body(Index)
                    end
                    BatchStart = atomic_add!(NextBatch, ParticleBatchSize)
                end
            end
        end
        return nothing
    end

    @inline function NeedsSingleNeighborCorrection(Iteration::Integer,
                                                    RefreshedAfterRebuild::Bool)
        return !RefreshedAfterRebuild &&
               Iteration > 0 &&
               mod(Iteration, SingleNeighborCorrectionInterval) == 0
    end

    # Reciprocal densities of the evaluated state are shared by every pair of a
    # force evaluation. Computing them once per particle turns the per-pair
    # divisions by density in the pair kernel and its models into products.
    @inline function FillInverseDensity!(InvDensity, Density)
        @inbounds @simd for i in eachindex(InvDensity, Density)
            InvDensity[i] = inv(Density[i])
        end
        return nothing
    end

    # ------------------------------------------------------------------
    # Symmetric pair evaluation for NoShifting/NoKernelOutput.
    #
    # Each unordered pair is evaluated once, from the cell with the lower
    # index: a cell handles the pairs inside itself (j > i) and those with its
    # neighbor cells of higher index. The terms particle j would compute as
    # the center are derived from the same evaluation, so every per-pair
    # value matches the per-particle loop; only the summation order differs.
    #
    # Workers take dynamic batches of owner cells and write to private
    # accumulator copies that a parallel pass then sums. This costs
    # nthreads × N × (D+1) floats, but needs no atomics, cell colouring, or
    # barriers inside the pair loop, and halves the pair work.
    struct SymmetricAccumulators{T,V}
        DensityRate::Vector{Vector{T}}
        Acceleration::Vector{Vector{V}}
    end

    function SymmetricAccumulators(Position::AbstractVector{SVector{D,T}}) where {D,T}
        Slots = nthreads(:default)
        Slots > 1 || return SymmetricAccumulators(Vector{T}[], Vector{SVector{D,T}}[])
        return SymmetricAccumulators([Vector{T}(undef, length(Position)) for _ in 1:Slots],
                                     [Vector{SVector{D,T}}(undef, length(Position)) for _ in 1:Slots])
    end

    # Modes with optional per-particle outputs keep the per-particle loop.
    MakeInteractionWorkspace(MetaData, Position) = nothing
    MakeInteractionWorkspace(::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, Position) where {D,T,B,L} = SymmetricAccumulators(Position)

    @inline EvaluateInteractions!(::Nothing, Args...; Kwargs...) = NeighborLoopPerParticle!(Args...; Kwargs...)

    # Single-cell batches balance small grids best; larger grids amortise the
    # shared counter over more cells per batch.
    @inline SymmetricCellBatchSize(CellCount) = max(1, cld(CellCount, 64 * nthreads(:default)))

    function EvaluateInteractions!(Workspace::SymmetricAccumulators, SimDensityDiffusion, SimViscosity, SimKernel,
                                   SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants, SimParticles, ParticleRanges,
                                   CellListIndices, NeighborCellLists, dρdtI, Acceleration,
                                   ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared;
                                   Position=SimParticles.Position, Density=SimParticles.Density,
                                   Pressure=SimParticles.Pressure, Velocity=SimParticles.Velocity,
                                   InvDensity=similar(Density)) where {D,T,B,L}
        ParticleType = SimParticles.Type
        FillInverseDensity!(InvDensity, Density)
        CellCount = length(NeighborCellLists)
        Slots = length(Workspace.DensityRate)
        if Slots <= 1 || nthreads(:default) == 1
            fill!(dρdtI, zero(eltype(dρdtI)))
            fill!(Acceleration, zero(eltype(Acceleration)))
            SymmetricCells!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
                SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                ParticleRanges, NeighborCellLists, dρdtI, Acceleration, 1:CellCount)
            @inbounds for i in eachindex(Acceleration)
                AccelerationNormSquared[i] = dot(Acceleration[i], Acceleration[i])
            end
            return nothing
        end

        Batch = SymmetricCellBatchSize(CellCount)
        NextBatch = Atomic{Int}(1)
        @threads for Slot in 1:Slots
            DensityRateSlot = Workspace.DensityRate[Slot]
            AccelerationSlot = Workspace.Acceleration[Slot]
            resize!(DensityRateSlot, length(Position))
            resize!(AccelerationSlot, length(Position))
            fill!(DensityRateSlot, zero(eltype(DensityRateSlot)))
            fill!(AccelerationSlot, zero(eltype(AccelerationSlot)))
            BatchStart = atomic_add!(NextBatch, Batch)
            while BatchStart <= CellCount
                SymmetricCells!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
                    SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                    ParticleRanges, NeighborCellLists, DensityRateSlot, AccelerationSlot,
                    BatchStart:min(BatchStart + Batch - 1, CellCount))
                BatchStart = atomic_add!(NextBatch, Batch)
            end
        end

        # Sum the private copies; each particle is written by exactly one worker.
        @threads for i in eachindex(Position)
            dρdt_acc = zero(eltype(dρdtI))
            acc_acc = zero(eltype(Acceleration))
            @inbounds for Slot in 1:Slots
                dρdt_acc += Workspace.DensityRate[Slot][i]
                acc_acc += Workspace.Acceleration[Slot][i]
            end
            @inbounds begin
                dρdtI[i] = dρdt_acc
                Acceleration[i] = acc_acc
                AccelerationNormSquared[i] = dot(acc_acc, acc_acc)
            end
        end
        return nothing
    end

    # Owner-cell traversal. `Cells` may index the empty sentinel cell.
    function SymmetricCells!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
                             SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                             ParticleRanges, NeighborCellLists, DensityRateOut, AccelerationOut, Cells)
        @inbounds for Cell in Cells
            First = ParticleRanges[Cell]
            Last = ParticleRanges[Cell + 1] - 1
            First <= Last || continue
            Neighbors = NeighborCellLists[Cell]
            for i in First:Last
                xᵢ = Position[i]
                vᵢ = Velocity[i]
                ρᵢ = Density[i]
                ρᵢ⁻¹ = InvDensity[i]
                Pᵢ = Pressure[i]
                dρdt_acc = zero(eltype(DensityRateOut))
                acc_acc = zero(eltype(AccelerationOut))
                for j in (i + 1):Last
                    dρdt_acc, acc_acc = ComputeSymmetricInteraction!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
                        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                        DensityRateOut, AccelerationOut, xᵢ, vᵢ, ρᵢ, ρᵢ⁻¹, Pᵢ, dρdt_acc, acc_acc, i, j,
                    )
                end
                for Other in Neighbors
                    Other > Cell || continue
                    for j in ParticleRanges[Other]:(ParticleRanges[Other + 1] - 1)
                        dρdt_acc, acc_acc = ComputeSymmetricInteraction!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
                            SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                            DensityRateOut, AccelerationOut, xᵢ, vᵢ, ρᵢ, ρᵢ⁻¹, Pᵢ, dρdt_acc, acc_acc, i, j,
                        )
                    end
                end
                DensityRateOut[i] += dρdt_acc
                AccelerationOut[i] += acc_acc
            end
        end
        return nothing
    end

    # One pair, evaluated once. Returns particle i's running accumulators and
    # adds particle j's contributions to DensityRateOut[j] / AccelerationOut[j].
    # The i terms are the per-particle loop's expressions verbatim; the j terms
    # use the exact identities ∇ⱼWᵢⱼ = -∇ᵢWᵢⱼ and dot(-vⱼᵢ, ∇ⱼWᵢⱼ) = dot(-vᵢⱼ, ∇ᵢWᵢⱼ),
    # and the models return what j would compute as the center.
    Base.@propagate_inbounds function ComputeSymmetricInteraction!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        DensityRateOut, AccelerationOut, xᵢ, vᵢ, ρᵢ, ρᵢ⁻¹, Pᵢ, dρdt_acc, acc_acc, i, j) where {D,T,
                                                                     B<:MDBCMode,
                                                                     L<:LogMode,
                                                                     SDD<:SPHDensityDiffusion,
                                                                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H² = SimKernel

        xᵢⱼ = xᵢ - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρⱼ = Density[j]
            ρⱼ⁻¹ = InvDensity[j]

            vᵢⱼ = vᵢ - Velocity[j]
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            Vⱼ = m₀ * ρⱼ⁻¹
            Vᵢ = m₀ * ρᵢ⁻¹
            dρdt⁺ = -ρᵢ * Vⱼ * density_symmetric_term
            dρdtⱼ⁺ = -ρⱼ * Vᵢ * density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ
            DensityRateOut[j] += dρdtⱼ⁺ + Dⱼ

            Pⱼ = Pressure[j]
            ρᵢρⱼ_inv = ρᵢ⁻¹ * ρⱼ⁻¹
            Pfac = (Pᵢ + Pⱼ) * ρᵢρⱼ_inv
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_termᵢ, visc_termⱼ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

            acc_acc += dvdt⁺ + visc_termᵢ
            AccelerationOut[j] += -dvdt⁺ + visc_termⱼ
        end

        return dρdt_acc, acc_acc
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationNormSquared;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity,
                                      InvDensity = similar(Density)) where {D,T,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        FillInverseDensity!(InvDensity, Density)
        @inbounds ForEachParticle!(eachindex(Position)) do i
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                )
            end
            for ParticleSpan in NeighborParticleRanges(NeighborCellIndices, ParticleRanges)
                @inbounds for j in ParticleSpan
                    dρdt_acc, acc_acc = ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            AccelerationNormSquared[i] = dot(acc_acc, acc_acc)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationNormSquared;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity,
                                      InvDensity = similar(Density)) where {D,T,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        FillInverseDensity!(InvDensity, Density)
        @inbounds ForEachParticle!(eachindex(Position)) do i
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, i, j,
                    )
            end
            for ParticleSpan in NeighborParticleRanges(NeighborCellIndices, ParticleRanges)
                @inbounds for j in ParticleSpan
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc =
                        ComputeInteractionsPerParticle!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                            Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                            kernel_grad_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            AccelerationNormSquared[i] = dot(acc_acc, acc_acc)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationNormSquared;
                                      Position = SimParticles.Position,
                                      Density = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity,
                                      InvDensity = similar(Density)) where {D,T,
                                                  S<:ShiftingMode,B<:MDBCMode,
                                                  L<:LogMode,SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        ParticleType = SimParticles.Type
        FillInverseDensity!(InvDensity, Density)
        @inbounds ForEachParticle!(eachindex(Position)) do i
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                    ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                        shift_r_acc, i, j,
                    )
            end
            for ParticleSpan in NeighborParticleRanges(NeighborCellIndices, ParticleRanges)
                @inbounds for j in ParticleSpan
                    dρdt_acc, acc_acc, shift_c_acc, shift_r_acc =
                        ComputeInteractionsPerParticle!(
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                            Velocity, ParticleType, dρdt_acc, acc_acc, shift_c_acc,
                            shift_r_acc, i, j,
                        )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationNormSquared[i] = dot(acc_acc, acc_acc)
        end

        return nothing
    end

    function NeighborLoopPerParticle!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                      SimConstants, SimParticles, ParticleRanges,
                                      CellListIndices, NeighborCellLists, dρdtI,
                                      Acceleration, ∇Cᵢ,
                                      ∇◌rᵢ, AccelerationNormSquared;
                                      Position = SimParticles.Position,
                                      Density  = SimParticles.Density,
                                      Pressure = SimParticles.Pressure,
                                      Velocity = SimParticles.Velocity,
                                      InvDensity = similar(Density)) where {D,T,
                                                  S<:ShiftingMode,
                                                  K<:KernelOutputMode,
                                                  B<:MDBCMode,L<:LogMode,
                                                  SDD<:SPHDensityDiffusion,
                                                  SV<:SPHViscosity}
        @unpack Kernel, KernelGradient = SimParticles
        ParticleType = SimParticles.Type
        FillInverseDensity!(InvDensity, Density)
        @inbounds ForEachParticle!(eachindex(Position)) do i
            dρdt_acc = zero(dρdtI[i])
            acc_acc = zero(Acceleration[i])
            kernel_acc = zero(Kernel[i])
            kernel_grad_acc = zero(KernelGradient[i])
            shift_c_acc = zero(∇Cᵢ[i])
            shift_r_acc = zero(∇◌rᵢ[i])
            CellListIndex = CellListIndices[i]
            SameCellStart = ParticleRanges[CellListIndex]
            SameCellEnd = ParticleRanges[CellListIndex + 1] - 1
            NeighborCellIndices = NeighborCellLists[CellListIndex]

            @inbounds for j in SameCellStart:(i - 1)
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            @inbounds for j in (i + 1):SameCellEnd
                dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                    shift_r_acc = ComputeInteractionsPerParticle!(
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                    Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                    kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                )
            end
            for ParticleSpan in NeighborParticleRanges(NeighborCellIndices, ParticleRanges)
                @inbounds for j in ParticleSpan
                    dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
                        shift_r_acc = ComputeInteractionsPerParticle!(
                        SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                        SimConstants, SimParticles, Position, Density, InvDensity, Pressure,
                        Velocity, ParticleType, dρdt_acc, acc_acc, kernel_acc,
                        kernel_grad_acc, shift_c_acc, shift_r_acc, i, j,
                    )
                end
            end

            dρdtI[i] = dρdt_acc
            Acceleration[i] = acc_acc
            Kernel[i] = kernel_acc
            KernelGradient[i] = kernel_grad_acc
            ∇Cᵢ[i] = shift_c_acc
            ∇◌rᵢ[i] = shift_r_acc
            AccelerationNormSquared[i] = dot(acc_acc, acc_acc)
        end

        return nothing
    end

    f(SimKernel, GhostPoint) = CartesianIndex(map(x -> MapFloor(x, SimKernel.H⁻¹), Tuple(GhostPoint)))
    # Fixed ghosts keep the same stencil until the cell map is rebuilt. Store
    # only active ghosts, preserving both particle and stencil traversal order.
    struct MDBCNeighborCache{N}
        GhostIndices::Vector{Int}
        NeighborCells::Vector{SVector{N,Int}}
    end

    MakeMDBCNeighborCache(::SimulationMetaData{D,T,S,K,NoMDBC,L}, SimParticles) where {D,T,S,K,L} = nothing

    function MakeMDBCNeighborCache(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, SimParticles) where {D,T,S,K,L}
        GhostPoints = SimParticles.GhostPoints
        # Moving or fluid ghosts use the existing lookup path on every call.
        any(i -> !iszero(GhostPoints[i]) && SimParticles.Type[i] != Fixed, eachindex(GhostPoints)) && return nothing
        return MDBCNeighborCache{3^D}(Int[], SVector{3^D,Int}[])
    end

    UpdateMDBCNeighborCache!(::Nothing, SimKernel, SimParticles, UniqueCellsView) = nothing

    # `UniqueCellsView` is the sorted active-cell list; FindCellIndex maps an
    # absent cell to the empty sentinel at index 1.
    function UpdateMDBCNeighborCache!(Cache::MDBCNeighborCache{N}, SimKernel::SPHKernelInstance{K,D}, SimParticles, UniqueCellsView) where {N,K,D}
        empty!(Cache.GhostIndices)
        empty!(Cache.NeighborCells)
        FullStencil = ConstructStencil(Val(D))
        @inbounds for i in eachindex(SimParticles.GhostPoints)
            GhostPoint = SimParticles.GhostPoints[i]
            iszero(GhostPoint) && continue
            GhostCell = f(SimKernel, GhostPoint)
            Cells = SVector{N,Int}(ntuple(Index -> FindCellIndex(UniqueCellsView, GhostCell + FullStencil[Index]), Val(N)))
            push!(Cache.GhostIndices, i)
            push!(Cache.NeighborCells, Cells)
        end
        return nothing
    end

    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                               SimConstants, ParticleRanges, UniqueCellsView,
                               SimParticles, bᵧ, Aᵧ, ::Nothing = nothing) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}

        @unpack Position, Density, GhostPoints, GhostNormals = SimParticles
        ParticleType = SimParticles.Type

        FullStencil = ConstructStencil(Val(Dimensions))

        @inbounds ForEachParticle!(eachindex(GhostPoints)) do iter
            GhostPoint = GhostPoints[iter]

            if !iszero(GhostPoint)
                # zero‐initialize per‐ghost accumulators
                b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
                A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}

                # compute and accumulate into the locals
                GhostCellIndex = f(SimKernel, GhostPoints[iter])
                @inbounds for offset ∈ FullStencil
                    SCellIndex = GhostCellIndex + offset
                    NeighborIdx = FindCellIndex(UniqueCellsView, SCellIndex)

                    StartIndex_       = ParticleRanges[NeighborIdx]
                    EndIndex_         = ParticleRanges[NeighborIdx + 1] - 1

                    for j in StartIndex_:EndIndex_
                        b_acc, A_acc = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                        Position, Density, ParticleType,
                                                        GhostPoints, iter, j, b_acc, A_acc)
                    end
                end

                # write out once
                bᵧ[iter] = b_acc
                Aᵧ[iter] = A_acc
            end
        end

        return nothing
    end

    function NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges,
                               UniqueCellsView, SimParticles, bᵧ, Aᵧ,
                               Cache::MDBCNeighborCache)
        @unpack Position, Density, GhostPoints = SimParticles
        ParticleType = SimParticles.Type
        @inbounds ForEachParticle!(eachindex(Cache.GhostIndices)) do GhostIndex
            i = Cache.GhostIndices[GhostIndex]
            b_acc = zero(bᵧ[i])
            A_acc = zero(Aᵧ[i])
            for Cell in Cache.NeighborCells[GhostIndex]
                for j in ParticleRanges[Cell]:(ParticleRanges[Cell + 1] - 1)
                    b_acc, A_acc = ComputeInteractionsMDBC!(
                        SimKernel, SimMetaData, SimConstants, Position, Density,
                        ParticleType, GhostPoints, i, j, b_acc, A_acc,
                    )
                end
            end
            bᵧ[i] = b_acc
            Aᵧ[i] = A_acc
        end
        return nothing
    end

    # The previous generic `ComputeInteractions!` implementation was unused
    # in favour of the per-particle variants (ComputeInteractionsPerParticle! etc.).
    # It has been removed to reduce code size and avoid dead code.

    @inline function compute_kernel_output_local(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                                 kernel_acc, kernel_grad_acc, SimKernel,
                                                 q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                                 B<:MDBCMode,
                                                                 L<:LogMode}
        return kernel_acc, kernel_grad_acc
    end

    @inline function compute_kernel_output_local(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L},
                                                 kernel_acc, kernel_grad_acc, SimKernel,
                                                 q, ∇ᵢWᵢⱼ) where {D,T,S<:ShiftingMode,
                                                                 B<:MDBCMode,
                                                                 L<:LogMode}
        Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
        return kernel_acc + Wᵢⱼ, kernel_grad_acc + ∇ᵢWᵢⱼ
    end

    @inline function ShiftCScale(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                 M0, RhoI, RhoJ) where {D,T,S<:ShiftingMode,
                                                       B<:MDBCMode,
                                                       L<:LogMode}
        return (M0 / RhoJ) * (M0 / RhoJ)
    end

    @inline function ShiftCScale(::SimulationMetaData{D,T,S,StoreKernelOutput,B,L},
                                 M0, RhoI, RhoJ) where {D,T,S<:ShiftingMode,
                                                       B<:MDBCMode,
                                                       L<:LogMode}
        return (M0 / RhoJ) * (M0 / RhoI)
    end

    # Carry the neighbor loop's valid-index guarantee through this extra helper
    # layer; direct callers still receive normal bounds checks.
    Base.@propagate_inbounds function ComputeInteractionsPerParticleNoShiftingCore!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j) where {D,T,
                                                                     K<:KernelOutputMode,
                                                                     B<:MDBCMode,
                                                                     L<:LogMode,
                                                                     SDD<:SPHDensityDiffusion,
                                                                     SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]
            ρᵢ⁻¹ = InvDensity[i]
            ρⱼ⁻¹ = InvDensity[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            Vⱼ = m₀ * ρⱼ⁻¹
            dρdt⁺ = -ρᵢ * Vⱼ * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            ρᵢρⱼ_inv = ρᵢ⁻¹ * ρⱼ⁻¹
            Pfac = (Pᵢ + Pⱼ) * ρᵢρⱼ_inv
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,K,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j) where {D,T,
                                                                     K<:KernelOutputMode,
                                                                     B<:MDBCMode,
                                                                     L<:LogMode,
                                                                     SDD<:SPHDensityDiffusion,
                                                                     SV<:SPHViscosity}
        return ComputeInteractionsPerParticleNoShiftingCore!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
            SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
            dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, i, j,
        )
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, i, j) where {D,T,B<:MDBCMode,L<:LogMode,
                                        SDD<:SPHDensityDiffusion,
                                        SV<:SPHViscosity}
        dρdt_acc, acc_acc, _, _ = ComputeInteractionsPerParticleNoShiftingCore!(
            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants,
            SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
            dρdt_acc, acc_acc, nothing, nothing, i, j,
        )

        return dρdt_acc, acc_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,K,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc,
        shift_r_acc, i, j) where {D,T,S<:ShiftingMode,
                                  K<:KernelOutputMode,
                                  B<:MDBCMode,
                                  L<:LogMode,
                                  SDD<:SPHDensityDiffusion,
                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]
            ρᵢ⁻¹ = InvDensity[i]
            ρⱼ⁻¹ = InvDensity[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            Vⱼ = m₀ * ρⱼ⁻¹
            dρdt⁺ = -ρᵢ * Vⱼ * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            ρᵢρⱼ_inv = ρᵢ⁻¹ * ρⱼ⁻¹
            Pfac = (Pᵢ + Pⱼ) * ρᵢρⱼ_inv
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

            acc_acc += dvdt⁺ + visc_term

            kernel_acc, kernel_grad_acc = compute_kernel_output_local(SimMetaData, kernel_acc, kernel_grad_acc, SimKernel, q, ∇ᵢWᵢⱼ)

            MotionLimiterCondition = ParticleType[i]==Fluid && ParticleType[j]==Fluid #MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            Vᵢ = m₀ * ρᵢ⁻¹
            shift_c_acc += Vⱼ * Wᵢⱼ * Vᵢ * ∇ᵢWᵢⱼ * MotionLimiterCondition
            shift_r_acc += Vⱼ * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_acc, acc_acc, kernel_acc, kernel_grad_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsPerParticle!(
        SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
        SimMetaData::SimulationMetaData{D,T,S,NoKernelOutput,B,L}, SimConstants,
        SimParticles, Position, Density, InvDensity, Pressure, Velocity, ParticleType,
        dρdt_acc, acc_acc, shift_c_acc, shift_r_acc, i, j) where {D,T,
                                                                  S<:ShiftingMode,
                                                                  B<:MDBCMode,
                                                                  L<:LogMode,
                                                                  SDD<:SPHDensityDiffusion,
                                                                  SV<:SPHViscosity}
        @unpack m₀, dx = SimConstants
        @unpack h⁻¹, H², h = SimKernel

        xᵢⱼ = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
        if xᵢⱼ² <= H²
            dᵢⱼ² = xᵢⱼ²
            dᵢⱼ = sqrt(dᵢⱼ²)
            q = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

            ρᵢ = Density[i]
            ρⱼ = Density[j]
            ρᵢ⁻¹ = InvDensity[i]
            ρⱼ⁻¹ = InvDensity[j]

            vᵢ = Velocity[i]
            vⱼ = Velocity[j]
            vᵢⱼ = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            Vⱼ = m₀ * ρⱼ⁻¹
            dρdt⁺ = -ρᵢ * Vⱼ * density_symmetric_term

            Dᵢ, _ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

            dρdt_acc += dρdt⁺ + Dᵢ

            Pᵢ = Pressure[i]
            Pⱼ = Pressure[j]
            ρᵢρⱼ_inv = ρᵢ⁻¹ * ρⱼ⁻¹
            Pfac = (Pᵢ + Pⱼ) * ρᵢρⱼ_inv
            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺ = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

            acc_acc += dvdt⁺ + visc_term

            MotionLimiterCondition = ParticleType[i]==Fluid && ParticleType[j]==Fluid #MotionLimiterValue(eltype(ρᵢ), ParticleType[i]) * MotionLimiterValue(eltype(ρᵢ), ParticleType[j])
            shift_c_acc += Vⱼ * Wᵢⱼ * Vⱼ * ∇ᵢWᵢⱼ * MotionLimiterCondition
            shift_r_acc += Vⱼ * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MotionLimiterCondition
        end

        return dρdt_acc, acc_acc, shift_c_acc, shift_r_acc
    end

    Base.@propagate_inbounds function ComputeInteractionsMDBC!(SimKernel, SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}, SimConstants, Position, Density, ParticleType, GhostPoints, i, j,
                                                              b_acc = zero(SVector{Dimensions + 1, FloatType}),
                                                              A_acc = zero(SMatrix{Dimensions + 1, Dimensions + 1, FloatType})) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
        @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
        
        @unpack h⁻¹, h, η², H², αD = SimKernel 

        DimensionsPlus = Dimensions + 1
        # ᵢ is ghost node! ⱼ is fluid node

        if ParticleType[j] == Fluid

            xᵢⱼ  = GhostPoints[i] - Position[j]

            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(abs(xᵢⱼ²))
                q = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
        
                ρⱼ = Density[j]

        
                Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)

                ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ = m₀ / ρⱼ
        
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ
        
                bΔ  = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

                # Filling the Aᵧ matrix is done in column-major order
                xⱼᵢ = -xᵢⱼ
                first_column = [VⱼWᵢⱼ; Vⱼ * ∇ᵢWᵢⱼ]
                AΔ = SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                    first_column...,
                    ((xⱼᵢ * first_column')')...
                )
                # Rejected neighbors leave the accumulators untouched.
                b_acc += bΔ
                A_acc += AΔ
            end
        end
        
    
        return b_acc, A_acc
    end

    function ApplyMDBCBeforeHalf!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, _args...) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
        return nothing
    end

    # UniqueCells[1] is the neighbor-search sentinel; only the remaining cells
    # represent physical grid geometry.
    @inline function PhysicalCellView(UniqueCells, IndexCounter)
        return view(UniqueCells, 2:IndexCounter)
    end

    @inline function PrepareGridExportData(::Val{true}, ParticleRanges, IndexCounter, NeighborCellLists)
        cell_particle_counts = ComputeCellParticleCounts(
            ParticleRanges,
            IndexCounter,
        )
        cell_neighbor_counts = ComputeCellNeighborCounts(
            ParticleRanges,
            NeighborCellLists,
            IndexCounter,
        )
        # NeighborCellLists stores indices in the full sentinel-inclusive space,
        # so compute first and remove the sentinel only at the output boundary.
        PhysicalCellIndices = 2:IndexCounter
        return view(cell_particle_counts, PhysicalCellIndices),
               view(cell_neighbor_counts, PhysicalCellIndices)
    end

    @inline function PrepareGridExportData(::Val{false}, ParticleRanges, IndexCounter, NeighborCellLists)
        return nothing, nothing
    end

    function ApplyMDBCBeforeHalf!(SimMetaData::SimulationMetaData{D,T,S,K,SimpleMDBC,L},
                                  SimKernel, SimConstants, SimParticles,
                                  ParticleRanges, UniqueCells, MDBCNeighbors = nothing
                                 ) where {D,T,S<:ShiftingMode,K<:KernelOutputMode,L<:LogMode}
        @no_escape begin
            DimensionsPlus = D + 1
            bᵧ = @alloc(SVector{DimensionsPlus, T}, length(SimParticles.Position))
            Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, T, DimensionsPlus*DimensionsPlus}, length(SimParticles.Position))

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            @timeit SimMetaData.HourGlass "02 NeighborLoopMDBC!" NeighborLoopMDBC!(
                SimKernel, SimMetaData, SimConstants, ParticleRanges,
                UniqueCellsView, SimParticles, bᵧ, Aᵧ, MDBCNeighbors,
            )
            @timeit SimMetaData.HourGlass "03 ApplyMDBCCorrection" ApplyMDBCCorrection(
                SimConstants,
                SimParticles,
                bᵧ,
                Aᵧ,
            )
        end

        return nothing
    end

    function ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)

        Position    = SimParticles.Position
        Density     = SimParticles.Density
        GhostPoints = SimParticles.GhostPoints

        ρ₀ = SimConstants.ρ₀
        T = eltype(Density)
        DetTolerance = T(1e-3)
        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @simd ivdep for i in eachindex(Position)
            A = Aᵧ[i]

            # Since Aᵧ is not reset anymore, we need to check if it is zero
            if !iszero(GhostPoints[i])
                if abs(det(A)) >= DetTolerance
                        GhostPointDensity = A \ bᵧ[i]
                        diff = Position[i] - GhostPoints[i]
                        v1   = first(GhostPointDensity) + sum(GhostPointDensity[j+1] * diff[j] for j in eachindex(diff))
                        Density[i] = isnan(v1) ? ρ₀ : v1
                elseif first(A) > zero(T)
                        v = first(bᵧ[i]) / first(A)
                        Density[i] = isnan(v) ? ρ₀ : v
                end
            end
        end
    end
    
    function GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)
        # Assuming group markers are sequential
        MotionDefinition = Vector{Union{Nothing, MotionDetails{Dimensions, FloatType}}}(undef, maximum(SimParticles.GroupMarker))

        for geom in SimGeometry
            group_marker = geom.GroupMarker
            if geom.Motion !== nothing
                MotionDefinition[group_marker] = geom.Motion
            else
                MotionDefinition[group_marker] = nothing
            end
        end
        if !any(!isnothing, MotionDefinition)
            MotionDefinition = nothing
        end
        return MotionDefinition
    end


    # Per-particle local Δx removed: use single scalar `SimMetaData.Δx`.

    @inbounds function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                                      SimConstants, SimParticles, FullStencil,
                                      ParticleRanges, UniqueCells, CellListIndices,
                                      SortingScratchSpace,
                                      NeighborCellLists, dρdtI, Velocityₙ⁺,
                                      Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ,
                                      MotionDefinition::Union{
                                          Nothing,
                                          AbstractVector{
                                              Union{
                                                  Nothing,
                                                  MotionDetails{Dimensions, FloatType},
                                              },
                                          },
                                      }, OutputCallback) where {
                                                Dimensions, FloatType, SMode, KMode,
                                                BMode, LMode,
                                                SDD<:SPHDensityDiffusion,
                                                SV<:SPHViscosity}
        ParticleType   = SimParticles.Type
        ParticleMarker = SimParticles.GroupMarker
        GhostPoints    = hasproperty(SimParticles, :GhostPoints) ? SimParticles.GhostPoints : nothing
        GhostNormals   = hasproperty(SimParticles, :GhostNormals) ? SimParticles.GhostNormals : nothing

        ProposedDt = SimMetaData.CurrentTimeStep
        TimeSteppingMode = SimMetaData.TimeSteppingMode

        @no_escape begin
            AccelerationNormSquared = @alloc(FloatType, length(SimParticles.Position))
            # Reciprocal densities of the state each force evaluation reads.
            InvDensity = @alloc(FloatType, length(SimParticles.Position))
            MDBCNeighbors = MakeMDBCNeighborCache(SimMetaData, SimParticles)
            InteractionWorkspace = MakeInteractionWorkspace(SimMetaData, SimParticles.Position)

            @timeit SimMetaData.HourGlass "00 Initialize Neighbor Data" begin
                @timeit SimMetaData.HourGlass "01 UpdateNeighbors!" SimMetaData.IndexCounter = UpdateNeighbors!(
                    SimParticles,
                    SimKernel.H⁻¹,
                    SortingScratchSpace,
                    ParticleRanges,
                    UniqueCells,
                    CellListIndices,
                )
                UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
                @timeit SimMetaData.HourGlass "02 BuildNeighborCellLists!" BuildNeighborCellLists!(
                    NeighborCellLists,
                    FullStencil,
                    UniqueCellsView,
                    ParticleRanges,
                )
                # UpdateNeighbors! sorts SimParticles. Reset the displacement
                # reference so separately allocated predictor data remains aligned.
                copyto!(Positionₙ⁺, SimParticles.Position)
                SimMetaData.Δx = zero(FloatType)
                UpdateMDBCNeighborCache!(MDBCNeighbors, SimKernel, SimParticles, UniqueCellsView)
            end

            if TimeSteppingMode isa SingleNeighborTimeStepping
                @timeit SimMetaData.HourGlass "00 Init MDBC"                              ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells, MDBCNeighbors)
                @timeit SimMetaData.HourGlass "00a Init Pressure"                         Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
                @timeit SimMetaData.HourGlass "00b Init NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                    SimConstants, SimParticles, ParticleRanges, CellListIndices,
                    NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                )
            end

            NextOutputTime = next_output_time(SimMetaData)
            while SimMetaData.TotalTime < SimMetaData.SimulationTime
                @timeit SimMetaData.HourGlass "00 Simulation Step" begin
                    RefreshedSingleNeighborDerivative = false
                    if !isfinite(ProposedDt) || ProposedDt <= zero(ProposedDt)
                        throw(DomainError(ProposedDt, "the proposed simulation timestep must be finite and positive"))
                    end

                    RemainingTime = SimMetaData.SimulationTime - SimMetaData.TotalTime
                    # Output deadlines must not alter the integration sequence. Only
                    # shorten the final physical step so the run ends at SimulationTime.
                    dt = min(ProposedDt, RemainingTime)
                    if SimMetaData.TotalTime + dt <= SimMetaData.TotalTime
                        throw(DomainError(dt, "the simulation timestep is too small to advance TotalTime"))
                    end
                    IsFinalStep = dt == RemainingTime
                    dt₂ = dt * 0.5
                    @timeit SimMetaData.HourGlass "01 Calculate IndexCounter"  begin

                        @timeit SimMetaData.HourGlass "01 UpdateΔx!" SimMetaData.Δx = UpdateΔx!(
                            SimMetaData.Δx,
                            Positionₙ⁺,
                            SimParticles.Position,
                        )
                        ShouldRebuild = SimMetaData.Δx >= SimKernel.h

                        # This is a motion heuristic, not a guaranteed validity
                        # bound: H-wide cells and a one-cell stencil have no
                        # extra search margin. Pairs in formerly nonadjacent
                        # cells can enter support before Δx reaches h. See the
                        # neighbor-reuse limitation documented in README.md.
                        if ShouldRebuild
                            @timeit SimMetaData.HourGlass "02 UpdateNeighbors!" SimMetaData.IndexCounter = UpdateNeighbors!(
                                SimParticles,
                                SimKernel.H⁻¹,
                                SortingScratchSpace,
                                ParticleRanges,
                                UniqueCells,
                                CellListIndices,
                            )
                            SimMetaData.Δx    = zero(eltype(dρdtI))
                            UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                            @timeit SimMetaData.HourGlass "03 BuildNeighborCellLists!" BuildNeighborCellLists!(
                                NeighborCellLists,
                                FullStencil,
                                UniqueCellsView,
                                ParticleRanges,
                            )
                            copyto!(Positionₙ⁺, SimParticles.Position)
                            UpdateMDBCNeighborCache!(MDBCNeighbors, SimKernel, SimParticles, UniqueCellsView)

                            # Single-neighbor stepping carries a midpoint derivative
                            # into the next predictor. Re-anchor it at this accepted
                            # full state after a real sort, not after an output event,
                            # so the arrays stay particle-aligned.
                            if TimeSteppingMode isa SingleNeighborTimeStepping
                                @timeit SimMetaData.HourGlass "03a Rebuild MDBC" ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells, MDBCNeighbors)
                                @timeit SimMetaData.HourGlass "03b Rebuild Pressure" Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
                                @timeit SimMetaData.HourGlass "03c Rebuild NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                                    SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                    SimConstants, SimParticles, ParticleRanges, CellListIndices,
                                    NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                                )
                                RefreshedSingleNeighborDerivative = true
                            end
                        end
                    end

                    if TimeSteppingMode isa SingleNeighborTimeStepping &&
                       NeedsSingleNeighborCorrection(
                           SimMetaData.Iteration,
                           RefreshedSingleNeighborDerivative,
                       )
                        @timeit SimMetaData.HourGlass "04 Periodic Single-Neighbor Correction" begin
                            @timeit SimMetaData.HourGlass "01 MDBC" ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells, MDBCNeighbors)
                            @timeit SimMetaData.HourGlass "02 Pressure" Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)
                            @timeit SimMetaData.HourGlass "03 NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                                SimConstants, SimParticles, ParticleRanges, CellListIndices,
                                NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                            )
                        end
                        RefreshedSingleNeighborDerivative = true
                    end

                    @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                    if TimeSteppingMode isa SymplecticTimeStepping
                        @timeit SimMetaData.HourGlass "02 Apply MDBC before Pressure"             ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells, MDBCNeighbors)
                        @timeit SimMetaData.HourGlass "03 Pressure"                               Pressure!(SimParticles.Pressure, SimParticles.Density, SimConstants)

                        @timeit SimMetaData.HourGlass "04 First NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                        )

                        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

                        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

                        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                        @timeit SimMetaData.HourGlass "07 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
                        @timeit SimMetaData.HourGlass "08 Second NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    else
                        @timeit SimMetaData.HourGlass "02 Apply MDBC before Half TimeStep"       ApplyMDBCBeforeHalf!(SimMetaData, SimKernel, SimConstants, SimParticles, ParticleRanges, UniqueCells, MDBCNeighbors)

                        @timeit SimMetaData.HourGlass "03 Update To Half TimeStep"               HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)

                        @timeit SimMetaData.HourGlass "04 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, ParticleType)

                        @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(SimParticles, dt₂, MotionDefinition, SimMetaData)

                        @timeit SimMetaData.HourGlass "05 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺, SimConstants)
                        @timeit SimMetaData.HourGlass "06 NeighborLoop" EvaluateInteractions!(InteractionWorkspace,
                            SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                            SimConstants, SimParticles, ParticleRanges, CellListIndices,
                            NeighborCellLists, dρdtI, SimParticles.Acceleration, ∇Cᵢ, ∇◌rᵢ, AccelerationNormSquared, InvDensity = InvDensity,
                            Position = Positionₙ⁺,
                            Density = ρₙ⁺,
                            Velocity = Velocityₙ⁺,
                        )
                    end

                    @timeit SimMetaData.HourGlass "07 Final Density"                         DensityEpsi!(SimParticles.Density, dρdtI, ρₙ⁺, dt)

                    @timeit SimMetaData.HourGlass "08 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(SimParticles.Density, SimConstants.ρ₀, ParticleType)

                    @timeit SimMetaData.HourGlass "09 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt)

                    @timeit SimMetaData.HourGlass "10 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)
                    if IsFinalStep
                        SimMetaData.TotalTime = SimMetaData.SimulationTime
                    end
                    push!(SimMetaData.TimeSteps, dt)

                    @timeit SimMetaData.HourGlass "11 Update TimeStep"                       ProposedDt = UpdateTimeStep(AccelerationNormSquared, SimConstants, SimKernel)

                end

                # A physical step may cross several requested output deadlines.
                # Emit each frame from the same completed state without restarting
                # or otherwise mutating the integrator.
                while SimMetaData.TotalTime >= NextOutputTime
                    SimMetaData.OutputIterationCounter += 1
                    OutputCallback()
                    NextOutputTime >= SimMetaData.SimulationTime && break
                    NextOutputTime = next_output_time(SimMetaData)
                end
            end
        end
        
        return nothing
    end
    
    function ShowPerformanceReport(IO, HourGlass)
        DetailedIO = IOContext(IO, :limit => false, :displaysize => (-1, -1))
        println(DetailedIO, "\nPerformance profile, sorted by elapsed time:")
        show(DetailedIO, HourGlass; sortby=:time, complement=true, gc=true)
        println(DetailedIO, "\n\nRecorded sections, globally sorted by allocations:")
        show(DetailedIO, flatten(HourGlass); sortby=:allocations, complement=false, gc=true)
        println(DetailedIO)

        return nothing
    end

    ShowPerformanceReport(HourGlass) = ShowPerformanceReport(stdout, HourGlass)

    function ValidateOutputSchedule(OutputTimes::Real)
        if !isfinite(OutputTimes) || OutputTimes <= zero(OutputTimes)
            throw(ArgumentError("OutputTimes must be a finite, positive interval"))
        end
        return nothing
    end

    function ValidateOutputSchedule(OutputTimes::AbstractVector)
        PreviousTime = nothing
        for OutputTime in OutputTimes
            if !isfinite(OutputTime) || OutputTime <= zero(OutputTime)
                throw(ArgumentError("each requested output time must be finite and positive"))
            end
            if PreviousTime !== nothing && OutputTime <= PreviousTime
                throw(ArgumentError("requested output times must be strictly increasing"))
            end
            PreviousTime = OutputTime
        end
        return nothing
    end

    function FinalizeSimulationOutput!(SimMetaData, SimLogger, output; log_status::String="finished")
        @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

        ShowPerformanceReport(SimMetaData.HourGlass)

        AutoOpenParaview(SimMetaData, output.variable_names)

        FinalizeLog!(SimMetaData, SimLogger; status=log_status)
        AutoOpenLogFile(SimLogger, SimMetaData)

        return nothing
    end

    """
        RunWithSimulationFinalizer!(RunFunction, SimMetaData, SimLogger, output)

    Run the simulation body with a single cleanup boundary around it. Output is
    drained on normal completion, Ctrl+C, process exit, and solver/writer errors.
    Non-interrupt errors are rethrown after cleanup.
    """
    function RunWithSimulationFinalizer!(RunFunction, SimMetaData, SimLogger, output)
        OutputFinalized = Ref(false)

        function FinalizeOnce!(Status)
            if OutputFinalized[]
                return nothing
            end
            try
                FinalizeSimulationOutput!(SimMetaData, SimLogger, output; log_status=Status)
            catch CleanupError
                @error "Simulation cleanup failed" exception=(CleanupError, catch_backtrace())
            finally
                OutputFinalized[] = true
            end
            return nothing
        end

        atexit() do
            if !OutputFinalized[]
                @warn "Julia is exiting before the simulation completed; closing VTKHDF output, finalizing the log, and opening ParaView for the data written so far."
                FinalizeOnce!("stopped as Julia exited")
            end
        end

        Base.exit_on_sigint(false)

        # This is the smallest reliable Ctrl+C boundary for REPL/VS Code/terminal
        # execution: without catching `InterruptException`, Julia returns control
        # to the caller and the `atexit` hook is not guaranteed to run.
        try
            RunFunction(OutputFinalized)
        catch e
            if e isa InterruptException
                @warn "Simulation interrupted; closing VTKHDF output, finalizing the log, and opening ParaView for the data written so far."
                FinalizeOnce!("interrupted")
                return nothing
            end
            @warn "Simulation failed; closing VTKHDF output and finalizing the log before rethrowing the error."
            FinalizeOnce!("failed")
            rethrow()
        end

        return nothing
    end

    ###===
    function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        SimTimeStepping::TimeSteppingMode,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SMode,KMode,BMode,LMode,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        NumberOfPoints = length(SimParticles)

        SimMetaData.TimeSteppingMode = SimTimeStepping
        ValidateOutputSchedule(SimMetaData.OutputTimes)
        if !isfinite(SimMetaData.SimulationTime) || SimMetaData.SimulationTime < zero(FloatType)
            throw(ArgumentError("SimulationTime must be finite and non-negative"))
        end

        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimMetaData, SimParticles.Position)

        LoadMDBCNormals!(SimMetaData, SimParticles, ParticleNormalsPath)

        InitializeLog!(SimMetaData, SimLogger, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1 + 1) # +1 for the last particle, +1 for dummy entry
        # One slot is reserved for the sentinel cell used by the neighbor search.
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints + 1)
        CellListIndices        = zeros(Int, NumberOfPoints)
        FullStencil            = ConstructStencil(Val(Dimensions))
        NeighborCellLists      = PackedNeighborCellLists(length(UniqueCells))
        SortingScratchSpace = NeighborSortScratch(NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

        MotionDefinition = GenerateMotionDetails(SimParticles, SimGeometry, Dimensions, FloatType)

        SimMetaData.CurrentTimeStep = SimConstants.CFL * (SimKernel.h / SimConstants.c₀)

        RunWithSimulationFinalizer!(SimMetaData, SimLogger, output) do OutputFinalized
            # The initial particle frame is the state supplied by the caller,
            # before neighbor sorting or derivative initialization.
            SimMetaData.OutputIterationCounter = 0
            output.enqueue_particles(SimMetaData.OutputIterationCounter)

            function SaveCurrentState!()
                @timeit SimMetaData.HourGlass "13 Save Particle Data"  begin
                    @timeit SimMetaData.HourGlass "01 Log Progress" LogStep!(SimMetaData, SimLogger)
                    @timeit SimMetaData.HourGlass "02 Prepare Grid Data" begin
                        UniqueCellsView = PhysicalCellView(UniqueCells, SimMetaData.IndexCounter)
                        cell_particle_counts, cell_neighbor_counts = PrepareGridExportData(
                            Val(SimMetaData.ExportGridCellParticleCounts),
                            ParticleRanges,
                            SimMetaData.IndexCounter,
                            NeighborCellLists,
                        )
                    end
                    @timeit SimMetaData.HourGlass "03 Queue Particle Snapshot" output.enqueue_particles(SimMetaData.OutputIterationCounter)
                    @timeit SimMetaData.HourGlass "04 Queue Grid Snapshot" output.enqueue_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, cell_particle_counts=cell_particle_counts, cell_neighbor_counts=cell_neighbor_counts)
                end

                return nothing
            end

            SimulationLoop(
                SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData,
                SimConstants, SimParticles, FullStencil, ParticleRanges,
                UniqueCells, CellListIndices, SortingScratchSpace,
                NeighborCellLists, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
                ∇Cᵢ, ∇◌rᵢ, MotionDefinition, SaveCurrentState!,
            )

            try
                FinalizeSimulationOutput!(SimMetaData, SimLogger, output)
            finally
                OutputFinalized[] = true
            end
        end
    end
    

end
