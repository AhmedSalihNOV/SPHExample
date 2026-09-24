using Test
using SPHExample
using StaticArrays
using LinearAlgebra: norm

# Requires InteractionReferenceCase and CheckedInteractionReference from
# interaction_reference.jl, which runtests.jl includes first.
@testset "symmetric pair evaluation matches the per-particle reference" begin
    SPH = SPHExample.SPHCellList
    Cases = (
        (2, Float64, WendlandC2(), ZeroDensityDiffusion(), ZeroViscosity(), false),
        (2, Float32, CubicSpline{Float32}(), LinearDensityDiffusion(), ArtificialViscosity(), true),
        (3, Float64, WendlandC2(), ComplexDensityDiffusion(), Laminar(), true),
        (3, Float32, CubicSpline{Float32}(), ZeroGravityLinearDensityDiffusion(), LaminarSPS(), false),
        (2, Float64, CubicSpline{Float64}(), ComplexDensityDiffusion(), ArtificialViscosity(), true),
        (3, Float64, WendlandC2(), LinearDensityDiffusion(), LaminarSPS(), false),
    )
    for (D, T, KernelModel, Diffusion, Viscosity, Midpoint) in Cases, Copies in (1, 23)
        @testset "$D dimensions, $T, $(typeof(Diffusion)), $(typeof(Viscosity)), $Copies clusters" begin
            Case = InteractionReferenceCase(Val(D), T, KernelModel, NoShifting, NoKernelOutput;
                                            Midpoint, Copies, Packed=Copies > 1)
            Expected = CheckedInteractionReference(Case, Diffusion, Viscosity, NoShifting, NoKernelOutput)
            (; Constants, Kernel, MetaData, Particles, ParticleRanges, CellListIndices,
               NeighborCellLists, Position, Density, Velocity, Pressure) = Case
            Count = length(Particles)
            Workspace = SPH.MakeInteractionWorkspace(MetaData, Position)
            @test Workspace isa SPH.SymmetricAccumulators
            # Each pair is summed once in cell order instead of once per
            # particle, so allow a few more ulps of the field scale than the
            # per-particle comparison; isolated particles stay exactly zero.
            Tolerance = 32eps(T)
            Close(Actual, Reference) = all(isapprox.(Actual, Reference; rtol=Tolerance,
                                                     atol=Tolerance * maximum(norm, Reference)))
            SerialWorkspace = SPH.SymmetricAccumulators(Vector{T}[], Vector{SVector{D,T}}[])
            for Slots in (Workspace, SerialWorkspace)
                DensityRate = fill(T(-9), Count)
                Acceleration = fill(SVector{D,T}(ntuple(_ -> T(-9), D)), Count)
                AccelerationNormSquared = fill(T(-9), Count)
                ShiftC = similar(Acceleration)
                ShiftR = similar(DensityRate)
                OriginalState = deepcopy((Position, Density, Velocity, Pressure))
                SPH.EvaluateInteractions!(
                    Slots, Diffusion, Viscosity, Kernel, MetaData, Constants, Particles,
                    ParticleRanges, CellListIndices, NeighborCellLists, DensityRate,
                    Acceleration, ShiftC, ShiftR, AccelerationNormSquared;
                    Position, Density, Pressure, Velocity,
                )
                @test Close(DensityRate, Expected.DensityRate)
                @test Close(Acceleration, Expected.Acceleration)
                @test AccelerationNormSquared ≈ sum.(abs2, Acceleration) rtol=8eps(T)
                @test OriginalState == (Position, Density, Velocity, Pressure)
                @test any(X -> !iszero(X), Acceleration)
                Isolated = findfirst(==(12), Particles.ID)
                @test iszero(DensityRate[Isolated])
                @test iszero(Acceleration[Isolated])
            end
        end
    end

    # Modes with per-particle kernel or shifting outputs keep the per-particle loop.
    for (Shifting, KernelOutput) in ((NoShifting, StoreKernelOutput),
                                     (PlanarShifting, NoKernelOutput),
                                     (PlanarShifting, StoreKernelOutput))
        Case = InteractionReferenceCase(Val(2), Float64, WendlandC2(), Shifting, KernelOutput)
        @test SPH.MakeInteractionWorkspace(Case.MetaData, Case.Position) === nothing
    end
end
