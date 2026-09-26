"""
CUDA kernels for the SPH time step.

All kernels use a *gather* formulation: the thread(s) of particle `i` loop
over the particles of the neighbouring cells and sum the contributions into
registers, then write the totals once. There are no atomics, no per thread
accumulation arrays and no reduction step.

Small cases do not have enough particles to fill the GPU with one thread per
particle, so the neighbour loop of a particle can be split over `K` lanes of
a warp (`K` a power of two up to 32): lane `k` visits every `K`-th candidate
of each cell row and the partial sums are combined with warp shuffles.

The CPU code evaluates every pair once and assigns the two halves of a pair
to `i` and `j`. For the built in models most terms are antisymmetric, but the
density diffusion term is deliberately not (`Dⱼ = -Dᵢ` is used as a cheap
approximation) and a user supplied model may not be either. To reproduce the
CPU results exactly the gather kernel reconstructs which particle of a pair
was the CPU's `i`: for pairs inside one cell the lower index, for pairs in
different cells the particle in the cell that is processed (the higher index,
because the CPU stencil only visits cells with a lower linear index).
"""
module GPUKernels

using CUDA
using StaticArrays
using LinearAlgebra

using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels
using ..SimulationEquations
using ..SimulationGeometry
using ..GPUCellGrid
using ..GPUReductions

export launch_interactions!, launch_mdbc!, launch_motion!, launch_half_step!, launch_final_step!,
       step_map, step_reduce, choose_lanes, ELEMENTWISE_THREADS

const ELEMENTWISE_THREADS = REDUCE_THREADS
const FULL_MASK = 0xffffffff

#---------------------------------------------------------------
# Helpers shared by the kernels
#---------------------------------------------------------------

"""
    choose_lanes(n; target = 4 * resident threads of the device) -> K

Number of warp lanes per particle for the gather kernels: 1 for large particle
counts, more for small ones so that at least `target` threads are launched.
Always a power of two between 1 and 32.
"""
function choose_lanes(n::Integer; target::Integer = 4 * CUDA.attribute(CUDA.device(),
                                                          CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * 2048)
    n <= 0 && return 1
    k = nextpow(2, cld(target, n))
    return clamp(k, 1, 32)
end

# Warp shuffle reduction of the `K` consecutive lanes of a particle. All
# lanes of the warp must execute this (no early return before it).
@inline function lanes_sum(x::T, ::Val{K}) where {T <: Real, K}
    s = K ÷ 2
    while s >= 1
        x += CUDA.shfl_down_sync(FULL_MASK, x, s, K)
        s ÷= 2
    end
    return x
end

@inline function lanes_sum(x::SVector{N, T}, ::Val{K}) where {N, T, K}
    return SVector{N, T}(ntuple(k -> lanes_sum(x[k], Val(K)), Val(N)))
end

@inline function lanes_sum(x::SMatrix{N, M, T, L}, ::Val{K}) where {N, M, T, L, K}
    return SMatrix{N, M, T, L}(ntuple(k -> lanes_sum(x[k], Val(K)), Val(L)))
end

@inline lanes_sum(x, ::Val{1}) = x

"""
Prescribed motion of `Moving` particles (mirrors `ProgressMotion` of the CPU
code). `motion` is a NamedTuple of device arrays indexed by group marker.
"""
@inline function apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    @inbounds if ParticleType[i] == Moving
        g = Int(GroupMarker[i])
        if motion.has[g]
            start    = motion.start[g]
            duration = motion.duration[g]
            ShouldMove = (start <= TotalTime) & (TotalTime <= start + duration)
            v = motion.velocity[g] * motion.direction[g] * ShouldMove
            Velocity[i] = v
            Position[i] += v * dt₂
        end
    end
    return nothing
end

@inline function limit_density(ρ, ρ₀, MotionLimiter)
    return ((ρ < ρ₀) & !Bool(MotionLimiter)) ? ρ₀ : ρ
end

#---------------------------------------------------------------
# Prescribed motion of moving bodies (only launched when a case has them)
#---------------------------------------------------------------

function motion_kernel!(Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime, n::Int32)
    i = thread_index()
    i > n && return nothing
    apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    return nothing
end

function launch_motion!(Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) motion_kernel!(
        Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Particle interactions (gather)
#---------------------------------------------------------------

function interaction_kernel!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, ChunkID,
                             Position, Density, InvDensity, Pressure, Velocity, ParticleType, SimParticles,
                             CellStart, CellID, grid::CellGrid{D},
                             SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                             ::Val{FlagKernel}, ::Val{FlagShift}, ::Val{BoundaryForces}, ::Val{K},
                             n::Int32) where {D, FlagKernel, FlagShift, BoundaryForces, K}
    t    = thread_index()
    i    = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)
    valid = i <= n

    (; m₀, dx)  = SimConstants
    (; h⁻¹, H²) = SimKernel

    T = eltype(eltype(Position))
    dρdt  = zero(T)
    acc   = zero(SVector{D, T})
    Wsum  = zero(T)
    ∇Wsum = zero(SVector{D, T})
    ∇C    = zero(SVector{D, T})
    ∇r    = zero(T)

    @inbounds if valid
        xᵢ  = Position[i]
        vᵢ  = Velocity[i]
        ρᵢ  = Density[i]
        ρᵢ⁻¹ = InvDensity[i]
        Pᵢ  = Pressure[i]
        MLᵢ = MotionLimiterValue(T, ParticleType[i])

        # Boundary particles only need the density rate; their acceleration is
        # never applied (MotionLimiter = 0). Skipping the momentum terms for them
        # is optional because the CPU code includes their acceleration in the
        # force based time step criterion.
        forces = BoundaryForces | (MLᵢ != zero(T))

        c      = CellID[i]
        own_lo = CellStart[c] + Int32(1)
        own_hi = CellStart[c + Int32(1)]

        for off in row_offsets(grid)
            row0 = c + off - Int32(1)
            jlo  = CellStart[row0] + Int32(1)
            jhi  = CellStart[row0 + Int32(3)]
            j = jlo + lane
            while j <= jhi
                if j != i
                    xᵢⱼ  = xᵢ - Position[j]
                    xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
                    if xᵢⱼ² <= H²
                        dᵢⱼ   = sqrt(xᵢⱼ²)
                        q     = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                        ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                        ρⱼ   = Density[j]
                        ρⱼ⁻¹ = InvDensity[j]
                        vⱼ   = Velocity[j]
                        vᵢⱼ  = vᵢ - vⱼ
                        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
                        dρdt += -ρᵢ * (m₀ * ρⱼ⁻¹) * density_symmetric_term

                        # Which particle of the pair was `i` on the CPU? Evaluate the
                        # models from that particle's point of view (branch free: the
                        # sign flip and the swaps of index and density are selects).
                        # The densities handed to the models are those of the arrays
                        # being evaluated (the predictor state in the corrector loop).
                        same_cell = (j >= own_lo) & (j <= own_hi)
                        i_first   = same_cell ? (i < j) : (i > j)
                        sgn  = i_first ? one(T) : -one(T)
                        ia   = i_first ? i : j
                        ja   = i_first ? j : i
                        ρa   = i_first ? ρᵢ : ρⱼ
                        ρb   = i_first ? ρⱼ : ρᵢ
                        ρa⁻¹ = i_first ? ρᵢ⁻¹ : ρⱼ⁻¹
                        ρb⁻¹ = i_first ? ρⱼ⁻¹ : ρᵢ⁻¹
                        D1, D2 = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants,
                                                           SimParticles, sgn * xᵢⱼ, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                           ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja, ParticleType)
                        dρdt += i_first ? D1 : D2

                        if forces
                            v1, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles,
                                                      sgn * xᵢⱼ, sgn * vᵢⱼ, sgn * ∇ᵢWᵢⱼ, xᵢⱼ²,
                                                      ρa, ρb, ρa⁻¹, ρb⁻¹, ia, ja)
                            visc  = sgn * v1

                            Pⱼ   = Pressure[j]
                            Pfac = (Pᵢ + Pⱼ) * (ρᵢ⁻¹ * ρⱼ⁻¹)
                            f_ab = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
                            dvdt = -m₀ * (Pfac + f_ab) * ∇ᵢWᵢⱼ
                            acc += dvdt + visc
                        end

                        if FlagKernel
                            Wsum  += SPHKernels.Wᵢⱼ(SimKernel, q)
                            ∇Wsum += ∇ᵢWᵢⱼ
                        end

                        if FlagShift
                            MLcond = MLᵢ * MotionLimiterValue(T, ParticleType[j])
                            ∇C += (m₀ * ρᵢ⁻¹) * ∇ᵢWᵢⱼ
                            # Sign convention follows the CPU code, see
                            # https://arxiv.org/abs/2110.10076
                            ∇r += (m₀ * ρⱼ⁻¹) * dot(-xᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
                        end
                    end
                end
                j += Int32(K)
            end
        end
    end

    if K > 1
        dρdt = lanes_sum(dρdt, Val(K))
        acc  = lanes_sum(acc, Val(K))
        if FlagKernel
            Wsum  = lanes_sum(Wsum, Val(K))
            ∇Wsum = lanes_sum(∇Wsum, Val(K))
        end
        if FlagShift
            ∇C = lanes_sum(∇C, Val(K))
            ∇r = lanes_sum(∇r, Val(K))
        end
    end

    @inbounds if valid & (lane == Int32(0))
        dρdtI[i]        = dρdt
        Acceleration[i] = acc
        if FlagKernel
            Kernel[i]         = Wsum
            KernelGradient[i] = ∇Wsum
        end
        if FlagShift
            ∇Cᵢ[i]  = ∇C
            ∇◌rᵢ[i] = ∇r
        end
        ChunkID[i] = Int(blockIdx().x)
    end
    return nothing
end

"""
    launch_interactions!(...; threads, lanes)

Launch the gather interaction kernel. `Position`, `Density`, `InvDensity`,
`Pressure` and `Velocity` are the arrays of the state being evaluated (the
half step arrays for the second neighbour loop); `InvDensity` holds the
precomputed reciprocals of `Density`. The viscosity and diffusion models
receive the densities and reciprocals of this state as arguments, like the
CPU code. Custom models additionally get the same arrays as the NamedTuple
`SimParticles`, so nothing in the kernel can read a stale state. `lanes` is
the number of warp lanes per particle (`Val`). With
`boundary_forces = Val(false)` the momentum terms are skipped for particles
with `MotionLimiter == 0`.
"""
function launch_interactions!(dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, ChunkID,
                              Position, Density, InvDensity, Pressure, Velocity, ParticleType,
                              CellStart, CellID, grid,
                              SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
                              FlagKernel::Val, FlagShift::Val; threads::Integer = 128, lanes::Val = Val(1),
                              boundary_forces::Val = Val(true))
    n = length(Position)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    SimParticles = (Position = Position, Density = Density, Velocity = Velocity, Pressure = Pressure,
                    Type = ParticleType)
    @cuda threads=threads blocks=cld(n * K, threads) interaction_kernel!(
        dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ, ChunkID,
        Position, Density, InvDensity, Pressure, Velocity, ParticleType, SimParticles,
        CellStart, CellID, grid,
        SimDensityDiffusion, SimViscosity, SimKernel, SimConstants,
        FlagKernel, FlagShift, boundary_forces, lanes, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# mDBC: ghost node interpolation and density correction
#---------------------------------------------------------------

# Accumulate the contributions of the fluid particles in the (1-based) index
# range `jlo:jhi` (every `K`-th starting at `lane`) to the ghost node `gp`.
@inline function mdbc_range(b, A, jlo::Int32, jhi::Int32, lane::Int32, ::Val{K}, gp::SVector{D, T},
                            Position, Density, ParticleType, SimKernel, m₀) where {K, D, T}
    (; h⁻¹, H²) = SimKernel
    DP = D + 1
    j = jlo + lane
    @inbounds while j <= jhi
        if ParticleType[j] == Fluid
            xᵢⱼ  = gp - Position[j]
            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(xᵢⱼ²)
                q   = dᵢⱼ * h⁻¹ # in [0, 2]: the guard above enforces xᵢⱼ² <= H² = (2h)²
                ρⱼ  = Density[j]

                Wᵢⱼ   = SPHKernels.Wᵢⱼ(SimKernel, q)
                ∇ᵢWᵢⱼ = ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ    = m₀ / ρⱼ
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ

                b += SVector{DP, T}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

                xⱼᵢ          = -xᵢⱼ
                first_column = SVector{DP, T}(VⱼWᵢⱼ, (Vⱼ * ∇ᵢWᵢⱼ)...)
                A += first_column * SVector{DP, T}(one(T), xⱼᵢ...)'
            end
        end
        j += Int32(K)
    end
    return b, A
end

# Visit the three by three (by three) cells around the ghost node. Ghost
# nodes may lie outside the particle grid, so every row is range checked.
@inline function mdbc_rows(b, A, gp::SVector{2, T}, lg, grid::CellGrid{2}, CellStart, Position,
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val) where {T}
    n1, n2 = grid.dims
    xlo = max(lg[1] - Int32(1), Int32(0))
    xhi = min(lg[1] + Int32(1), n1 - Int32(1))
    @inbounds if xlo <= xhi
        for dy in Int32(-1):Int32(1)
            ly = lg[2] + dy
            if (ly >= Int32(0)) & (ly < n2)
                row0 = Int32(1) + xlo + n1 * ly
                jlo  = CellStart[row0] + Int32(1)
                jhi  = CellStart[row0 + (xhi - xlo) + Int32(1)]
                b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, gp, Position, Density, ParticleType, SimKernel, m₀)
            end
        end
    end
    return b, A
end

@inline function mdbc_rows(b, A, gp::SVector{3, T}, lg, grid::CellGrid{3}, CellStart, Position,
                           Density, ParticleType, SimKernel, m₀, lane, lanes::Val) where {T}
    n1, n2, n3 = grid.dims
    xlo = max(lg[1] - Int32(1), Int32(0))
    xhi = min(lg[1] + Int32(1), n1 - Int32(1))
    @inbounds if xlo <= xhi
        for dz in Int32(-1):Int32(1)
            lz = lg[3] + dz
            if (lz >= Int32(0)) & (lz < n3)
                for dy in Int32(-1):Int32(1)
                    ly = lg[2] + dy
                    if (ly >= Int32(0)) & (ly < n2)
                        row0 = Int32(1) + xlo + n1 * (ly + n2 * lz)
                        jlo  = CellStart[row0] + Int32(1)
                        jhi  = CellStart[row0 + (xhi - xlo) + Int32(1)]
                        b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, gp, Position, Density, ParticleType,
                                          SimKernel, m₀)
                    end
                end
            end
        end
    end
    return b, A
end

function mdbc_kernel!(Density, Position, GhostPoints, ParticleType, CellStart, grid::CellGrid{D},
                      SimKernel, SimConstants, ::Val{K}, n::Int32) where {D, K}
    t    = thread_index()
    i    = (t - Int32(1)) ÷ Int32(K) + Int32(1)
    lane = (t - Int32(1)) % Int32(K)

    T  = eltype(eltype(Position))
    DP = D + 1
    (; m₀, ρ₀) = SimConstants

    b = zero(SVector{DP, T})
    A = zero(SMatrix{DP, DP, T, DP * DP})

    gp = zero(SVector{D, T})
    valid = false
    @inbounds if i <= n
        gp    = GhostPoints[i]
        valid = !iszero(gp)
    end

    @inbounds if valid
        cg = cell_coords(gp, SimKernel.H⁻¹)
        lg = ntuple(d -> cg[d] - grid.origin[d], Val(D))
        b, A = mdbc_rows(b, A, gp, lg, grid, CellStart, Position, Density, ParticleType, SimKernel, m₀,
                         lane, Val(K))
    end

    if K > 1
        b = lanes_sum(b, Val(K))
        A = lanes_sum(A, Val(K))
    end

    # Density correction (mirrors ApplyMDBCCorrection of the CPU code)
    # https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
    @inbounds if valid & (lane == Int32(0))
        if abs(det(A)) >= 1e-3
            sol  = A \ b
            diff = Position[i] - gp
            grad = SVector{D, T}(ntuple(k -> sol[k + 1], Val(D)))
            v1   = sol[1] + dot(grad, diff)
            Density[i] = isnan(v1) ? ρ₀ : v1
        elseif A[1, 1] > zero(T)
            v = b[1] / A[1, 1]
            Density[i] = isnan(v) ? ρ₀ : v
        end
    end
    return nothing
end

function launch_mdbc!(Density, Position, GhostPoints, ParticleType, CellStart, grid, SimKernel,
                      SimConstants; threads::Integer = 128, lanes::Val = Val(1))
    n = length(Density)
    n == 0 && return nothing
    K = typeof(lanes).parameters[1]
    @cuda threads=threads blocks=cld(n * K, threads) mdbc_kernel!(
        Density, Position, GhostPoints, ParticleType, CellStart, grid, SimKernel, SimConstants, lanes, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Half step (symplectic predictor) fused with density limiting, motion, the
# pressure of the half step density and its reciprocal
#---------------------------------------------------------------

function half_step_kernel!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, dt₂, TotalTime, SimConstants, n::Int32)
    i = thread_index()
    i > n && return nothing
    (; g, ρ₀, c₀) = SimConstants
    T = eltype(Density)
    @inbounds begin
        type = ParticleType[i]
        ML  = MotionLimiterValue(T, type)
        acc = Acceleration[i]
        acc += ConstructGravitySVector(acc, g * GravityFactorValue(T, type))
        Acceleration[i] = acc
        Positionₙ⁺[i]   = Position[i] + Velocity[i] * dt₂ * ML
        Velocityₙ⁺[i]   = Velocity[i] + acc * dt₂ * ML
        ρ = Density[i] + dρdtI[i] * dt₂
        ρ = limit_density(ρ, ρ₀, ML)
        ρₙ⁺[i] = ρ
        InvDensityₙ⁺[i] = inv(ρ)

        apply_motion!(i, Position, Velocity, ParticleType, GroupMarker, motion, dt₂, TotalTime)

        Pressure[i] = EquationOfStateGamma7(ρ, c₀, ρ₀)
    end
    return nothing
end

function launch_half_step!(Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
                           Position, Velocity, Acceleration, Density, dρdtI,
                           ParticleType, GroupMarker, motion, dt₂, TotalTime, SimConstants)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(n, ELEMENTWISE_THREADS) half_step_kernel!(
        Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, InvDensityₙ⁺, Pressure,
        Position, Velocity, Acceleration, Density, dρdtI,
        ParticleType, GroupMarker, motion, dt₂, TotalTime, SimConstants, Int32(n))
    return nothing
end

#---------------------------------------------------------------
# Final step: density limiting, density update, symplectic corrector, the
# pressure for the next step and the per step reduction (time step limits and
# displacement) for the next step.
#
# The corrector advances the position with the half step velocity `Velocityₙ⁺`
# (the velocity the second neighbour loop was evaluated at) times `dt`, which
# is the scheme of the CPU `FullTimeStep`.
#---------------------------------------------------------------

"""
Per particle quantities reduced once per step: the viscous time step term,
the force based time step term and the displacement since the last half step
(used to decide when the cell list must be rebuilt).
"""
@inline function step_map(i, Position, Velocity, Acceleration, Positionₙ⁺, h, η²)
    @inbounds begin
        r = Position[i]
        v = Velocity[i]
        a = Acceleration[i]
        posn = Positionₙ⁺[i]
    end
    return step_values(r, v, a, posn, h, η²)
end

@inline function step_values(r, v, a, posn, h, η²)
    visc = abs(h * dot(v, r) / (dot(r, r) + η²))
    dt1  = sqrt(h / norm(a))
    disp = norm(posn - r)
    return SVector(visc, dt1, disp)
end

@inline step_reduce(a::SVector{3, T}, b::SVector{3, T}) where {T} =
    SVector{3, T}(max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))

function final_step_kernel!(Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, dt, SimKernel, SimConstants,
                            partial, ::Val{FlagShift}, n::Int32) where {FlagShift}
    (; g, ρ₀, c₀) = SimConstants
    T = eltype(Density)
    acc_red = SVector{3, T}(zero(T), T(Inf), zero(T))

    i = thread_index()
    stride = blockDim().x * gridDim().x
    @inbounds while i <= n
        type = ParticleType[i]
        ML = MotionLimiterValue(T, type)

        # LimitDensityAtBoundary! followed by DensityEpsi!
        ρ    = limit_density(Density[i], ρ₀, ML)
        epsi = -(dρdtI[i] / ρₙ⁺[i]) * dt
        ρ   *= (2 - epsi) / (2 + epsi)
        Density[i]  = ρ
        Pressure[i] = EquationOfStateGamma7(ρ, c₀, ρ₀)

        # FullTimeStep
        acc = Acceleration[i]
        acc += ConstructGravitySVector(acc, g * GravityFactorValue(T, type))
        Acceleration[i] = acc
        v = Velocity[i] + acc * dt * ML
        Velocity[i] = v

        # Symplectic corrector: the position moves with the half step velocity
        vₙ⁺ = Velocityₙ⁺[i]
        x = Position[i]
        if FlagShift
            D     = length(v)
            A     = 2      # Value between 1 to 6 advised
            A_FST = 0      # zero for internal flows
            A_FSM = D      # 2d, 3d val different
            A_FSC = (∇◌rᵢ[i] - A_FST) / (A_FSM - A_FST)
            δxᵢ = A_FSC < 0 ? zero(v) : -A_FSC * A * SimKernel.h * norm(vₙ⁺) * dt * ∇Cᵢ[i]
            x += (vₙ⁺ * dt + δxᵢ) * ML
        else
            x += (vₙ⁺ * dt) * ML
        end
        Position[i] = x

        # Values for the time step and cell list update decision of the next step
        acc_red = step_reduce(acc_red, step_values(x, v, acc, Positionₙ⁺[i], SimKernel.h, SimKernel.η²))

        i += stride
    end

    block_reduce_store!(partial, acc_red, step_reduce)
    return nothing
end

"""
Launch the final step. Uses `red.nblocks` blocks in a grid stride loop so the
per block reduction results fit the reduction workspace; finish the
reduction with `finish_reduction(red, step_reduce, init)`.
"""
function launch_final_step!(Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
                            Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, dt, SimKernel, SimConstants,
                            red::ReductionWorkspace, FlagShift::Val)
    n = length(Position)
    n == 0 && return nothing
    @cuda threads=ELEMENTWISE_THREADS blocks=red.nblocks final_step_kernel!(
        Position, Velocity, Acceleration, Density, Pressure, dρdtI, ρₙ⁺, Positionₙ⁺,
        Velocityₙ⁺, ParticleType, ∇Cᵢ, ∇◌rᵢ, dt, SimKernel, SimConstants,
        red.partial, FlagShift, Int32(n))
    return nothing
end

end # module
