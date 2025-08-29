using PrecompileTools
using StaticArrays
using StructArrays

@setup_workload begin
    D = 2
    T = Float64
    sc = SimulationConstants{T}()
    ker = SPHKernelInstance{D, T}(WendlandC2(); dx = sc.dx)
    pos = [SVector{D, T}(0, 0), SVector{D, T}(1, 0)]
    vel = [SVector{D, T}(0, 0), SVector{D, T}(0, 0)]
    acc = [SVector{D, T}(0, 0), SVector{D, T}(0, -9.81)]
    dens = fill(sc.ρ₀, 2)
    press = zeros(T, 2)
    motion = ones(T, 2)
    cells = fill(CartesianIndex{D}(0, 0), 2)
    chunks = zeros(Int, 2)
    kernel = zeros(T, 2)
    kernelgrad = fill(SVector{D, T}(0, 0), 2)
    grav = zeros(T, 2)
    boundary = zeros(UInt8, 2)
    id = collect(1:2)
    types = fill(Fluid, 2)
    group = ones(UInt, 2)
    ghostp = fill(SVector{D, T}(0, 0), 2)
    ghostn = fill(SVector{D, T}(0, 0), 2)
    particles = StructArray((
        Cells = cells,
        ChunkID = chunks,
        Kernel = kernel,
        KernelGradient = kernelgrad,
        Position = pos,
        Acceleration = acc,
        Velocity = vel,
        Density = dens,
        Pressure = press,
        GravityFactor = grav,
        MotionLimiter = motion,
        BoundaryBool = boundary,
        ID = id,
        Type = types,
        GroupMarker = group,
        GhostPoints = ghostp,
        GhostNormals = ghostn,
    ))
    dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ =
        AllocateSupportDataStructures(particles.Position)
    meta = SimulationMetaData{D, T}(
        SimulationName = "precompile",
        SaveLocation = ".",
        SimulationTime = T(1.0),
    )
    threaded = AllocateThreadedArrays(meta, particles, dρdtI, ∇Cᵢ, ∇◌rᵢ)
    ParticleRanges = zeros(Int, length(particles) + 2)
    UniqueCells = fill(CartesianIndex{D}(0, 0), length(particles))
    CellDict = Dict{CartesianIndex{D}, Int}()
    _, SortingScratch = Base.Sort.make_scratch(
        nothing,
        eltype(particles),
        length(particles),
    )
    inv_cutoff = ker.H⁻¹
    xdiff = pos[1] - pos[2]
    vdiff = vel[1] - vel[2]
    Δt_val = T(1e-3)
end

@compile_workload begin
    Δt(pos, vel, acc, sc, ker)
    AllocateSupportDataStructures(particles.Position)
    Pressure!(press, dens, sc)
    EquationOfStateGamma7(dens[1], sc.c₀, sc.ρ₀)
    EquationOfState(dens[1], sc.c₀, sc.γ, sc.ρ₀)
    DensityEpsi!(dens, dρdtI, ρₙ⁺, Δt_val)
    LimitDensityAtBoundary!(dens, sc.ρ₀, motion)
    ConstructGravitySVector(SVector{D, T}(0, 0), sc.g)
    Estimate7thRoot(T(2))
    InverseHydrostaticEquationOfState(sc.ρ₀, press[1], sc.Cb⁻¹)
    Wᵢⱼ(ker, T(0.5))
    grad = ∇Wᵢⱼ(ker, T(0.5), xdiff)
    tensile_correction(
        ker,
        press[1],
        dens[1],
        press[2],
        dens[2],
        T(0.5),
        sc.dx,
    )
    compute_viscosity(
        ZeroViscosity(),
        ker,
        sc,
        particles,
        xdiff,
        vdiff,
        grad,
        T(1),
        1,
        2,
    )
    compute_viscosity(
        ArtificialViscosity(),
        ker,
        sc,
        particles,
        xdiff,
        vdiff,
        grad,
        T(1),
        1,
        2,
    )
    compute_viscosity(
        Laminar(),
        ker,
        sc,
        particles,
        xdiff,
        vdiff,
        grad,
        T(1),
        1,
        2,
    )
    compute_viscosity(
        LaminarSPS(),
        ker,
        sc,
        particles,
        xdiff,
        vdiff,
        grad,
        T(1),
        1,
        2,
    )
    compute_density_diffusion(
        ZeroDensityDiffusion(),
        ker,
        sc,
        particles,
        xdiff,
        grad,
        T(1),
        1,
        2,
        motion,
    )
    compute_density_diffusion(
        ZeroGravityLinearDensityDiffusion(),
        ker,
        sc,
        particles,
        xdiff,
        grad,
        T(1),
        1,
        2,
        motion,
    )
    compute_density_diffusion(
        LinearDensityDiffusion(),
        ker,
        sc,
        particles,
        xdiff,
        grad,
        T(1),
        1,
        2,
        motion,
    )
    compute_density_diffusion(
        ComplexDensityDiffusion(),
        ker,
        sc,
        particles,
        xdiff,
        grad,
        T(1),
        1,
        2,
        motion,
    )
    ConstructStencil(Val(D))
    ExtractCells!(particles, inv_cutoff)
    UpdateNeighbors!(
        particles,
        inv_cutoff,
        SortingScratch,
        ParticleRanges,
        UniqueCells,
        CellDict,
    )
    AllocateThreadedArrays(meta, particles, dρdtI, ∇Cᵢ, ∇◌rᵢ)
end

