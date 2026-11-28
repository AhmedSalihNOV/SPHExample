# Included in SPHCellList.
"""
    GPUWorkspace()

Reusable device particle, support, reduction, timestep and cell-grid storage for
sequential same-shape `RunSimulation(...; SimWorkspace = workspace)` calls.
Each call supplies fresh host particles and metadata (`TotalTime = Iteration = 0`).
Physical constants, initial values and output settings may change. Allocation
shape, precision, kernel family, CUDA context and interaction launch geometry
must match the first run.

Currently supports stationary, nonperiodic fixed/fluid NoMDBC or SimpleMDBC
with ordinary symplectic or single-neighbor stepping, without shifting,
diagnostics, external forces, damping or other auxiliary simulation systems.
Output staging and empty motion/body structures are allocated per run. Graphs
are recaptured each run; only storage and grid/scan capacity are retained.
The workspace owns its buffers and cannot be used concurrently. `empty!`
releases its references and permits a different shape. Failed/interrupted runs
discard the cached storage. `runs` and `reuses` count completed runs.
"""
mutable struct GPUWorkspace
    signature::Any
    buffers::Any
    lock::ReentrantLock
    busy::Bool
    runs::Int
    reuses::Int
end

GPUWorkspace() = GPUWorkspace(nothing, nothing, ReentrantLock(), false, 0, 0)

function acquire_workspace!(workspace)
    trylock(workspace.lock) || throw(ArgumentError("GPUWorkspace is already in use"))
    if workspace.busy
        unlock(workspace.lock)
        throw(ArgumentError("GPUWorkspace is already in use"))
    end
    workspace.busy = true
    return nothing
end

function release_workspace!(workspace)
    workspace.busy = false
    unlock(workspace.lock)
    return nothing
end

function Base.empty!(workspace::GPUWorkspace)
    acquire_workspace!(workspace)
    try
        workspace.buffers === nothing || CUDA.synchronize()
        workspace.buffers = nothing
        workspace.signature = nothing
    finally
        release_workspace!(workspace)
    end
    return workspace
end

function validate_workspace_run(kwargs)
    meta, geometry = kwargs[:SimMetaData], kwargs[:SimGeometry]
    scheme = kwargs[:SimTimeStepping]
    supported = workspace_modes(meta) && meta.Periodicity === nothing &&
        meta.VariableResolution === nothing && meta.GPUTimestepDiagnosticsEvery == 0 &&
        scheme isa Union{SymplecticTimeStepping, SingleNeighborTimeStepping} &&
        all(g -> g.Type in (Fixed, Fluid) && g.Motion === nothing, geometry) &&
        all(t -> t in (Fixed, Fluid), kwargs[:SimParticles].Type)
    for name in (:SimExternalForces, :SimRigidBodyDynamics, :SimOpenBoundaries, :SimDamping,
                 :SimDivergenceCleaning, :SimDensityObserver, :SimStateSampling, :SimRotation, :SimMotion)
        supported &= get(kwargs, name, nothing) === nothing
    end
    supported || throw(ArgumentError("GPUWorkspace requires stationary nonperiodic fixed/fluid NoMDBC or SimpleMDBC, no shifting/diagnostics/auxiliary systems, and ordinary symplectic or single-neighbor stepping"))
    meta.TotalTime == 0 && meta.Iteration == 0 ||
        throw(ArgumentError("GPUWorkspace starts independent runs; supply fresh metadata with TotalTime=0 and Iteration=0"))
    return nothing
end

workspace_modes(::SimulationMetaData{D,T,S,K,B}) where {D,T,S,K,B} =
    S === NoShifting && B in (NoMDBC, SimpleMDBC)

function workspace_signature(host, meta::SimulationMetaData{D,T,S,K,B}, kernel, scheme) where {D,T,S,K,B}
    compact = meta.GPUCompactBuffers
    ghosts = !compact || B !== NoMDBC
    owners = ghosts && hasproperty(host, :GhostPoints) ? findall(!iszero, host.GhostPoints) : Int[]
    config = effective_configuration(meta, kernel, ZeroDensityDiffusion(), ArtificialViscosity(); particle_count = length(host))
    return (; device = CUDA.device(), context = CUDA.context(), host_type = typeof(host), metadata_type = typeof(meta),
        count = length(host), position_type = position_float_type(meta), kernel_type = typeof(kernel),
        id_type = compact ? metadata_storage_type(Int32, Int, host.ID) : Int,
        group_type = compact ? group_marker_storage_type(host.GroupMarker, UInt) : UInt,
        ghost_count = length(owners), compact, cache = meta.GPUCacheInteractionFields,
        fuse_predictor = meta.GPUFusePredictor && meta.GPUInteractionFastMath && B === NoMDBC && scheme isa SymplecticTimeStepping,
        pairwise = !meta.GPULegacyTimestep, threads = config.effective.interaction_threads,
        lanes = config.effective.lanes_per_particle,
        reach = meta.GPUCellSubdivision, x_reach = meta.GPUXCellSubdivision,
        fixed_grid = meta.GPUFixedGrid, expansion = meta.GPUGridExpansion,
        deterministic = meta.GPUDeterministicSort, max_cells = meta.GPUMaxCells)
end

workspace_zero(::Type{E}) where {E} = zero(E)
workspace_zero(::Type{ParticleType}) = Fixed
workspace_zero(::Type{PosCell{D,T}}) where {D,T} = PosCell{D,T}(zero(SVector{D,T}), Int32(0))
clear_workspace_array!(array::CuArray) = isempty(array) ? nothing : fill!(array, workspace_zero(eltype(array)))
clear_workspace_array!(other) = nothing

function reset_workspace!(buffers, host, meta, kernel)
    # Drain the preceding run before invalidating captures or writing any buffer.
    CUDA.synchronize()
    (; gpu, red, cl, sup, state, pair_signal) = buffers
    invalidate_graphs!(state)
    foreach(clear_workspace_array!, values(gpu.scratch))
    for name in fieldnames(typeof(sup))
        clear_workspace_array!(getfield(sup, name))
    end
    clear_workspace_array!(red.partial)
    if pair_signal isa PairSignalPartials
        clear_workspace_array!(pair_signal.values)
        clear_workspace_array!(pair_signal.reduced)
    end
    # Reordering swaps live/scratch arrays. Restore input order into whichever
    # buffers are live now, before the regular first-run rebuild initializes caches.
    for name in (:Position, :Velocity, :Density, :ID, :Type, :GroupMarker, :Acceleration, :Kernel, :KernelGradient)
        destination = getproperty(gpu, name)
        isempty(destination) && continue
        if hasproperty(host, name)
            copyto!(destination, convert(Vector{eltype(destination)}, getproperty(host, name)))
        else
            clear_workspace_array!(destination)
        end
    end
    has_ghosts = hasproperty(host, :GhostPoints) && hasproperty(host, :GhostNormals)
    owners = isempty(gpu.GhostPoints) || !has_ghosts ? Int[] : findall(!iszero, host.GhostPoints)
    copyto!(gpu.GhostIndex, Int32.(owners))
    if !isempty(gpu.GhostPoints) && !has_ghosts
        clear_workspace_array!(gpu.GhostPoints)
        clear_workspace_array!(gpu.GhostNormals)
    elseif !isempty(gpu.GhostPoints)
        selected = gpu.BoundarySlot === nothing ? eachindex(host) : owners
        copyto!(gpu.GhostPoints, convert(Vector{eltype(gpu.GhostPoints)}, host.GhostPoints[selected]))
        copyto!(gpu.GhostNormals, convert(Vector{eltype(gpu.GhostNormals)}, host.GhostNormals[selected]))
    end
    if gpu.BoundarySlot !== nothing
        slots = zeros(Int32, length(host))
        slots[owners] = Int32.(eachindex(owners))
        copyto!(gpu.BoundarySlot, slots)
    end
    fill!(gpu.CellID, Int32(0))

    cl.grid = typeof(cl.grid)()
    header = GPUCellGrid.GridBuildState(cl.grid, GPUCellGrid.GRID_NOT_NEEDED)
    cl.grid_state_host[1] = header
    copyto!(cl.grid_dev, [cl.grid])
    copyto!(cl.grid_state_dev, [header])
    cl.grid_prepared = cl.grid_initialized = false
    cl.newly_frozen = cl.frozen_total = 0
    cl.nrebuilds = cl.grid_status_readbacks = cl.capacity_growths = cl.bbox_host_readbacks = 0
    cl.freeze_count === nothing || fill!(cl.freeze_count, Int32(0))
    # Histogram, permutation, bbox and scan active prefixes are overwritten by
    # the mandatory first rebuild. Keep their capacities, including fixed grids.

    fill!(state.fh, 0.0)
    fill!(state.ih, Int32(0))
    state.fh[F_DX] = one(eltype(host.Density)) + kernel.h
    copyto!(state.f, state.fh)
    copyto!(state.i, state.ih)
    fill!(state.blocks_done, Int32(0))
    state.primed = false
    state.readbacks = state.graph_batches = state.direct_batches = state.graph_capture_failures = 0
    state.radial_interactions = state.scalar_sps_viscosity = state.preload_interaction_types = false
    state.interaction_always_inline = state.interaction_fastmath = false
    state.graph_capture_error = nothing
    state.configuration = nothing
    return nothing
end

function RunSimulation(; SimWorkspace::Union{Nothing,GPUWorkspace} = nothing, kwargs...)
    SimWorkspace === nothing && return run_simulation!(; SimWorkspace, kwargs...)
    workspace = SimWorkspace
    acquire_workspace!(workspace)
    try
        validate_workspace_run(kwargs)
        meta = kwargs[:SimMetaData]
        reset_timer!(meta.HourGlass)
        meta.CurrentTimeStep = 0
        meta.OutputIterationCounter = meta.StepsTakenForLastOutput = meta.IndexCounter = 0
        result = run_simulation!(; SimWorkspace, kwargs...)
        if result.termination_reason === :completed
            workspace.runs += 1
            workspace.reuses += Int(result.diagnostics.workspace_reused)
        else
            workspace.buffers = workspace.signature = nothing
        end
        return result
    catch
        # Never expose partially initialized or interrupted device state to a
        # subsequent run. Preserve the original error if CUDA itself failed.
        try
            CUDA.synchronize()
        catch
        end
        workspace.buffers = workspace.signature = nothing
        rethrow()
    finally
        release_workspace!(workspace)
    end
end
