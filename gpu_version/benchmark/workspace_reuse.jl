using SPHExampleGPU, CUDA, StaticArrays, TimerOutputs, Statistics, Printf, HDF5

function workspace_case(; workspace = nothing, sound = 20f0, offset = 0.0,
        duration = 0.015, cache = true, compact = true, double_position = true,
        single = false, fixed_grid = false, boundary = NoMDBC, D = 2, T = Float32,
        dx = 0.02, graph = true, async = true, single_file = true, fastmath = false,
        reverse_input = false, profile_allocations = false)
    constants = SimulationConstants{T}(; dx = T(dx), c₀ = T(sound), g = zero(T), m₀ = T(1000 * dx^D))
    V = SVector{D,T}
    points = vec([V(ntuple(d -> T(offset) + T(0.02) * index[d], D))
        for index in CartesianIndices(ntuple(_ -> 1:7, D))])
    wall = [V(ntuple(d -> d == D ? T(offset - 0.02) : T(offset + 0.08), D)),
        V(ntuple(_ -> T(offset + 1.6), D))]
    normal = V(ntuple(d -> d == D ? T(0.08) : zero(T), D))
    geometry = [
        SPHGeometry{D,T}(Particles = particle_struct_array(wall, T(1000);
            GhostPoints = wall .+ Ref(normal), GhostNormals = fill(normal, length(wall))), GroupMarker = 1, Type = Fixed),
        SPHGeometry{D,T}(Particles = particle_struct_array(points, T(1001);
            Velocity = fill(V(ntuple(d -> d == 1 ? T(0.1) : zero(T), D)), length(points))), GroupMarker = 2, Type = Fluid)]
    return mktempdir() do save
        meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,boundary,StoreLog}(
            SimulationName = "WorkspaceReuse", SaveLocation = save, SimulationTime = duration,
            OutputTimes = duration / 3, VisualizeInParaview = false, OpenLogFile = false,
            ExportSingleVTKHDF = single_file, ExportGridCells = true, GPUAsyncOutput = async,
            GPUUseGraph = graph, GPULanesPerParticle = 1, GPUInteractionFastMath = fastmath,
            GPUCacheInteractionFields = cache, GPUCompactBuffers = compact,
            GPUDoublePosition = double_position, GPUFixedGrid = fixed_grid,
            GPUClipTimeStepToOutput = true)
        host = AllocateDataStructures(geometry, meta)
        reverse_input && (host = host[reverse(eachindex(host))])
        logger = SimulationLogger(save)
        result = nothing
        run!() = RunSimulation(SimGeometry = geometry, SimMetaData = meta,
            SimParticles = host, SimLogger = logger, SimConstants = constants,
            SimKernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx),
            SimDensityDiffusion = LinearDensityDiffusion(), SimViscosity = ArtificialViscosity(),
            SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping(),
            SimWorkspace = workspace)
        elapsed = 0.0
        device_bytes = nothing
        try
            redirect_stdout(devnull) do
                if profile_allocations
                    device_bytes = CUDA.@allocated result = run!()
                else
                    elapsed = @elapsed result = run!()
                end
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        output = Dict{String,Any}()
        function read_group(group, prefix)
            for name in keys(group)
                child = group[name]
                key = prefix * "/" * name
                child isa HDF5.Group ? read_group(child, key) : (output[key] = read(child))
                close(child)
            end
        end
        for name in sort(filter(name -> endswith(name, ".vtkhdf"), readdir(save)))
            h5open(joinpath(save, name), "r") do file
                read_group(file, name)
            end
        end
        order = sortperm(host.ID)
        fields = (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker)
        state = NamedTuple{fields}(Tuple(copy(getproperty(host, f)[order]) for f in fields))
        all(rho -> isfinite(rho) && rho > 0, state.Density) || error("Invalid density")
        all(value -> all(isfinite, value), state.Position) || error("Invalid positions")
        setup = TimerOutputs.time(meta.HourGlass["00a Upload To GPU"]) / 1e9
        return (; state, output, elapsed, setup, device_bytes, steps = meta.Iteration,
            time = meta.TotalTime, diagnostics = result.diagnostics, execution = result.configuration.execution)
    end
end

function check_workspace_runs(reference, candidate)
    for field in (:state, :output, :steps, :time)
        isequal(getproperty(reference, field), getproperty(candidate, field)) || error("Workspace mismatch: $field")
    end
    for field in (:cell_list_rebuilds, :step_state_readbacks, :output_frames, :frozen_particles)
        getproperty(reference.diagnostics, field) == getproperty(candidate.diagnostics, field) || error("Work differs: $field")
    end
    reference.execution == candidate.execution || error("Graph/direct execution differs")
    reference.execution.graph_capture_failures == 0 || error("CUDA graph capture failed")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    workspace = GPUWorkspace()
    for _ in 1:2, cached in (false, true)
        workspace_case(; workspace = cached ? workspace : nothing)
    end
    println(CUDA.name(CUDA.device()), " | Julia ", VERSION, " | same-shape workspace sweep")
    ratios, setup_ratios = Float64[], Float64[]
    open(joinpath(@__DIR__, "workspace_reuse_results.csv"), "w") do io
        println(io, "round,particles,sound_speed,fresh_seconds,reused_seconds,fresh_setup_seconds,reused_setup_seconds,fresh_device_allocation_bytes,reused_device_allocation_bytes,steps,rebuilds,step_state_readbacks,graph_batches,direct_batches,fresh_grid_status_readbacks,reused_grid_status_readbacks,fresh_bbox_readbacks,reused_bbox_readbacks,fresh_grid_growths,reused_grid_growths,exact_state_and_output")
        for round in 1:7
            sound = 18f0 + Float32(round)
            runs = Dict(cached => workspace_case(; workspace = cached ? workspace : nothing, sound)
                for cached in (isodd(round) ? (false, true) : (true, false)))
            reference, candidate = runs[false], runs[true]
            check_workspace_runs(reference, candidate)
            # Allocation instrumentation is separate from unprofiled timings.
            fresh_alloc = workspace_case(; sound, profile_allocations = true)
            reuse_alloc = workspace_case(; sound, workspace, profile_allocations = true)
            check_workspace_runs(fresh_alloc, reuse_alloc)
            push!(ratios, reference.elapsed / candidate.elapsed)
            push!(setup_ratios, reference.setup / candidate.setup)
            println(io, join((round, length(candidate.state.Density), sound, reference.elapsed, candidate.elapsed, reference.setup, candidate.setup,
                fresh_alloc.device_bytes, reuse_alloc.device_bytes, reference.steps,
                reference.diagnostics.cell_list_rebuilds, reference.diagnostics.step_state_readbacks,
                reference.execution.graph_batches, reference.execution.direct_batches,
                reference.diagnostics.grid_status_readbacks, candidate.diagnostics.grid_status_readbacks,
                reference.diagnostics.bbox_host_readbacks, candidate.diagnostics.bbox_host_readbacks,
                reference.diagnostics.capacity_growths, candidate.diagnostics.capacity_growths, true), ','))
            flush(io)
            @printf("%d fresh %.5fs reuse %.5fs; setup %.3f/%.3f ms; allocated %d/%d bytes\n",
                round, reference.elapsed, candidate.elapsed, 1e3reference.setup, 1e3candidate.setup,
                fresh_alloc.device_bytes, reuse_alloc.device_bytes)
        end
    end
    @printf("Median paired fresh/reused: complete run %.3f, upload/setup %.3f\n", median(ratios), median(setup_ratios))
end
