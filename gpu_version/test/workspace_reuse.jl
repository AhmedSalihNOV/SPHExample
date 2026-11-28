module WorkspaceReuseTests
using Test, SPHExampleGPU, CUDA, StaticArrays
include(joinpath(@__DIR__, "..", "benchmark", "workspace_reuse.jl"))

@testset "workspace exclusive ownership" begin
    workspace = GPUWorkspace()
    SPHExampleGPU.SPHCellList.acquire_workspace!(workspace)
    try
        @test_throws ArgumentError empty!(workspace)
        @test_throws ArgumentError RunSimulation(; SimWorkspace = workspace)
    finally
        SPHExampleGPU.SPHCellList.release_workspace!(workspace)
    end
    @test empty!(workspace) === workspace
end

@testset "workspace fresh-run and auxiliary-mode contract" begin
    driver = SPHExampleGPU.SPHCellList
    meta = SimulationMetaData{2,Float32}(SimulationName = "WorkspaceValidation", SaveLocation = tempdir())
    args = (; SimMetaData = meta, SimGeometry = SPHGeometry{2,Float32}[],
        SimParticles = (; Type = [Fluid]), SimTimeStepping = SymplecticTimeStepping())
    @test driver.validate_workspace_run(args) === nothing
    meta.TotalTime = 1
    @test_throws ArgumentError driver.validate_workspace_run(args)
    meta.TotalTime = 0
    meta.Iteration = 1
    @test_throws ArgumentError driver.validate_workspace_run(args)
    meta.Iteration = 0
    @test_throws ArgumentError driver.validate_workspace_run(merge(args, (; SimDamping = ())))
    @test_throws ArgumentError driver.validate_workspace_run(merge(args, (; SimTimeStepping = :unsupported)))
end

if CUDA.functional()
    @testset "independent same-shape GPU runs" begin
        for options in ((;), (; fastmath = true), (; single = true, cache = false, compact = false, double_position = false, graph = false),
                        (; fixed_grid = true, async = false, single_file = false),
                        (; boundary = SimpleMDBC), (; D = 3, T = Float64, boundary = SimpleMDBC))
            workspace = GPUWorkspace()
            first_run = workspace_case(; workspace, options...)
            @test first_run.diagnostics.capacity_growths > 0
            @test isempty(workspace.buffers.state.graphs)
            buffers = workspace.buffers
            position_pointers = Set(UInt(pointer(array)) for array in (buffers.gpu.Position, buffers.gpu.scratch.Position))
            counts_pointer = UInt(pointer(buffers.cl.Counts))
            capacity = length(buffers.cl.Counts)
            for (sound, offset, dx) in ((25f0, 0.4, 0.02), (17f0, -0.24, 0.025), (20f0, 0.0, 0.02))
                # Seed stale controls and derived values; the next run must not
                # observe them, including graph capture failures from an old run.
                fill!(buffers.state.f, NaN)
                fill!(buffers.state.i, Int32(123))
                fill!(buffers.state.blocks_done, Int32(7))
                fill!(buffers.sup.dρdtI, NaN)
                fill!(buffers.gpu.Position, eltype(buffers.gpu.Position)(ntuple(_ -> NaN, length(eltype(buffers.gpu.Position)))))
                fill!(buffers.gpu.Density, 9)
                fill!(buffers.gpu.Type, Frozen)
                buffers.state.graph_capture_failures = 9
                reverse_input = sound == 17f0
                reference = workspace_case(; sound, offset, dx, reverse_input, options...)
                candidate = workspace_case(; workspace, sound, offset, dx, reverse_input, options...)
                @test check_workspace_runs(reference, candidate) === nothing
                @test candidate.diagnostics.workspace_reused
                @test buffers === workspace.buffers
                @test position_pointers == Set(UInt(pointer(array)) for array in (buffers.gpu.Position, buffers.gpu.scratch.Position))
                @test length(buffers.cl.Counts) >= capacity
                # All fixture bounds fit the initially provisioned cell buffers.
                @test UInt(pointer(buffers.cl.Counts)) == counts_pointer
                @test candidate.diagnostics.capacity_growths == 0
                @test buffers.state.graph_capture_failures == 0
                @test first_run.state != candidate.state || sound == 20f0
            end
            @test workspace.runs == 4 && workspace.reuses == 3
            if isempty(options)
                reference = workspace_case(; graph = false, sound = 21f0)
                candidate = workspace_case(; workspace, graph = false, sound = 21f0)
                @test check_workspace_runs(reference, candidate) === nothing
                @test isempty(workspace.buffers.state.graphs)
                @test candidate.execution.graph_batches == 0
            end
            previous_host = deepcopy(first_run.state)
            empty!(workspace)
            @test workspace.buffers === nothing
            @test isequal(first_run.state, previous_host)
        end
    end
    @testset "shape changes and failed runs invalidate cached storage" begin
        workspace = GPUWorkspace()
        workspace_case(; workspace)
        @test_throws ArgumentError workspace_case(; workspace, compact = false)
        @test workspace.buffers === nothing && !workspace.busy
        @test !workspace_case(; workspace, compact = false).diagnostics.workspace_reused
        @test_throws ArgumentError workspace_case(; workspace, D = 3)
        @test workspace.buffers === nothing
    end
end
end
