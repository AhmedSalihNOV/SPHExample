using Test
using SPHExampleGPU
using CUDA
using StaticArrays
using StructArrays
using LinearAlgebra
using HDF5

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

# Project of the CPU package used as the reference. Defaults to the repository
# containing `gpu_version`; set `SPHEXAMPLE_CPU_REF` to compare against another
# checkout (e.g. a branch of the CPU code). The shared cases construct the meta
# data with mode type parameters, so the CPU checkout must provide that API.
const REPO      = normpath(get(ENV, "SPHEXAMPLE_CPU_REF", joinpath(@__DIR__, "..", "..")))
const CPU_REF   = joinpath(@__DIR__, "cpu_reference.jl")
const CPU_META  = joinpath(REPO, "src", "SimulationMetaDataConfiguration.jl")
const HAVE_CPU  = isfile(joinpath(REPO, "src", "SPHExample.jl")) && isfile(CPU_META) &&
                  occursin("TimeSteppingMode", read(CPU_META, String))

"""
Run `case` on the GPU for `simtime` seconds of physical time and return the
particles sorted by ID together with the meta data.
"""
function run_gpu(case::BenchCase, ::Type{T}, simtime; kwargs...) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    kw.SimMetaData.SimulationTime = T(simtime)
    kw.SimMetaData.OutputTimes    = T(simtime)
    for (k, v) in kwargs
        setproperty!(kw.SimMetaData, k, v)
    end
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    order = sortperm(particles.ID)
    return particles[order], kw.SimMetaData
end

"""
Run the CPU reference in a separate Julia process (separate environment) and
return the stored state.
"""
function run_cpu_reference(case::BenchCase, simtime)
    out = tempname() * ".h5"
    cmd = `$(Base.julia_cmd()) -t 8,0 --project=$(REPO) $(CPU_REF) $(case.name) $(simtime) $(out)`
    run(pipeline(cmd; stdout = devnull, stderr = devnull))
    return h5open(out, "r") do fid
        (ID = read(fid["ID"]), Density = read(fid["Density"]), Pressure = read(fid["Pressure"]),
         Position = read(fid["Position"]), Velocity = read(fid["Velocity"]),
         Iteration = read(fid["Iteration"]), TotalTime = read(fid["TotalTime"]))
    end
end

relerr(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), eps(eltype(b))))

@testset "SPHExampleGPU" begin
    @test CUDA.functional()

    @testset "mode types mirror the CPU API" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save)
        @test meta isa SimulationMetaData{2, Float32, NoShifting, NoKernelOutput, NoMDBC, NoLog}
        @test meta.TimeSteppingMode isa SingleNeighborTimeStepping
        meta = SimulationMetaData{2, Float32, PlanarShifting}(SimulationName = "m", SaveLocation = save)
        @test meta isa SimulationMetaData{2, Float32, PlanarShifting, NoKernelOutput, NoMDBC, NoLog}
        meta = SimulationMetaData{3, Float64, NoShifting, StoreKernelOutput, SimpleMDBC, StoreLog}(
            SimulationName = "m", SaveLocation = save, OutputTimes = 0.01)
        @test meta isa SimulationMetaData{3, Float64, NoShifting, StoreKernelOutput, SimpleMDBC, StoreLog}
        @test meta.OutputTimes === 0.01
        # the shared case file builds against this API: typed meta data plus
        # the time stepping scheme for `RunSimulation`
        for c in BENCH_CASES
            kw = c.build(Float64, save)
            @test kw.SimTimeStepping isa TimeSteppingMode
            @test kw.SimMetaData isa SimulationMetaData{c.dims, Float64, S, K, B, StoreLog} where {S, K, B}
            mdbc = kw.SimMetaData isa SimulationMetaData{c.dims, Float64, S, K, SimpleMDBC, L} where {S, K, L}
            @test mdbc == (kw.ParticleNormalsPath !== nothing)
        end
        kw = BENCH_CASES[1].build(Float64, save)
        particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
        @test length(particles) == length(AllocateDataStructures(kw.SimGeometry))
        @test hasproperty(particles, :GhostPoints)
    end

    @testset "output schedule is clamped to the simulation end" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float64}(SimulationName = "m", SaveLocation = save,
                                              SimulationTime = 0.25, OutputTimes = 0.1,
                                              OutputIterationCounter = 1)
        # one based frame counter: frame 1 is the initial state at t = 0
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.1
        meta.OutputIterationCounter = 2
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.2
        # the last interval ends at the simulation end, not at 0.3
        meta.OutputIterationCounter = 3
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputIterationCounter = 4
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputTimes = [0.1, 0.2]
        meta.OutputIterationCounter = 1
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.1
        meta.OutputIterationCounter = 2
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.2
        meta.OutputIterationCounter = 3
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
        meta.OutputTimes = [0.5]
        meta.OutputIterationCounter = 1
        @test SPHExampleGPU.SPHCellList.next_output_time(meta) == 0.25
    end

    @testset "type-derived factors" begin
        for T in (Float32, Float64)
            for (type, gravity, limiter) in ((Fluid, -1, 1), (Fixed, 0, 0),
                                              (Moving, 1, 0))
                @test GravityFactorValue(T, type) === T(gravity)
                @test MotionLimiterValue(T, type) === T(limiter)
            end
        end
    end

    @testset "output variables are validated against the modes" begin
        save = mktempdir()
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save,
                                              OutputVariables = ["Kernel", "Density", "GhostPoints", "Density"])
        kept = @test_logs (:warn, r"Kernel") (:warn, r"GhostPoints") resolve_output_variables!(meta)
        @test kept == ["Density"]
        @test meta.OutputVariables == ["Density"]
        meta = SimulationMetaData{2, Float32}(SimulationName = "m", SaveLocation = save, OutputVariables = ["ChunkID"])
        @test_throws ErrorException resolve_output_variables!(meta)
        meta = SimulationMetaData{2, Float32, NoShifting, StoreKernelOutput, SimpleMDBC}(
            SimulationName = "m", SaveLocation = save, OutputVariables = ["Kernel", "GhostNormals", "Acceleration"])
        @test resolve_output_variables!(meta) == ["Kernel", "GhostNormals", "Acceleration"]
        @test DEFAULT_OUTPUT_VARIABLES == ["Velocity", "Density", "Pressure", "ID", "Type", "GroupMarker"]
    end

    @testset "only the requested variables are written" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        save = mktempdir()
        kw   = case.build(Float32, save)
        meta = kw.SimMetaData
        meta.SimulationTime  = 0.004f0
        meta.OutputTimes     = 0.002f0
        meta.ExportGridCells = true
        meta.OutputVariables = ["Density", "Velocity", "ID", "Acceleration", "GhostPoints", "Kernel"]
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger    = SimulationLogger(save; to_console = false)
        RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
        kernel_mode = meta isa SimulationMetaData{2, Float32, S, StoreKernelOutput} where {S}
        expected = kernel_mode ? ["Density", "Velocity", "ID", "Acceleration", "GhostPoints", "Kernel"] :
                                 ["Density", "Velocity", "ID", "Acceleration", "GhostPoints"]
        @test meta.OutputVariables == expected
        n = length(particles)
        h5open(joinpath(save, meta.SimulationName * ".vtkhdf"), "r") do fid
            pd = fid["VTKHDF"]["PointData"]
            @test sort(keys(pd)) == sort(expected)
            @test size(pd["Velocity"]) == (3, 3n)        # 2D widened to 3 components, 3 frames
            @test size(pd["GhostPoints"]) == (3, 3n)
            @test length(pd["Density"]) == 3n
            @test any(!iszero, read(pd["GhostPoints"]))  # ghost data of the boundary particles
        end
        h5open(joinpath(save, meta.SimulationName * "_GridCells.vtkhdf"), "r") do fid
            @test keys(fid["VTKHDF"]["CellData"]) == ["CellData"]
        end
        # the complete final state is on the host, including fields not written
        @test all(isfinite, particles.Pressure)
        @test length(unique(particles.ID)) == n
    end

    @testset "cell grid helpers" begin
        invH = 1 / 0.04
        @test map_floor(0.0, invH)   == 0
        @test map_floor(0.019, invH) == 0
        @test map_floor(0.021, invH) == 1
        @test map_floor(-0.021, invH) == -1
        @test map_floor(-0.019, invH) == 0
        grid = CellGrid{2}((Int32(-3), Int32(-2)), (Int32(10), Int32(8)), Int32(80))
        for c in ((-2, -1), (0, 0), (5, 4))
            lin = SPHExampleGPU.GPUCellGrid.linear_cell(grid, Int32.(c))
            l   = SPHExampleGPU.GPUCellGrid.local_coords(grid, lin)
            @test l .+ grid.origin == Int32.(c)
        end
    end

    @testset "fused reduction" begin
        n = 100_003
        x = CuArray(rand(Float64, n))
        ws = ReductionWorkspace{SVector{2, Float64}}(n)
        f(i, x) = (@inbounds v = x[i]; SVector(v, -v))
        op(a, b) = SVector(max(a[1], b[1]), min(a[2], b[2]))
        r = reduce_svector(ws, f, op, SVector(-Inf, Inf), n, x)
        xh = Array(x)
        @test r[1] == maximum(xh)
        @test r[2] == -maximum(xh)
    end

    @testset "counting sort orders particles by cell" begin
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        initial = AllocateDataStructures(case.build(Float64, mktempdir()).SimGeometry)
        device = upload_particles(initial)
        @test !hasproperty(device, :GravityFactor)
        @test !hasproperty(device, :MotionLimiter)
        @test !hasproperty(device.scratch, :GravityFactor)
        @test !hasproperty(device.scratch, :MotionLimiter)
        particles, meta = run_gpu(case, Float64, 1e-4)
        # after the run the host arrays are the (cell sorted) device state; the
        # sort above restored ID order, so check on a fresh download instead
        @test length(unique(particles.ID)) == length(particles)
        @test all(isfinite, particles.Density)
        @test !hasproperty(particles, :GravityFactor)
        @test !hasproperty(particles, :MotionLimiter)
        # derived or diagnostic-only fields are neither stored nor written
        @test !hasproperty(device, :BoundaryBool)
        @test !hasproperty(device, :ChunkID)
        @test !hasproperty(particles, :BoundaryBool)
        @test !hasproperty(particles, :ChunkID)
    end

    @testset "deterministic repeat" begin
        case = BENCH_CASES[findfirst(c -> c.name == "DamBreak2D_MDBC_dp0.01", BENCH_CASES)]
        p1, _ = run_gpu(case, Float64, 0.005)
        p2, _ = run_gpu(case, Float64, 0.005)
        @test p1.Density == p2.Density
        @test p1.Position == p2.Position
    end

    @testset "Float32 runs and stays physical" begin
        for name in ("StillWedge2D_MDBC_dp0.02", "DamBreak3D_dp0.02")
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            p, meta = run_gpu(case, Float32, 0.01)
            @test eltype(p.Density) == Float32
            @test all(isfinite, p.Density)
            @test all(x -> all(isfinite, x), p.Position)
            @test 900 < minimum(p.Density) && maximum(p.Density) < 1100
        end
    end

    @testset "final step corrector advances with the half step velocity" begin
        # The symplectic corrector of the CPU `FullTimeStep` moves the position
        # with `Velocityₙ⁺ * dt`. The GPU kernel is checked element by element
        # against that formula (with and without shifting) and against the
        # earlier averaged velocity scheme, which it must not reproduce.
        T = Float64
        V = SVector{2, T}
        n = 4_096
        rnd() = V(randn(T), randn(T))
        types  = rand([Fluid, Fixed, Moving], n)
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, CFL = 0.5)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        dt     = T(1e-4)
        for FlagShift in (false, true)
            Position     = [rnd() for _ in 1:n]
            Velocity     = [rnd() for _ in 1:n]
            Acceleration = [10 * rnd() for _ in 1:n]
            Velocityₙ⁺   = [rnd() for _ in 1:n]
            Positionₙ⁺   = Position .+ [1e-3 * rnd() for _ in 1:n]
            Density  = 1000 .+ 50 .* rand(T, n)
            ρₙ⁺      = 1000 .+ 50 .* rand(T, n)
            dρdtI    = randn(T, n)
            ∇Cᵢ      = [rnd() for _ in 1:n]
            ∇◌rᵢ     = 4 .* rand(T, n) .- 1     # negative values must give no shift

            dP  = CuArray(Position);      dV  = CuArray(Velocity)
            dA  = CuArray(Acceleration);  dρ  = CuArray(Density)
            dPr = CUDA.zeros(T, n);       ddρ = CuArray(dρdtI)
            dρn = CuArray(ρₙ⁺);           dPn = CuArray(Positionₙ⁺)
            dVn = CuArray(Velocityₙ⁺);    dty = CuArray(types)
            dC  = CuArray(∇Cᵢ);           dr  = CuArray(∇◌rᵢ)
            red = ReductionWorkspace{SVector{3, T}}(n)
            launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr,
                               HostStep(dt, zero(T)), kern, consts, red, Val(FlagShift))
            xg = Array(dP)
            vg = Array(dV)

            x_ref = similar(Position); v_ref = similar(Velocity); x_avg = similar(Position)
            for i in 1:n
                ML  = MotionLimiterValue(T, types[i])
                GF  = GravityFactorValue(T, types[i])
                acc = Acceleration[i] + ConstructGravitySVector(Acceleration[i], consts.g * GF)
                v   = Velocity[i] + acc * dt * ML
                δx  = zero(V)
                if FlagShift
                    A_FSC = (∇◌rᵢ[i] - 0) / (2 - 0)
                    δx = A_FSC < 0 ? zero(V) : -A_FSC * 2 * kern.h * norm(Velocityₙ⁺[i]) * dt * ∇Cᵢ[i]
                end
                v_ref[i] = v
                x_ref[i] = Position[i] + (Velocityₙ⁺[i] * dt + δx) * ML
                x_avg[i] = Position[i] + (((v + (v - acc * dt * ML)) / 2) * dt + δx) * ML
            end
            @test all(isapprox.(vg, v_ref; rtol = 1e-13, atol = 1e-15))
            @test all(isapprox.(xg, x_ref; rtol = 1e-13, atol = 1e-15))
            # the old averaged scheme differs for every moving particle
            moving = types .== Fluid
            @test maximum(norm.(xg[moving] .- x_avg[moving])) > 1e-6
            # non-fluid particles do not move
            @test all(xg[.!moving] .== Position[.!moving])
        end
    end

    @testset "device resident step state" begin
        # The finish kernel must reproduce the host formula for `dt` and the
        # displacement bookkeeping exactly, and the stop flag has to gate
        # every kernel of a step.
        S = SPHExampleGPU.GPUStepState
        T = Float64
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, CFL = 0.5)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        h      = kern.h
        red    = ReductionWorkspace{SVector{3, T}}(100_000)
        @test red.nblocks > 1
        partials = [SVector{3, T}(rand(), 1 + rand(), 0.2 * rand()) for _ in 1:red.nblocks]
        partials[min(7, red.nblocks)] = SVector{3, T}(3.0, 0.5, 0.25)   # extrema of all three components
        copyto!(red.partial, partials)
        visc, dt1, disp = 3.0, 0.5, 0.25
        dt_ref = consts.CFL * min(dt1, h / (consts.c₀ + visc))

        st = StepState{T}(; time = 0.0, dx = 0.0)
        S.set_output_time!(st, 1.0)
        launch_finish!(st, red, kern, consts)
        readback!(st)
        @test st.fh[S.F_DT]   == dt_ref
        @test st.fh[S.F_DISP] == 4 * disp
        @test st.fh[S.F_DX]   == 4 * disp
        @test st.ih[S.I_PHASE] == S.PHASE_DT_READY
        @test st.ih[S.I_STOP]  == S.STOP_REBUILD          # 4 * disp >= h
        # nothing runs while stopped
        launch_commit!(st)
        readback!(st)
        @test st.ih[S.I_ITER] == 0 && st.fh[S.F_TIME] == 0
        # after the rebuild the stored dt is kept and the step commits
        S.resume_after_rebuild!(st)
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.fh[S.F_DX] == 0
        @test st.fh[S.F_DT] == dt_ref
        @test st.fh[S.F_TIME] == dt_ref
        @test st.ih[S.I_ITER] == 1
        @test st.ih[S.I_PHASE] == S.PHASE_NEED_DT
        @test st.ih[S.I_STOP]  == S.STOP_NONE
        # a second step accumulates the displacement bound
        copyto!(red.partial, fill(SVector{3, T}(0.1, 2.0, 1e-4), red.nblocks))
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.fh[S.F_DX] == 4e-4
        @test st.fh[S.F_DT] == consts.CFL * min(2.0, h / (consts.c₀ + 0.1))
        @test st.fh[S.F_TIME] == dt_ref + st.fh[S.F_DT]
        @test st.ih[S.I_ITER] == 2
        # output time reached: stop before computing anything
        S.set_output_time!(st, 0.0)
        launch_finish!(st, red, kern, consts)
        launch_commit!(st)
        readback!(st)
        @test st.ih[S.I_STOP] == S.STOP_OUTPUT
        @test st.ih[S.I_ITER] == 2
        @test st.ih[S.I_PHASE] == S.PHASE_NEED_DT

        # batch size: estimate until rebuild / output, at most kmax, at least 1
        bs = SPHExampleGPU.SPHCellList.batch_size
        st.fh[S.F_DX] = 0.0; st.fh[S.F_DISP] = h / 10; st.fh[S.F_DT] = 1e-3; st.fh[S.F_TIME] = 0.0
        @test bs(st, h, 1.0, 32) == 11          # 10 steps until Δx reaches h, plus one
        @test bs(st, h, 0.0035, 32) == 4        # 3.5 steps until the output, plus one
        @test bs(st, h, 1.0, 4) == 4
        @test bs(st, h, 1.0, 1) == 1
        st.fh[S.F_DX] = h
        @test bs(st, h, 1.0, 32) == 1           # rebuild already due
        st.fh[S.F_DX] = 0.0; st.fh[S.F_DISP] = 0.0; st.fh[S.F_DT] = 0.0
        @test bs(st, h, 1.0, 32) == 32          # no information: fill the batch

        # a stopped state gates the particle kernels
        n = 2_048
        V = SVector{2, T}
        rnd() = V(randn(T), randn(T))
        P = [rnd() for _ in 1:n]
        dP = CuArray(P); dV = CuArray([rnd() for _ in 1:n]); dA = CuArray([rnd() for _ in 1:n])
        dρ = CuArray(1000 .+ rand(T, n)); dPr = CUDA.zeros(T, n); ddρ = CuArray(randn(T, n))
        dρn = CuArray(1000 .+ rand(T, n)); dPn = CuArray(P); dVn = CuArray([rnd() for _ in 1:n])
        dty = CuArray(fill(Fluid, n)); dC = CuArray([rnd() for _ in 1:n]); dr = CuArray(rand(T, n))
        red2 = ReductionWorkspace{SVector{3, T}}(n)
        launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr, st, kern, consts, red2, Val(false))
        @test Array(dP) == P
        S.set_output_time!(st, 1.0)
        launch_final_step!(dP, dV, dA, dρ, dPr, ddρ, dρn, dPn, dVn, dty, dC, dr, st, kern, consts, red2, Val(false))
        @test Array(dP) != P
    end

    @testset "graph replay and batching reproduce direct launches" begin
        # The same steps, once as replayed CUDA graphs in batches and once
        # launched directly with a read back after every step, must give the
        # same bits (deterministic sort keeps the particle order identical).
        case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
        p1, m1 = run_gpu(case, Float64, 0.004)
        p2, m2 = run_gpu(case, Float64, 0.004; GPUUseGraph = false, GPUMaxStepsPerSync = 1)
        @test m1.Iteration == m2.Iteration > 1
        @test m1.TotalTime == m2.TotalTime
        @test p1.ID == p2.ID
        @test p1.Position == p2.Position
        @test p1.Velocity == p2.Velocity
        @test p1.Density  == p2.Density
    end

    @testset "models use the densities passed by the kernel" begin
        # The kernel hands the models ρᵢ, ρⱼ and their reciprocals of the state
        # being evaluated. The built in models must not reach back into a
        # particle array (which in the corrector loop would hold the wrong
        # state), so `SimParticles = nothing` has to work.
        T  = Float64
        consts = SimulationConstants{T}(dx = 0.02, c₀ = 42.0, δᵩ = 0.1, α = 0.02, ν₀ = 1e-4)
        kern   = SPHKernelInstance{2, T}(WendlandC2(); dx = consts.dx)
        xᵢⱼ    = SVector{2, T}(0.011, -0.007)
        vᵢⱼ    = SVector{2, T}(0.3, 0.1)
        d²     = dot(xᵢⱼ, xᵢⱼ)
        ∇W     = ∇Wᵢⱼ(kern, sqrt(d²) * kern.h⁻¹, xᵢⱼ)
        ρᵢ, ρⱼ = T(1003.5), T(998.2)
        types  = [Fluid, Fluid]
        for model in (ZeroViscosity(), ArtificialViscosity(), Laminar(), LaminarSPS())
            Πᵢ, Πⱼ = compute_viscosity(model, kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                       ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2)
            @test all(isfinite, Πᵢ) && all(isfinite, Πⱼ)
            @test Πⱼ ≈ -Πᵢ
        end
        for model in (ZeroDensityDiffusion(), ZeroGravityLinearDensityDiffusion(),
                      LinearDensityDiffusion(), ComplexDensityDiffusion())
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, types)
            @test isfinite(Dᵢ) && Dⱼ == -Dᵢ
        end
        # The gated models are exactly zero unless both particles are fluid and
        # non zero (the gate is a select of the full term) when both are.
        for model in (LinearDensityDiffusion(), ComplexDensityDiffusion())
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fluid, Fixed])
            @test Dᵢ == 0 && Dⱼ == 0
            Dᵢ, Dⱼ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                               ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fixed, Fluid])
            @test Dᵢ == 0 && Dⱼ == 0
            Dᵢ, _ = compute_density_diffusion(model, kern, consts, nothing, xᵢⱼ, ∇W, d²,
                                              ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2, [Fluid, Fluid])
            @test Dᵢ != 0
        end
        # Laminar viscosity is antisymmetric in the density arguments and
        # depends on them, so passing different densities must change it.
        a, _ = compute_viscosity(Laminar(), kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                 ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), 1, 2)
        b, _ = compute_viscosity(Laminar(), kern, consts, nothing, xᵢⱼ, vᵢⱼ, ∇W, d²,
                                 2ρᵢ, 2ρⱼ, inv(2ρᵢ), inv(2ρⱼ), 1, 2)
        @test a ≈ 2b
    end

    if HAVE_CPU
        @testset "matches CPU reference: $(name)" for (name, simtime) in (
                ("StillWedge2D_MDBC_dp0.02", 0.02),
                ("MovingSquare2D_dp0.04",    0.01),
                ("DamBreak3D_dp0.02",        0.005),
                ("Duckling3D_MDBC_dp0.01",   0.005),
            )
            case = BENCH_CASES[findfirst(c -> c.name == name, BENCH_CASES)]
            ref  = run_cpu_reference(case, simtime)
            p, meta = run_gpu(case, Float64, simtime)

            @test meta.Iteration == ref.Iteration
            @test p.ID == ref.ID
            dρ = relerr(p.Density, ref.Density)
            dx = maximum(norm.(p.Position .- eachcol(ref.Position)))
            dv = maximum(norm.(p.Velocity .- eachcol(ref.Velocity)))
            @info "CPU vs GPU ($name): steps=$(meta.Iteration) max rel Δρ=$(dρ) max |Δx|=$(dx) max |Δv|=$(dv)"
            @test dρ < 1e-8
            @test dx < 1e-9
            @test dv < 1e-7
        end
    else
        @warn "CPU package with the mode type API not found at $(REPO); skipping CPU comparison tests " *
              "(set SPHEXAMPLE_CPU_REF to a checkout that has it)"
    end
end
