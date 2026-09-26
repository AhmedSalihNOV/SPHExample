# Benchmark of the particle state layout read by the pair kernel: separate
# arrays (`GPUPackedLayout = false`) against the packed 16 byte aligned
# position + pressure and velocity + density vectors (`true`), following the
# interleaved per kernel protocol of gpu_version/README.md.
#
#     julia --project=gpu_version gpu_version/benchmark/bench_packed_layout.jl [--float64] [--rounds N] [--subdiv S] [--lanes K] <case substrings...>
#
# For every case both variants are set up once (same initial particles),
# warmed up, then measured in `rounds` interleaved rounds (separate, packed,
# separate, packed, ...) under `CUDA.@profile`. Reported per variant: device
# time per step of the interaction kernel, of the element-wise kernels that
# fill the packed buffers (the state preparation and the half step kernel),
# of the mDBC kernel and in total, each as the median over rounds of the
# ratio to the separate layout.
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf
using Statistics

include(joinpath(@__DIR__, "cases.jl"))

function setup(case::BenchCase, ::Type{T}, packed::Bool; subdiv::Int = 1, lanes::Int = 0) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    if kw.ParticleNormalsPath !== nothing   # SimpleMDBC cases
        _, gp, gn = LoadBoundaryNormals(Val(case.dims), T, kw.ParticleNormalsPath)
        for gi in eachindex(gp)
            particles.GhostPoints[gi]  = gp[gi]
            particles.GhostNormals[gi] = gn[gi]
        end
    end
    Pressure!(particles.Pressure, particles.Density, kw.SimConstants)
    n   = length(particles)
    gpu = upload_particles(particles)
    sup = GPUSupportArrays{case.dims, T}(n)
    red = ReductionWorkspace{SVector{3, T}}(n)
    cl  = CellListWorkspace{case.dims, T}(n; reach = subdiv)
    mot = MotionArrays(kw.SimGeometry, particles)
    st  = StepState{T}(; dx = one(T) + kw.SimKernel.h)
    kw.SimMetaData.TimeSteppingMode = kw.SimTimeStepping
    kw.SimMetaData.OutputIterationCounter = 1
    kw.SimMetaData.GPUCellSubdivision  = subdiv
    kw.SimMetaData.GPULanesPerParticle = lanes
    kw.SimMetaData.GPUPackedLayout     = packed
    # many output intervals so that repeated SimulationLoop calls keep stepping
    kw.SimMetaData.SimulationTime = T(1e6)
    return (; kw, gpu, sup, red, cl, mot, st, packed)
end

function loop!(v)
    kw = v.kw
    SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, kw.SimMetaData, kw.SimConstants,
                   v.gpu, v.cl, v.sup, v.red, v.mot, v.st)
    kw.SimMetaData.OutputIterationCounter += 1
    return nothing
end

function kernel_times(prof)
    df = prof.device   # NamedTuple of columns
    t = Dict{String, Float64}()
    for r in eachindex(df.name)
        name = String(df.name[r])
        key = occursin("interaction_kernel", name) ? "interaction" :
              occursin("mdbc_kernel", name)        ? "mdbc" :
              (occursin("prepare_state_kernel", name) || occursin("half_step_kernel", name)) ? "fill" :
              "other"
        t[key] = get(t, key, 0.0) + (df.stop[r] - df.start[r])
    end
    t["total"] = sum(values(t))
    return t
end

function getopt(args, name, default, parsefn = x -> parse(Int, x))
    i = findfirst(==(name), args)
    i === nothing && return default, args
    return parsefn(args[i + 1]), [args[1:i-1]; args[i+2:end]]
end

function main(args)
    T = "--float64" in args ? Float64 : Float32
    rounds, args = getopt(args, "--rounds", 9)
    subdiv, args = getopt(args, "--subdiv", 1)
    lanes, args  = getopt(args, "--lanes", 0)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)
    keys_ = ("interaction", "fill", "mdbc", "total")

    println("GPU $(CUDA.name(CUDA.device())), $T, $rounds interleaved rounds, subdivision $subdiv, lanes $(lanes == 0 ? "auto" : lanes)")
    for case in cases
        variants = [setup(case, T, p; subdiv = subdiv, lanes = lanes) for p in (false, true)]
        # warm up / compile
        for _ in 1:3, v in variants
            loop!(v)
        end
        CUDA.synchronize()

        # measured rounds; every SimulationLoop call runs one output interval
        times = [Dict{String, Vector{Float64}}(k => Float64[] for k in keys_) for _ in variants]
        steps = zeros(Int, length(variants))
        for r in 1:rounds, (k, v) in enumerate(variants)
            it0  = v.kw.SimMetaData.Iteration
            prof = CUDA.@profile loop!(v)
            ns   = v.kw.SimMetaData.Iteration - it0
            steps[k] += ns
            kt = kernel_times(prof)
            for key in keys_
                push!(times[k][key], 1e3 * get(kt, key, 0.0) / ns)   # ms per step
            end
        end

        n = length(variants[1].gpu)
        @printf("\n%s: %d particles, steps per round ≈ %d, lanes %s\n", case.name, n, steps[1] ÷ rounds,
                lanes == 0 ? "auto=$(choose_lanes(n))" : string(lanes))
        @printf("%-9s %9s %9s %9s %9s %9s %9s %9s\n", "layout", "inter", "fill", "mdbc", "total",
                "ratio_i", "ratio_f", "ratio_t")
        for (k, v) in enumerate(variants)
            med(key) = median(times[k][key])
            ratio(key) = median(times[k][key] ./ times[1][key])
            @printf("%-9s %9.4f %9.4f %9.4f %9.4f %9.3f %9.3f %9.3f\n", v.packed ? "packed" : "separate",
                    med("interaction"), med("fill"), med("mdbc"), med("total"),
                    ratio("interaction"), ratio("fill"), ratio("total"))
        end
        println("  (ms of device time per step; fill = state preparation + half step kernels; ratio_* = median over rounds of the per round ratio to the separate layout)")
        flush(stdout)
        variants = nothing
        GC.gc(); CUDA.reclaim()
    end
end

main(ARGS)
