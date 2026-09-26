# Long-run comparison of the time stepping schemes on the still wedge (mDBC):
#   symplectic | single neighbour (rebuild refresh only) | single neighbour + 20 step re-anchoring
# The third variant inserts the CPU's periodic correction from the host side
# (one step per read back, no graph), so the package source is untouched.
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "cases.jl"))

const SCL = SPHExampleGPU.SPHCellList
const GSS = SPHExampleGPU.GPUStepState
using SPHExampleGPU.GPUStepState: set_output_time!, readback!, resume_after_rebuild!, invalidate_graphs!, I_STOP, STOP_REBUILD, STOP_OUTPUT
const T   = Float64
const TEND  = 2.0
const TOUT  = 0.1
const INTERVAL = 20

function setup(scheme)
    case = BENCH_CASES[findfirst(c -> c.name == "StillWedge2D_MDBC_dp0.02", BENCH_CASES)]
    kw   = case.build(T, mktempdir())
    meta = kw.SimMetaData
    meta.SimulationTime     = TEND
    meta.OutputTimes        = TOUT
    meta.GPUUseGraph        = false
    meta.GPUMaxStepsPerSync = 1
    meta.TimeSteppingMode   = scheme
    particles = AllocateDataStructures(kw.SimGeometry, meta)
    _, gp, gn = LoadBoundaryNormals(Val(2), T, kw.ParticleNormalsPath)
    for gi in eachindex(gp)
        particles.GhostPoints[gi]  = gp[gi]
        particles.GhostNormals[gi] = gn[gi]
    end
    Pressure!(particles.Pressure, particles.Density, kw.SimConstants)
    n   = length(particles)
    gpu = upload_particles(particles)
    sup = GPUSupportArrays{2, T}(n)
    red = ReductionWorkspace{SVector{3, T}}(n)
    cl  = CellListWorkspace{2, T}(n)
    mot = MotionArrays(kw.SimGeometry, particles)
    st  = StepState{T}(; dx = one(T) + kw.SimKernel.h)
    meta.OutputIterationCounter = 1
    return kw, particles, gpu, sup, red, cl, mot, st
end

# Mirror of SimulationLoop with one step per read back and an optional host
# side periodic re-anchoring of the carried derivative (CPU semantics: every
# INTERVAL completed steps unless a rebuild refresh already happened there).
function advance!(kw, gpu, sup, red, cl, mot, st, corr::Bool, stats)
    meta = kw.SimMetaData
    K = kw.SimKernel
    FlagKernel = Val(false); FlagShift = Val(false); UseMDBC = true
    SingleNeighbor = meta.TimeSteppingMode isa SingleNeighborTimeStepping
    lanes = Val(choose_lanes(length(gpu)))
    ctx = (; gpu, cl, sup, red, motion = mot, state = st, SimKernel = K, SimConstants = kw.SimConstants,
             SimDensityDiffusion = kw.SimDensityDiffusion, SimViscosity = kw.SimViscosity,
             FlagKernel, FlagShift, UseMDBC, SingleNeighbor, threads = meta.GPUInteractionThreads, lanes,
             bforces = Val(true), HourGlass = meta.HourGlass)
    t_out = SCL.next_output_time(meta)
    set_output_time!(st, t_out)
    if !st.primed
        SCL.launch_step_reduction!(red, gpu.Position, gpu.Velocity, gpu.Acceleration, sup.Positionₙ⁺, K)
        st.primed = true
    end
    while true
        SCL.enqueue_step!(ctx, false)
        readback!(st)
        SCL.sync_meta_data!(meta, st)
        stop = st.ih[I_STOP]
        if stop == STOP_REBUILD
            SCL.rebuild_cell_list!(gpu, cl, K.H⁻¹)
            invalidate_graphs!(st)
            resume_after_rebuild!(st)
            stats.rebuilds += 1
            if SingleNeighbor
                SCL.enqueue_state_derivative!(ctx, st, false, "r", "r")
                stats.last_refresh = meta.Iteration
            end
        elseif stop == STOP_OUTPUT || meta.TotalTime > t_out
            break
        elseif corr && SingleNeighbor && meta.Iteration > 0 && meta.Iteration % INTERVAL == 0 &&
               stats.last_refresh != meta.Iteration
            SCL.enqueue_state_derivative!(ctx, st, false, "c", "c")
            stats.last_refresh = meta.Iteration
            stats.corrections += 1
        end
    end
    return nothing
end

mutable struct Stats
    rebuilds::Int
    corrections::Int
    last_refresh::Int
end

function metrics(p, consts, zs)
    fl = findall(==(Fluid), p.Type)
    ρ  = p.Density[fl]; P = p.Pressure[fl]; x = p.Position[fl]; v = p.Velocity[fl]
    ρ₀ = consts.ρ₀; g = consts.g
    Phyd = [ρ₀ * g * (zs - xi[end]) for xi in x]
    return (maxdev = maximum(abs.(ρ ./ ρ₀ .- 1)),
            meanρ  = mean(ρ),
            rmsP   = sqrt(mean((P .- Phyd) .^ 2)),
            ek     = 0.5 * consts.m₀ * sum(dot(vi, vi) for vi in v),
            vmax   = maximum(norm, v))
end

function run(scheme, corr::Bool, label)
    kw, particles, gpu, sup, red, cl, mot, st = setup(scheme)
    meta = kw.SimMetaData
    zs = maximum(x[end] for (x, t) in zip(particles.Position, particles.Type) if t == Fluid) + kw.SimConstants.dx / 2
    stats = Stats(0, 0, -1)
    rows = []
    snapshots = Dict{Int, Any}()
    t0 = time()
    while meta.TotalTime < TEND
        advance!(kw, gpu, sup, red, cl, mot, st, corr, stats)
        SCL.download_particles!(particles, gpu, cl.grid)
        m = metrics(particles, kw.SimConstants, zs)
        push!(rows, (t = meta.TotalTime, it = meta.Iteration, m...))
        order = sortperm(particles.ID)
        snapshots[meta.OutputIterationCounter] = (ρ = particles.Density[order], P = particles.Pressure[order],
                                                   x = particles.Position[order], type = particles.Type[order])
        meta.OutputIterationCounter += 1
    end
    wall = time() - t0
    @printf("\n== %s: %d steps, %d rebuilds, %d periodic corrections, %.1f s wall\n", label, meta.Iteration,
            stats.rebuilds, stats.corrections, wall)
    @printf("%6s %7s %12s %10s %12s %12s %10s\n", "t", "steps", "max|ρ/ρ₀-1|", "mean ρ", "rms(P-Phyd)", "Ekin", "vmax")
    for r in rows
        @printf("%6.2f %7d %12.3e %10.3f %12.2f %12.3e %10.3e\n", r.t, r.it, r.maxdev, r.meanρ, r.rmsP, r.ek, r.vmax)
    end
    return rows, snapshots
end

# warm up / compile (short)
let
    kw, particles, gpu, sup, red, cl, mot, st = setup(SymplecticTimeStepping())
    kw.SimMetaData.SimulationTime = 0.01; kw.SimMetaData.OutputTimes = 0.01
    advance!(kw, gpu, sup, red, cl, mot, st, false, Stats(0, 0, -1))
    kw, particles, gpu, sup, red, cl, mot, st = setup(SingleNeighborTimeStepping())
    kw.SimMetaData.SimulationTime = 0.01; kw.SimMetaData.OutputTimes = 0.01
    advance!(kw, gpu, sup, red, cl, mot, st, true, Stats(0, 0, -1))
end

r_sym, s_sym = run(SymplecticTimeStepping(), false, "Symplectic (two loops per step)")
r_sn0, s_sn0 = run(SingleNeighborTimeStepping(), false, "SingleNeighbor, rebuild refresh only (current GPU)")
r_sn20, s_sn20 = run(SingleNeighborTimeStepping(), true, "SingleNeighbor + re-anchoring every $INTERVAL steps (CPU)")

println("\n== fluid density difference vs symplectic at the same output frame (particles matched by ID)")
@printf("%6s %14s %14s %14s %14s\n", "frame", "SN0 rms Δρ", "SN0 max Δρ", "SN20 rms Δρ", "SN20 max Δρ")
for k in sort(collect(keys(s_sym)))
    haskey(s_sn0, k) && haskey(s_sn20, k) || continue
    fl = s_sym[k].type .== Fluid
    d0  = s_sn0[k].ρ[fl]  .- s_sym[k].ρ[fl]
    d20 = s_sn20[k].ρ[fl] .- s_sym[k].ρ[fl]
    @printf("%6d %14.4e %14.4e %14.4e %14.4e\n", k, sqrt(mean(d0 .^ 2)), maximum(abs, d0), sqrt(mean(d20 .^ 2)), maximum(abs, d20))
end
