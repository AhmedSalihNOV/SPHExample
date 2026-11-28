# Reusing same-shape GPU workspaces

`RunSimulation(...; SimWorkspace = GPUWorkspace())` can retain core device
storage for a parameter sweep. Reuse the same workspace object for successive
calls, supplying fresh initial host particles and metadata each time. See the
[API and supported modes](../README.md#reusing-gpu-workspaces-for-parameter-sweeps).

The retained storage comprises particles and reorder scratch, predictors and
derived-field caches, timestep/reduction/pair-signal buffers, cell histograms,
bounding-box buffers and hierarchical scan capacity. The run restores particle
input order even after previous sorting, rebuilds compact ghost owner/slot maps,
clears derived/control state and forces a fresh initial rebuild. Fixed-grid
bounds are initialized from the new input; capacity and allocation generation
are retained. Histogram/scan active prefixes are overwritten by that rebuild.

CUDA captures are discarded before per-run auxiliary data goes out of scope.
They are recaptured with each run's constants and buffers. Observed capture
counts are saved in the returned result before cleanup. Output staging and
empty motion/body structures are per-run allocations. A shape/context mismatch
throws, concurrent access is rejected, and failed/interrupted calls invalidate
the cache. `empty!(workspace)` drops retained references; CUDA pool reservations
may remain. This is allocation/setup reuse, not an interaction-kernel change.

## Reproduction

From the GPU package directory:

```sh
julia --project=. test/workspace_reuse.jl
julia --project=. test/output_queue.jl
julia --project=. test/simulation_result.jl
julia --project=. benchmark/workspace_reuse.jl
```

The benchmark uses 49 fluid particles and two fixed particles in 2D. One fixed
particle extends the domain to force grid provisioning. Physics is Float32,
positions are Float64, compact buffers and interaction caches are enabled,
interaction fast math is disabled, and the cell subdivisions are R=1, Rx=4.
The seven parameter pairs sweep sound speed from 19 to 25 and simulate 0.015 s,
with output every 0.005 s. Both paths complete warm-up runs before sampling;
fresh/reused timing order alternates each round.

Unprofiled timing covers the complete `RunSimulation` call, including GPU
upload/reset, graph preparation, rebuilds, readbacks, output and capture cleanup.
Geometry/host-particle/logger construction, validation reads and directory
cleanup are outside timing. Upload/setup is the synchronized `00a Upload To GPU`
timer; grid provisioning additionally occurs during the initial rebuild.
CUDA allocation requests are sampled in separate runs using `CUDA.@allocated`,
outside all reported timings. Every pair must match final particle state,
every exported particle/grid dataset, final clock, accepted steps, rebuilds,
step-state readbacks and graph/direct batch counts exactly. Capture fallback,
nonfinite positions and nonpositive/nonfinite densities are rejected.

## RTX 5080 results, 2026-11-28

Windows WDDM, driver 616.92, Julia 1.12.7. All timing and allocation pairs passed
exact state/output comparisons. [Raw paired data](workspace_reuse_results.csv).

| Measurement | Fresh median | Reused median | Median paired fresh/reused [Q1, Q3] |
|---|---:|---:|---:|
| Upload/setup | 0.9902 ms | 0.6399 ms | 1.687 [1.342, 1.925] |
| Complete run | 20.0001 ms | 19.6181 ms | 1.006 [0.988, 1.075] |
| Requested CUDA allocation bytes/run | 31,100 | 364 | Same byte counts in all seven pairs |

Allocation requests fell by 30,736 bytes (98.83%) for this small workload.
Fresh runs grew the grid once; reused runs required no growth. Each pair had
matching accepted work: 36–48 steps depending on sound speed, one rebuild,
four step-state readbacks, zero grid-status/bbox host readbacks, matching graph
batch counts and no direct fallback batches. The allocation reduction is
separate from elapsed speed: upload/setup improved in this case, while the
complete-run paired interval includes unity. No whole-solver speedup or
neighbor-pair throughput improvement is established.

Requested allocation bytes are cumulative CUDA array allocation requests,
including initial buffers replaced during grid growth. They are not live
buffer payload, peak VRAM or CUDA allocator reservations; CUDA driver/context,
compiled-code and graph-internal allocations are not accounted for by this
array-allocation counter. Host memory and output-staging lifetime are separate.
An earlier matching run also reduced requested bytes to 364 and gave a median
setup ratio of 1.46, illustrating the timing variability of this small case.

## Validation scope

The focused workspace suite passed 208 assertions: exclusive ownership and
fresh-run checks; poisoned old particle/control/derived fields; reversed input
order and compact boundary maps; changed sound speed, position and smoothing
length; retained grid/scan provisioning; fixed-grid domain reset; fused
predictors; symplectic/single-neighbor and graph/direct execution; compact/full
buffers and cache/position precision; SimpleMDBC in 2D Float32 and 3D Float64;
exact single-file/numbered and asynchronous/synchronous output; immutable prior
host results; shape rejection and recovery after failure.

Existing output-queue tests passed 218 assertions and simulation-result tests
passed 410, including interruption/failure propagation. No CPU package or
dependency changes were required. The complete CUDA suite was not run.
One earlier stress run emitted a CUDACore warning while freeing a 16-byte GPU
allocation. The repeated stress test, final capture-lifetime/reset tests and
benchmark runs were clean; no cause was established for that earlier warning.
