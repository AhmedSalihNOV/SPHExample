# Ten simulation-speed priorities for this PC

Target: NVIDIA RTX 5080. Starting point: workspace-reuse commit `15cb64c7`.
These are experiments to prioritize, in shaa Allah; gains require measurement.
The aim is to identify which changes help this PC and which workloads benefit.

1. **Profile complete simulations.** Separate interaction work, cell-list
   rebuilding, launches, transfers and output. Compare a small StillWedge case
   with a larger 3D case, because their bottlenecks may differ. Warm Julia,
   kernels and graph captures before timing. Use profiling to locate costs,
   then confirm improvements with unprofiled complete runs. Host enqueue timers
   and sums of kernel durations do not establish elapsed solver speed.

2. **Improve CUDA graph batching.** Measure submitted versus accepted steps,
   graph batches and readbacks near rebuild/output deadlines. Test reducing
   inactive batch tails without adding extra synchronization or changing the
   device's stopping decision. Preserve exact timestep/clock arithmetic,
   clipped deadlines and pending timesteps across rebuilds. Keep graphs as the
   baseline: miniature persistent-loop gains do not establish a main-solver gain.

3. **Tune interaction launches.** Sweep valid thread counts and lanes per
   particle for small 2D and larger 3D workloads independently. Inspect register
   use, spills and residency alongside timings. More lanes can expose useful
   parallelism in a small case but add shuffle/reduction work elsewhere.
   Tune mDBC launches separately from fluid interactions. Record the resolved
   launch configuration and verify that the requested specialization ran.

4. **Reduce neighbor-search overhead.** Investigate cell-row traversal,
   candidate filtering and reuse of cell-relative positions. Count candidates,
   accepted neighbors and traversal work to explain results, then measure the
   complete solver. Check actual pair identity/orientation and trajectories:
   equal accepted-neighbor counts are insufficient. Cell subdivision can change
   reference diffusion behavior; retain R=1 and Rx=1 for CPU-reference comparisons.

5. **Improve memory access.** Keep interaction inputs dense and contiguous.
   Examine repeated position, velocity, density, reciprocal-density and pressure
   loads, plus register spilling and local-memory traffic. Compare selective
   caching with recomputation; extra cached arrays also require updates and
   storage. Validate pressure/reciprocals against the evaluated state, including
   predictors and corrected walls. Confirm elapsed gains before claiming that
   memory bandwidth is the limiting factor.

6. **Test selective fusion.** Combine compatible preparation, prediction or
   reduction stages one at a time. Fusion can remove launches and memory passes,
   but may increase registers, spills or dependency chains. Check both small
   launch-bound and larger interaction-bound cases. Preserve explicit ordering
   between neighbor reads, predictor writes, reductions and timestep commit;
   retain standalone behavior for low-level rebuild/download APIs.

7. **Reduce rebuild overhead safely.** Measure rebuild frequency and the cost
   of histogramming, scanning, sorting, gathering and refreshing derived fields.
   Reuse their capacity before changing algorithms. Investigate tighter
   conservative displacement bounds only with protection for full and predictor
   queries. Test moving/velocity-only walls, floating particles and boundary
   owner mappings whenever those modes are affected. Compare rebuild cost and
   full-run time together; a cheaper rebuild can still occur more often.

8. **Measure precision costs.** Use validated Float32 physics and quantify the
   numerical benefit and runtime cost of Float64 positions for each case.
   Compare cell-relative pair coordinates where supported. Keep accumulated
   simulation time in Float64 and avoid enlarging timesteps for convenience.
   Precision and arithmetic changes require longer checks of density, pressure,
   free-surface motion and relevant forces/measurements, including difficult
   wall and floating-body cases. Record tolerances before evaluating results.

9. **Extend workspace reuse.** Use `GPUWorkspace()` for supported same-shape
   sweeps, supplying fresh particles and metadata to every run. Next investigate
   reusable output staging and auxiliary storage, with explicit compatibility
   and ownership checks. Drain old work, restore input order, rebuild boundary
   maps and invalidate graph dependencies before reuse. Include failure,
   interruption and fixed-grid reset tests. Report setup/allocation benefits
   separately from timestep throughput and allocator reservations.

10. **Reduce output overhead.** Measure file writing, device downloads, pinned
    staging and queue backpressure separately. Tune asynchronous queue budgets
    while tracking host/device memory and peak live frames. Download only the
    requested fields and choose an output cadence that meets measurement needs.
    Validate both numbered and single-file VTKHDF output, including compact ghost
    expansion. Grid exports must reflect current-frame occupied cells with
    shared adjacent faces. Report changed output requirements explicitly when
    they reduce runtime.

## Measurement and acceptance

Keep one change per comparison. Alternate baseline/candidate order over multiple
warmed pairs and report paired medians and spread, particle/boundary counts,
dimensions, precision, device/driver and resolved settings. Avoid overlapping
other GPU experiments with timing. Separate setup, solver loop, output and
complete-run results where useful.

For execution-only changes, require matching accepted steps, rebuilds, readbacks
and particle/output values. Report provisioning-related allocation/growth
changes separately. For search, operator, precision or timestep changes, explain
any work-count differences and validate against agreed numerical tolerances and
long-run measurements. A speed result with different physical work needs that
qualification. Keep experimental paths opt-in until the evidence supports
broader use.

Report live device payload, host storage, allocation requests and CUDA allocator
reservations as different quantities. Count aliased scratch once. An allocation
counter is not a peak-VRAM measurement, and auxiliary-buffer savings are not
whole-solver memory savings.

## Current evidence and next step

The [workspace measurements](workspace_reuse.md) used a 51-particle synthetic
case on this PC. Requested CUDA allocation bytes fell from 31,100 to 364 per run;
the median paired upload/setup ratio was 1.687. The complete-run ratio was 1.006
with a middle-half interval of 0.988 to 1.075. This supports setup/allocation reuse;
it does not establish improved pair throughput or a complete-solver speedup.

Start with profiling one representative 2D and one representative 3D case.
Choose the next experiment from their measured dominant costs, then retain only
changes that pass the relevant correctness checks and paired timing comparison.
