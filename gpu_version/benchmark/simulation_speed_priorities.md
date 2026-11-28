# Ten simulation-speed priorities for this PC

Target: NVIDIA RTX 5080. Starting point: workspace-reuse commit `15cb64c7`.
These are experiments to prioritize, in shaa Allah; gains require measurement.

1. **Profile complete simulations.** Separate interactions, rebuilding, launches,
   transfers and output, then optimize the dominant cost in representative cases.
2. **Improve CUDA graph batching.** Reduce unnecessary queued steps near rebuilds
   and output deadlines. Keep graphs as the baseline: the miniature persistent
   benchmark does not establish a main-solver gain.
3. **Tune interaction launches.** Compare thread counts and lanes per particle
   separately for small 2D and larger 3D workloads.
4. **Reduce neighbor-search overhead.** Improve cell-row traversal, candidate
   filtering and data reuse; check accepted pairs and trajectories.
5. **Improve memory access.** Keep interaction inputs dense and investigate
   unnecessary loads, register spilling and local-memory traffic.
6. **Test selective fusion.** Combine compatible preparation, prediction or
   reduction stages when paired whole-solver timings show a benefit.
7. **Reduce rebuild overhead safely.** Reuse sorting/scan storage and investigate
   tighter conservative displacement bounds while preserving neighbor coverage.
8. **Measure precision costs.** Use validated Float32 physics and quantify the
   cost and numerical benefit of Float64 positions for each case.
9. **Extend workspace reuse.** Investigate reusable output staging and auxiliary
   storage. This primarily benefits short runs and parameter sweeps.
10. **Reduce output overhead.** Tune asynchronous queue sizes, avoid unnecessary
    downloads and choose output intervals that meet measurement requirements.

Use warmed, alternating comparisons with matching accepted steps, rebuilds and
readbacks, plus particle-state checks. Report setup, allocation and elapsed
solver results separately. Search/arithmetic changes also require numerical
and long-run validation; equal neighbor counts alone do not establish equality.

See [workspace measurements](workspace_reuse.md) for the current setup/allocation
results and their limits.
