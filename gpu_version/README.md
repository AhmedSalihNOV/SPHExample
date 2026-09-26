# SPHExampleGPU – CUDA version of SPHExample

`gpu_version/` is a copy of `src/` ported to NVIDIA GPUs with
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). It is a self contained Julia
package (`SPHExampleGPU`) with the same public API as the CPU package: the
example scripts only differ in `using SPHExampleGPU` instead of
`using SPHExample`. All physics (Wendland/cubic kernels, artificial, laminar
and laminar+SPS viscosity, all density diffusion models, dynamic and mDBC
boundary conditions, particle shifting, prescribed motion of bodies) is
supported in 2D and 3D, in `Float64` or `Float32`.

## Requirements

* An NVIDIA GPU with a recent driver (CUDA 12 or newer). The CUDA toolkit is
  downloaded automatically by CUDA.jl, `nvcc` is not needed.
* Julia 1.10 or newer.

## Getting started

```bash
# from the repository root
julia --project=gpu_version -e 'using Pkg; Pkg.instantiate()'
julia --project=gpu_version gpu_version/example/Dambreak2dMDBC.jl
```

The examples in `gpu_version/example/` mirror those in `example/`. Choose the
precision with `FloatType = Float32` or `Float64` at the top of a script.

### Which precision?

* `Float64` reproduces the CPU results to round-off (the test suite checks
  that densities, positions and velocities agree to about `1e-14` relative
  after a few dozen steps).
* `Float32` is where consumer and laptop GPUs are fast. GeForce/RTX-A/Quadro
  cards execute double precision at 1/32 (Ampere, Ada) or 1/64 (Blackwell) of
  their single precision rate, so `Float64` on such a GPU is not faster than a
  modern multi core CPU. Datacenter GPUs (A100, H100) have fast `Float64`.
  `Float32` is the precision DualSPHysics uses by default as well; densities
  stay well within the ~1 % variation of weakly compressible SPH.

## What the GPU does differently

The CPU code loops over cells, evaluates each particle pair once and scatters
the two halves into per thread arrays that are reduced afterwards. On the GPU
that would need atomics or large reductions, so the GPU version instead uses
the standard *gather* formulation:

* **One thread (or a few warp lanes) per particle.** The thread of particle
  `i` loops over the particles of the 9 (2D) or 27 (3D) surrounding cells and
  accumulates `dρ/dt`, the acceleration and the optional kernel and shifting
  sums in registers, then writes them once. No atomics, no per thread arrays,
  no reduction step, no zeroing of arrays. For small particle counts the
  neighbour loop is split over several lanes of a warp (each lane takes every
  `K`-th candidate) and the partial sums are merged with warp shuffles, so a
  3 000 particle 2D case still keeps the whole GPU busy.
* **Exact CPU semantics.** The CPU's density diffusion uses `Dⱼ = -Dᵢ` as a
  short cut that is not symmetric in the particle roles. The gather kernel
  reconstructs which particle of a pair was the CPU's `i` (lower index inside
  a cell, higher index across cells) and calls the very same
  `compute_viscosity`/`compute_density_diffusion` functions with the same
  signature, so the GPU reproduces the CPU results instead of a "close"
  variant. As on the CPU, the models receive the densities of the state being
  evaluated (`ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹`) as arguments, so the corrector loop sees the
  predictor density and velocity, and the reciprocals are computed once per
  particle instead of once per pair.
* **Fluid gating of density diffusion by select, not early exit.** The
  `LinearDensityDiffusion` and `ComplexDensityDiffusion` terms act only between
  two fluid particles. The CPU code returns early when either particle is not
  fluid (about 6% faster there). On the GPU that `return` is a divergent branch
  inside the pair loop: it pays off for warps of boundary particles and costs
  in warps of fluid particles with boundary neighbours. Measured on an RTX
  A1000 (Float32, pair kernel time, 9 interleaved rounds, median ratio to the
  old MotionLimiter multiply): early exit 1.00 (DamBreak2D), 1.02
  (StillWedge2D), 1.01 (DamBreak3D), 0.99 (Duckling3D), a wash; a branch
  free select of the finished term 0.98, 0.99, 0.90, 0.96. The models
  therefore compute the term unconditionally and select it. A multiply by one
  is exact, so this is equivalent to the old form; the only differences seen
  in tests are ulp level and come from FMA contraction in `dot`, which can
  change with any recompile.
* **Dense uniform grid + counting sort.** Particles are binned into cells of
  edge `H` on a grid covering their bounding box (plus a one cell margin).
  A histogram (atomics), a prefix scan and a scatter reorder all particle
  arrays by cell in a single fused gather kernel; the three cells of one row
  form one contiguous index range so a 3D particle scans 9 ranges instead of
  27 cells. The reorder is followed by an in-cell insertion sort, which makes
  runs bitwise reproducible (`GPUDeterministicSort = false` skips it). The
  cell list is only rebuilt when the accumulated displacement exceeds `h`,
  exactly like the CPU version. Gravity and motion factors are derived from
  particle type instead of stored or reordered as separate arrays.
* **Fused element-wise kernels.** Half step, density limiting, prescribed
  motion and pressure are one kernel; the final step also computes the
  pressure for the next step and the block wise reduction of the time step
  limits and the maximum displacement. The corrector advances the position
  with the half step velocity times `dt`, the same symplectic scheme as the
  CPU `FullTimeStep`. A step therefore consists of 6 kernel launches (8
  with mDBC and moving bodies), two of which are single block kernels that
  do the loop control.
* **Device resident time step.** The time step, the simulated time, the
  displacement bound that triggers a cell list rebuild and the loop flags
  live on the device. A single block kernel turns the block wise reduction
  of the previous step into `dt` and decides whether the step may run; the
  other kernels read `dt` from device memory and exit at once when it may
  not. The host enqueues a batch of steps (its size estimated from the last
  read back, at most `GPUMaxStepsPerSync`), reads the state back once, and
  only then rebuilds the cell list or writes an output. Because nothing in
  the launch sequence of a step changes from step to step, it is captured
  once as a CUDA graph and replayed (`GPUUseGraph`), which removes most of
  the per launch overhead that bounds the small 2D cases on Windows.
* **mDBC on the GPU.** One thread per boundary particle gathers the fluid
  neighbours of its ghost node, assembles the `(D+1)×(D+1)` system in
  registers and solves it with StaticArrays inside the kernel.
* **Asynchronous output.** Particle data is downloaded when an output is due
  and written to `vtkhdf` by a Julia task while the GPU already integrates the
  next output interval (`GPUAsyncOutput = false` disables this).

Data stays on the GPU for the whole run; the host `StructArray` passed to
`RunSimulation` holds the state of the last written output (reordered by
cell, like the CPU version) when the function returns.

### Modes are type parameters

As in the CPU package, the optional features of a run are type parameters of
`SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode}` rather
than boolean flags, so the fused kernels are specialised at compile time:

| Parameter | Types | Meaning |
|-----------|-------|---------|
| `SMode` | `NoShifting`, `PlanarShifting` | particle shifting |
| `KMode` | `NoKernelOutput`, `StoreKernelOutput` | store the kernel and kernel gradient sums for output |
| `BMode` | `NoMDBC`, `SimpleMDBC` | mDBC boundary condition (needs `ParticleNormalsPath`) |
| `LMode` | `NoLog`, `StoreLog` | write the simulation log file |

Trailing parameters can be omitted, `SimulationMetaData{2, Float32}(...)`
selects all defaults. `RunSimulation` takes the time stepping scheme as
`SimTimeStepping`: `SymplecticTimeStepping()` evaluates two neighbour loops per
step and is the scheme validated against the CPU; `SingleNeighborTimeStepping()`
reuses the corrector derivative of the previous step as the next predictor and
evaluates one neighbour loop per step. Because both packages share these types
and the `AllocateDataStructures(SimGeometry, SimMetaData)` form, the case
definitions in `benchmark/cases.jl` construct against either package.

### GPU specific `SimulationMetaData` options

| Option | Default | Meaning |
|--------|---------|---------|
| `GPUSyncTimers` | `false` | Synchronize after every phase so the `TimerOutputs` table shows real per kernel times (costs a few percent). |
| `GPUDeterministicSort` | `true` | Sort particles inside each cell after the counting sort; results become bitwise reproducible between runs. |
| `GPUMaxCells` | `50_000_000` | Abort with a clear message if the neighbour grid would need more cells (a particle escaped). |
| `GPUInteractionThreads` | `128` | Threads per block of the interaction and mDBC kernels. |
| `GPULanesPerParticle` | `0` (auto) | Warp lanes that share one particle's neighbour loop (1, 2, 4, ... 32). Small cases cannot fill the GPU with one thread per particle, so the automatic choice launches at least four times the resident thread capacity of the device and lets several lanes scan alternating neighbours, combined with warp shuffles. |
| `GPUBoundaryForces` | `true` | Evaluate the momentum equation for boundary particles too, as the CPU does (their acceleration only enters the force based time step limit). `false` skips it and saves 10-20 % in cases with many boundary particles, at the price of a slightly different adaptive time step. |
| `GPUAsyncOutput` | `true` | Write output files on a task while the GPU continues. |
| `GPUMaxStepsPerSync` | `32` | Upper bound on the steps enqueued between two host read backs of the device resident step state. The actual batch is the estimated number of steps until the next cell list rebuild or output. `1` reproduces a synchronization per step. |
| `GPUUseGraph` | `true` | Capture the launch sequence of a step as a CUDA graph and replay it. Disabled automatically with `GPUSyncTimers`. |

`OutputTimes` also accepts `Float64` values or vectors when `FloatType = Float32`.

## Layout

```
gpu_version/
├── Project.toml          # package SPHExampleGPU (adds CUDA.jl)
├── src/
│   ├── SPHExampleGPU.jl              # module glue, same exports as SPHExample
│   ├── GPUReductions.jl              # fused SVector block reductions
│   ├── GPUCellGrid.jl                # dense grid, counting sort, reorder kernel
│   ├── GPUKernels.jl                 # interaction, mDBC, half/final step kernels
│   ├── SPHCellList.jl                # device containers, time loop, RunSimulation
│   └── ...                           # unchanged host side files from src/
├── example/              # GPU versions of the example scripts
├── benchmark/            # CPU/GPU benchmark harness (shared case list)
└── test/                 # unit tests and CPU vs GPU comparison
```

Files not listed under `src/` above are identical to the CPU version except
for small changes that make them precision generic (`Float32` literals), the
`Float32` variant of `Estimate7thRoot`, and the removal of the CPU only
`AllocateThreadedArrays`.

## Tests

```bash
julia --project=gpu_version gpu_version/test/runtests.jl
```

The suite checks the grid helpers and reductions, bitwise reproducibility,
`Float32` sanity, and runs four cases (2D mDBC wedge, 2D moving square with
shifting and SPS turbulence, 3D dam break, 3D duckling with mDBC) with the CPU
package in a separate process and compares the final state.

## Benchmarks

```bash
# CPU (use -t N,0 on Julia 1.12: `-t auto` adds an interactive thread which the
# CPU code does not account for and crashes)
julia -t 24,0 --project=. gpu_version/benchmark/benchmark_cpu.jl
# GPU
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl --float32
julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl            # Float64
# kernel level profile of one case
julia --project=gpu_version gpu_version/benchmark/profile_steps.jl --float32 DamBreak3D_dp0.0085
```

Each case runs the same physics as the corresponding example script for a
few hundred steps after a warm up run that compiles everything. Results
measured on an Intel i7-12850HX (16 cores, 24 threads) and an NVIDIA RTX
A1000 Laptop GPU (16 SMs, 4 GB, 1/32 rate `Float64`) are listed in the table
below (time per step of the time loop, output excluded).

| Case | Dim | Particles | CPU 24 threads, Float64 [ms/step] | GPU Float32 [ms/step] | Speedup Float32 | GPU Float64 [ms/step] | Speedup Float64 |
|------|-----|----------:|------------:|------------:|--------:|------------:|--------:|
| StillWedge2D_MDBC_dp0.02 | 2D | 3,027 | 1.27 | 0.217 | **5.8x** | 3.16 | 0.4x |
| DamBreak2D_MDBC_dp0.01 | 2D | 6,678 | 1.21 | 0.245 | **4.9x** | 4.16 | 0.3x |
| StillWedge2D_DBC_dp0.01 | 2D | 11,246 | 1.83 | 0.302 | **6.1x** | 8.55 | 0.2x |
| MovingSquare2D_dp0.04 | 2D | 33,020 | 7.48 | 0.710 | **10.5x** | 14.24 | 0.5x |
| MovingSquare2D_dp0.02 | 2D | 128,775 | 29.20 | 1.598 | **18.3x** | 41.63 | 0.7x |
| DamBreak3D_dp0.02 | 3D | 17,446 | 8.23 | 1.367 | **6.0x** | 33.52 | 0.2x |
| Duckling3D_MDBC_dp0.01 | 3D | 54,817 | 20.45 | 2.747 | **7.4x** | 72.35 | 0.3x |
| DamBreak3D_dp0.0085 | 3D | 171,496 | 99.78 | 15.425 | **6.5x** | 478.41 | 0.2x |
| Duckling3D_MDBC_dp0.005 | 3D | 365,656 | 161.08 | 15.746 | **10.2x** | 476.85 | 0.3x |

Notes on reading the table:

* Times are the best of 2-3 runs of the whole time stepping loop (a few
  hundred adaptive steps each), file output excluded; measured 2026-09-25.
* The RTX A1000 Laptop GPU is a small 16 SM GPU whose `Float64` throughput is
  1/32 of `Float32` — that is why `Float64` loses to 24 CPU cores here. On a
  datacenter GPU (A100/H100, 1/2 rate) `Float64` scales like `Float32` does.
* The `Float32` speedup grows with the particle count (18x at 129k particles
  in 2D, 10x at 366k in 3D) because large launches amortize the fixed
  per-kernel overhead (~16 µs on Windows/WDDM); a desktop GPU with more SMs
  will extend this trend to even larger cases.
