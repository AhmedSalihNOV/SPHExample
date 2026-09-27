# GPU to CPU: transferable optimizations

Scratch handoff document, not part of the package. Delete when the work is done.

Source of the GPU claims: `C:/git/SPHExamplePrivate.worktrees/cuda-optimized-gpu-version/gpu_version`
(branch `agents/cuda-optimized-gpu-version`), mainly its `README.md`, `src/GPUCellGrid.jl`,
`src/GPUKernels.jl`, `src/GPUReductions.jl`.

## Where the CPU time actually goes

`TimerOutputs` breakdown, 24 threads, Julia started with `-t 24,0`.

DamBreak3D `dp = 0.0085`, 171,496 particles, 117 steps, zero neighbor-list rebuilds:

| Phase | Share | Per step |
|---|---|---|
| 06 NeighborLoop | 90.2 % | 42.7 ms |
| 03 Update To Half TimeStep | 2.7 % | |
| 09 Update To Final TimeStep | 2.2 % | |
| 01 UpdateDx | 1.0 % | |
| 07 Final Density | 0.7 % | |
| 05 Pressure | 0.3 % | |
| 11 Update TimeStep | 0.3 % | |
| 04 + 08 LimitDensityAtBoundary | 0.6 % | |

All element-wise passes together are about 7.8 %.

StillWedge2D mDBC `dp = 0.02`, 3027 particles, 1275 steps:

| Phase | Share |
|---|---|
| 06 NeighborLoop | 49.9 % |
| 02 Apply MDBC before Half TimeStep | 24.4 % |
| of which 02 NeighborLoopMDBC | 19.0 % |
| of which 03 ApplyMDBCCorrection | 5.0 % |

Heap traffic: 27.9 KiB per step in the pair loop, 15.6 KiB per step in mDBC. That is task
spawning, not physics.

## Thread scaling

Microseconds per step, same machine.

| Threads | StillWedge2D mDBC, 3027 part | DamBreak3D dp0.02, 17446 part |
|---|---|---|
| 1 | 1002.4 | 21704.2 |
| 4 | 325.1 | 6223.9 |
| 12 | 263.3 | 3270.0 |
| 24 | 342.1 | 3157.2 |

The 2D case is **slower on 24 threads than on 12**. Peak speedup is 3.8x in 2D and 6.9x in 3D
on 24 cores.

---

## 1. Half-width cells (highest value, blocker cleared)

The CPU grid is hard-wired to edge `H` with a 3^D stencil. `ConstructStencil` at
`src/SPHNeighborList.jl:177` returns `ntuple(_ -> -1:1, D)` and `H⁻¹` is the only scale ever
passed to `UpdateNeighbors!`. There is no width parameter anywhere.

The GPU made the width a type parameter `R` (`CellGrid{D,R}`, cells of edge `H/R`, stencil
`(2R+1)^D`). Measured on an RTX A1000, `R = 2` against `R = 1`:

| Case | Candidates per particle | Inside support |
|---|---|---|
| DamBreak2D | 30.5 to 21.8 | 20-31 % to 32-46 % |
| MovingSquare2D | 34.3 to 23.7 | same range |
| DamBreak3D | 520 to 322 | same range |
| Duckling3D | 262 to 159 | same range |

Pair kernel time ratio with one lane per particle: 0.84, 0.73, 0.77, 0.65. Whole step: 0.94,
0.76, 0.77, 0.73. A full 3D dam break at `dp = 0.0085` finished 40 s sooner.

Candidate count is the part that transfers directly. The CPU spends 90.2 % of a large 3D step
in the pair loop and the reduction in scanned volume is a property of the geometry, not of CUDA.

**The blocker in the GPU README does not apply to the current CPU code.** That README says only
`GPUCellSubdivision = 1` reproduces the CPU's orientation of the asymmetric density diffusion
term. That was true of an older CPU. All three models in `src/SPHDensityDiffusionModels.jl` now
return a pair `(Di, Dj)` in which `Dj` is what particle `j` computes as its own center, not
`-Di`. See the comments at lines 95-96, 156 and 224, and the second
`InverseHydrostaticEquationOfState` call in the complex model at line 229. Swapping which
particle plays the role of `i` therefore yields the same two contributions to the same two
particles. Results change only through floating-point summation order.

Implementation notes from the GPU, worth copying:

- Do not build the stencil as a tuple for `R >= 2`. The GPU uses a lazy row iterator precisely
  because a 25-element tuple unrolls the pair body 25 times and thrashes the instruction cache.
- Merge the `2R+1` cells of a stencil row into one contiguous particle range. The CPU already
  does the equivalent lazily in `NeighborParticleRanges` and eagerly in `PackedNeighborCellLists`,
  so this mostly falls out.
- The cost is 25 ranges instead of 9 in 3D, 4 to 8 times more cells, and an mDBC search over
  125 instead of 27 cells.
- The CPU's symmetric traversal exploits `Other > Cell` ordering. Check that this still holds
  when a row spans several cells.

Start by instrumenting candidates per particle and the share passing the distance test, so the
CPU has the same counter the GPU benchmark used.

## 2. Replace per-call `@threads` with a persistent worker pool

This is the CPU analogue of CUDA graphs and `GPUMaxStepsPerSync`, which exist on the GPU only to
amortize per-launch cost.

There are six `ForEachParticle!` call sites plus two bare `@threads` regions per step, each of
which spawns and joins a fresh task set. The 2D regression from 12 to 24 threads and the 27.9
KiB per step of heap traffic in the pair loop are both this.

`ForEachParticle!` already batches with an atomic counter, so the work-distribution logic can
stay. What changes is that workers are created once and parked on a barrier instead of being
respawned per call.

Expect the largest effect on small and 2D cases, which are exactly the ones that currently run
best on half the machine.

## 3. mDBC, three independent items

mDBC is 24.4 % of the StillWedge2D step.

**a. Test distance before loading the particle type.** `src/SPHCellList.jl:874` reads
`ParticleType[j]` and only then computes the separation. The GPU reversed this and notes that
only about 30 % of candidates inside the support pay the second load. Exactly equivalent, a few
lines of change.

**b. Fuse the linear solve into the gather loop.** `ApplyMDBCCorrection`
(`src/SPHCellList.jl:979`) walks all `N` particles and rejects with `iszero(GhostPoints[i])`,
reading two Bumper-allocated length-`N` arrays, one of `SVector{D+1}` and one of
`SMatrix{D+1,D+1}`. In 3D that is 20 floats per particle written and re-read for a handful of
ghosts. `b[i]` and `A[i]` are complete when particle `i`'s ghost loop ends, so the solve, the
density write and the pressure write can happen inside the gather loop. The GPU does all three
in `mdbc_kernel!`.

**c. Loop over the compacted ghost list.** `NeighborLoopMDBC!` already iterates
`Cache.GhostIndices`. Once (b) is done, `ApplyMDBCCorrection` disappears rather than needing its
own compaction.

**d. Dead code.** `sqrt(abs(xij2))` at `src/SPHCellList.jl:877`. `dot(x,x)` is non-negative.

## 4. Fuse the element-wise passes

Upper bound is about 7.8 % in large 3D and roughly 25 % in the small 2D case. Every pass except
the two pair loops is purely element-wise in `i`, so the two GPU clusters transfer as-is.

**Cluster A**, currently four separate passes plus one at the head of the next
`EvaluateInteractions!`: `HalfTimeStep`, `LimitDensityAtBoundary!(ρₙ⁺)`, `ProgressMotion`,
`Pressure!(ρₙ⁺)`, `FillInverseDensity!`. The GPU does these in one kernel
(`half_step_kernel!`). `ProgressMotion` mutating `Position`/`Velocity` is safe in this order
because `HalfTimeStep` has already consumed them and the corrector reads `Positionₙ⁺`.

**Cluster B**, currently three passes plus two serial traversals plus the next step's pressure:
`DensityEpsi!`, `LimitDensityAtBoundary!(Density)`, `FullTimeStep`, `UpdateΔx!`,
`maximum(AccelerationNormSquared)`, and the next step's `Pressure!`. The GPU does all of these
in `final_step_kernel!`. Nothing writes `Position` or `Positionₙ⁺` between `FullTimeStep` and
the next step's `UpdateΔx!`, so moving the displacement scan into the time step pass is exactly
equivalent. Folding `Pressure!` forward leaves only the mDBC-corrected boundary particles
needing a refresh, which item 3b already handles.

**Hard barrier, do not cross it.** The pair loops read `Pressure[j]`, `Density[j]`,
`Velocity[j]`, `Position[j]`. No pass that writes one of those can be fused into or across a
pair loop.

## 5. Kill the two serial O(N) reductions

`maximum(AccelerationNormSquared)` in `src/TimeStepping.jl:51` and `UpdateΔx!` in
`src/SPHNeighborList.jl:387` are both single-threaded full traversals in an otherwise threaded
step, and the first exists only to be read back from a length-`N` scratch array.

They measure 0.3 % and 1.0 % of the 3D step, so this is not about the flops. It is Amdahl: they
are serial fractions that cap thread scaling, and they matter more as item 2 lifts the ceiling.
Fold both into the threaded `FullTimeStep` pass as a per-thread reduction, the way the GPU rides
all three of its reductions along with an element-wise kernel.

## 6. Hoist the `i`-side operands in the non-symmetric pair methods

The symmetric `NoShifting`/`NoKernelOutput` path already hoists `xᵢ, vᵢ, ρᵢ, ρᵢ⁻¹, Pᵢ`
(`src/SPHCellList.jl:171-175`). The other four `ComputeInteractionsPerParticle!` methods re-read
all five per pair, for example at `src/SPHCellList.jl:760-786` and `:672-697`. The
`ParticleType[i] == Fluid` half of `MotionLimiterCondition` is likewise recomputed per pair at
`:798` and `:856`.

Free win for the shifting and kernel-output modes. LLVM may already hoist some of it; measure
before and after rather than assuming either way.

## 7. Optional, changes results: skip boundary momentum terms

The GPU's `GPUBoundaryForces = false` skips the pressure load, tensile correction, viscosity and
the `dvdt` product for particles whose acceleration is never applied. It saves 10 to 20 % in
cases with many boundary particles.

On the CPU this changes the adaptive time step, because `AccelerationNormSquared` currently
includes boundary accelerations. Ship it as an opt-in flag, not as the default, and document the
time step consequence.

---

## Explicitly not worth transferring

- **Branch-free select for the density diffusion fluid gate.** The GPU README records that the
  CPU's early `return` is about 6 % faster on the CPU. Keep the early exit.
- **`PosCell` cell-relative positions.** A no-op when positions and working precision are both
  `Float64`.
- **Lanes per particle and warp shuffles.** No CPU analogue.
- **Counting sort in place of `sortperm!`.** The CPU has an `issorted` fast path and an in-place
  cycle permutation, and the 3D profile above did zero rebuilds in 117 steps. Unmeasured in
  rebuild-heavy dynamic cases; measure the rebuild phase before touching this.
- **The gather formulation itself.** It doubles pair evaluations to avoid atomics. On the CPU
  the symmetric traversal with private accumulators is the better trade. Note though that the
  per-thread zero-fills and the slot summation (`src/SPHCellList.jl:132-133`, `:145-157`) cost
  `nthreads * N * (D+1)` floats written and re-read per pair loop, which is worth measuring
  against a thread-local-sparse alternative.

---

## Two divergences that are not performance items

**A CPU inconsistency in the shifting term.** Two methods that should differ only in whether
kernel output is stored compute different concentration gradients:

```
src/SPHCellList.jl:800   shift_c_acc += Vⱼ * Wᵢⱼ * Vᵢ * ∇ᵢWᵢⱼ * MotionLimiterCondition
src/SPHCellList.jl:857   shift_c_acc += Vⱼ * Wᵢⱼ * Vⱼ * ∇ᵢWᵢⱼ * MotionLimiterCondition
```

`Vᵢ = m₀ * ρᵢ⁻¹` on line 799, `Vⱼ` on the other. Enabling kernel output, a diagnostic flag,
currently changes the physics. One of these is wrong. The companion `shift_r_acc` lines agree.

**The adaptive time step formulas differ between CPU and GPU.** The CPU uses `h/c₀` and
`sqrt(h/max|a|)` with no viscous term (`src/TimeStepping.jl:27-53`). The GPU adds a viscous
criterion and uses `h/(c₀ + visc)`. The GPU also includes gravity in the acceleration it
reduces, while the CPU adds gravity only as a local inside `HalfTimeStep`/`FullTimeStep` and
never stores it back. For the same state the two codes take different steps. Decide which is
intended before using one as a reference for the other.

A related GPU-side item to verify: in single-neighbor mode no interaction kernel runs between
`final_step_kernel!` writing `acc + g` and the next `half_step_kernel!` adding `g` again to the
value it reads, so the carried acceleration may accumulate gravity. This was reported during the
survey and has not been confirmed by running the GPU code.
