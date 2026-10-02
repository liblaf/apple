# Neutral Newton forward performance

PCG is the largest measured cost: linear-system attempts occupy **63.5%** of the actual 200-iteration forward loop. Rejected systems alone take **12.93 s (18.6%)**. Both GPU FEM products and the CPU contact path with device transfers contribute; CPU sparse multiplication alone is a smaller component.

![Performance measurements](../data/profile-plots-001/performance.png)

## Actual trajectory

The original 0–100 segment took 30.103041 s; continuation 100–200 took 39.551401 s, totaling **69.654442 s**. The stored receipts attribute 31.297851 s to accepted linear systems and 12.933414 s to rejected attempts. These attempt timers include tiny setup/descent checks surrounding PCG. The remaining 25.423177 s includes diagonal assembly, energy/gradient evaluation, collision work, updates, logging and checkpoint I/O. Setup and shutdown are excluded from that production loop.

There were 4749 accepted-system CG iterations and 282 rejected linear systems. All 200 steps accepted their first Armijo trial: backtracking is not the cause of the measured cost. Resetting the shift and retrying is the requested policy; this profiling experiment leaves it unchanged.

## Matched replay

Three ten-step windows were replayed from the actual saved checkpoints, each with three baseline repetitions, one separate cProfile pass, and one separate synchronized hierarchical pass. One warmup step per window and all state reconstruction/preflight checks are excluded. This is a diagnostic replay, not a continuation beyond 200 or a new adopted neutral. Every replay matches its saved initial/final energy and gradient (relative tolerance 1e-8), accepted PCG iteration counts, shift-retry counts, and line-search trial counts. Largest absolute endpoint gradient discrepancy across measured passes: 2.182e-14 MPa·m².

| Window | Baseline seconds / step: median (range) | HVP calls | Accepted CG iterations | Rejected systems | Synchronized PCG share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–10 | 0.2311 (0.1931–0.2314) | 176 | 82 | 34 | 51.1% |
| 100–110 | 0.3597 (0.3591–0.3607) | 384 | 270 | 10 | 67.3% |
| 190–200 | 0.4152 (0.4138–0.4158) | 470 | 280 | 10 | 74.2% |

Later windows need more HVPs. From 100–110 to 190–200, accepted CG iterations rise only 270→280, while total HVPs rise 384→470. After subtracting accepted CG iterations and their ten true-residual checks, remaining HVPs rise 104→180; these include rejected systems and any extra true-residual checks. The greater amount of linear-solver work is consistent with the late-window slowdown.

## Where PCG time goes

These are **inclusive seconds under forced synchronization**, not additive rows or uninstrumented percentages. CPU SpMV is inside contact HVP; FEM/contact HVP are inside PCG. The complete timing JSON also records additive exclusive times.

| Scope | 0–10 | 100–110 | 190–200 |
| --- | ---: | ---: | ---: |
| All PCG attempts | 1.350 | 2.510 | 3.395 |
| FEM Hessian products | 0.592 | 1.227 | 1.501 |
| Contact HVP including transfers | 0.432 | 0.855 | 1.205 |
| CPU contact SpMV (inside contact HVP) | 0.070 | 0.143 | 0.168 |
| Contact Hessian assembly | 0.119 | 0.100 | 0.092 |
| Broad phase (CCD and updates) | 0.470 | 0.500 | 0.463 |
| Native CCD | 0.176 | 0.163 | 0.147 |

The contact HVP gathers a GPU direction, calls `.numpy(force=True)`, multiplies the cached SciPy sparse Hessian on CPU, converts the result back to a GPU tensor, and scatters it into the result. Its time therefore combines transfers, synchronization, indexing, Python/native glue and sparse multiplication. The cProfile outputs identify `.numpy()` and scalar tensor conversions as hot boundaries, but their waiting time cannot be interpreted as pure transfer bandwidth cost. FEM timing is completed Warp adapter work, not kernel-only GPU timing.

Contact Hessians are cached within each accepted state. Assembly happens once after a state change; the ten-step windows have 9, 9 and 10 assemblies because production preflight primed the start-state Hessian at 0 and 100. Assembly must not be multiplied by the number of HVPs. Broad-phase construction and CCD remain measurable even when no Armijo backtracking occurs.

## Performance work suggested by this profile

1. Benchmark a GPU-resident contact Hessian matvec that uploads the exact physical Hessian once per accepted state, preserving the current matrix, diagonal and residual checks. This targets repeated host/device round trips; the attainable speedup is unmeasured.
2. Inspect FEM HVP kernel and launch costs with a GPU kernel profiler before selecting a kernel optimization. This wall-clock profile establishes the boundary cost but not the kernel-level cause.
3. Reduce rejected-system work only in a separate solver-policy comparison. Shift reuse or a different preconditioner changes the user's declared method and has not been evaluated here.

No optimizer, material, collision or tolerance changes were applied. At iteration 200 the endpoint remains nonconverged with 62 inverted tetrahedra, so performance measurements do not establish a valid neutral.

## Reproduction and limits

Run from `exp/2026/09/22/neutral-newton`:

```bash
CHERRIES_NAME='Neutral Newton matched window performance profile' \
CHERRIES_TAGS='neutral,newton-cg,profiling,matched-replay' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYVISTA_OFF_SCREEN=true \
.venv/bin/python src/40-profile-forward.py \
  > data/profile-001.stdout.log 2>&1
```

The profiler completed successfully, exit 0, including Cherries shutdown. [Comet profile](https://www.comet.com/liblaf/apple/3b988c4eb27a4f2fbb90fd48d1fc8c41). The report/plot script is `src/50-report-profile.py` and executes zero forward solves.

Runtime: NVIDIA GeForce RTX 4090, PyTorch 2.12.0+cu130, IPC threads 4. Verified numerical source hashes match the saved production run; the current working tree has unrelated changes. Material/model setup took 4.256 s, excluded. This setup metric excludes Python imports and CUDA initialization. Input lineage and replay checkpoints are hash-bound in [profile receipts](../data/profile-001/summary.json); the executed instrumentation source is copied beside the receipts.

The GPU was shared with another compute job throughout observed before/after snapshots; these are workload timings, not isolated hardware benchmarks. Baseline trials are sequential, and changing shared load can affect both variation and comparisons. Synchronization overhead (and load variation) changes replay wall time: baseline medians are 2.311, 3.597, 4.152 s; synchronized passes are 2.644, 3.731, 4.577 s. No statistical speedup claim is made.

Raw results: [summary](../data/profile-001/summary.json), [window 190 timing tree](../data/profile-001/window-190/synchronized.json), [window 190 Python hotspots](../data/profile-001/window-190/cprofile.txt), and binary `.prof` files beside each text dump. [Plot SVG](../data/profile-plots-001/performance.svg), [plot PDF](../data/profile-plots-001/performance.pdf), [200-iteration energy and gradient](../data/convergence-plots-200/energy-gradient.png).
