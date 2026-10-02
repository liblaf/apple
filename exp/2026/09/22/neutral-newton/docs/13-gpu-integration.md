# Integrated GPU neutral solver

The GPU implementations are reusable library backends wired into the neutral Newton-CG driver. **GPU contact is the default** because it is faster for this workload. Full GPU sparse assembly is integrated and validated as `gpu_sparse`; its assembly and startup costs do not pay off for these short PCG solves.

![Backend comparison](../data/gpu-integration-review-003/gpu-backends.png)

## Actual main-solver runs

All three backends ran 200 accepted iterations from the same repaired constitutive reference through `10-run-neutral.py`. Materials, constraints, energy, gradient, exact diagonal, shift schedule, PCG tolerance, line search and CCD use the same policy. Each run saved a complete trace, protocol, checkpoints, source archive, and geometry/contact diagnostics.

| Backend | Forward loop s | Sparse startup s | Sum s | Sampled backend MiB |
| --- | ---: | ---: | ---: | ---: |
| Reference | 46.056 | 0.000 | 46.056 | 0.0 |
| GPU contact | 41.975 | 0.000 | 41.975 | 5.1 |
| GPU sparse | 53.132 | 19.652 | 72.784 | 1450.3 |

GPU contact's observed full loop was **1.097× faster**, a **8.9% time reduction**, than the fresh reference. The reference repeat took **46.249 s**. These are sequential observations, not statistical throughput estimates. Sparse construction happens during preflight and is added back explicitly. Contact upload during preflight was not separately timed. Common model construction, source/input validation, coordinate-HVP checks, terminal geometry auditing and Comet shutdown are excluded.

Memory values are the largest sampled live backend buffers at profile preflight, not total/peak GPU memory. Contact buffers are released after updates, so the terminal receipt reports zero resident contact bytes; that is not its working memory usage. Sparse buffers persist across states.

| Backend | HVP calls | Accepted PCG iterations | Rejected shift attempts |
| --- | ---: | ---: | ---: |
| Reference | 6745 | 4539 | 292 |
| GPU contact | 6752 | 4539 | 292 |
| GPU sparse | 6745 | 4539 | 292 |

The 200-step discrete branch records agree, but floating-point residual recomputations can change HVP counts. Accepted-PCG totals exclude work in rejected systems. Use the replay windows below for a controlled timing comparison.

## Matched-window profile

Each ten-step window has three uninstrumented repetitions, a separate residual-validation pass, a cProfile pass, and a CUDA-synchronized hierarchical pass. Initialization is excluded. The late window refreshes its first state's values inside timing. Initial sparse topology/union construction is measured separately from per-step work.

| Window | Reference s/step | GPU contact s/step | GPU sparse s/step |
| --- | ---: | ---: | ---: |
| 0–10 | 0.1640 | 0.1551 | 0.2338 |
| 100–110 | 0.2305 | 0.2093 | 0.2392 |
| 190–200 | 0.2606 | 0.2340 | 0.2608 |

GPU contact is **1.057–1.114× faster** across these matched windows. Use these fresh controls instead of the older profile recorded while another GPU compute job was active. Instrumented timings change GPU/CPU overlap and are for attribution. Contact HVPs reuse GPU CSR products instead of transferring every Krylov vector to CPU; IPC contact construction and CCD remain on CPU.

Late-window synchronized attribution (seconds for ten steps; nested rows must not be added):

| Component | Reference | GPU contact | GPU sparse |
| --- | ---: | ---: | ---: |
| PCG, including HVPs | 1.711 | 1.448 | 0.296 |
| FEM HVPs, within PCG | 1.215 | 1.203 | — |
| Sparse matrix refresh/cache checks | — | — | 1.353 |
| IPC broad phase, across CCD/updates | 0.453 | 0.457 | 0.456 |
| CCD kernel | 0.147 | 0.146 | 0.147 |

With GPU contact, PCG accounts for 59.7% of the 2.424 s instrumented window and FEM Hessian products alone account for 49.6%. IPC broad phase takes 18.9% and the CCD kernel 6.0%. Contact-product time falls from 0.336 s to 0.105 s. Full sparse assembly reduces PCG to 0.296 s but spends 1.353 s refreshing matrices/checking the cache, 51.5% of its 2.628 s window. This explains why faster sparse products do not make this solve faster overall.

## Numerical validation

Maximum checked GPU-HVP relative error against the original matrix-free FEM + CPU contact operator: **3.104e-16**, below the declared 1e-10 gate. All 30 accepted PCG solutions per backend in the validation windows were independently checked using that original operator. Worst relative residual: **0.000997482**, below 1e-3. Each replay also passed its saved endpoint energy/gradient tolerance (rtol 1e-8) and recorded PCG-iteration, shift-retry and line-search checks.

Full trajectories are not bitwise identical. All 200 accepted PCG counts, retry reasons/counts, backtracks and trial counts match the fresh reference. Continuous CCD/step-length factors differ slightly and are recorded separately. The following differences are diagnostics, not an assertion of exact trajectory equality:

| Backend vs fresh reference | Max terminal coordinate difference m | Max energy relative difference | Max gradient-norm relative difference |
| --- | ---: | ---: | ---: |
| GPU contact | 2.706e-08 | 3.495e-07 | 4.878e-06 |
| GPU sparse | 3.902e-10 | 9.571e-08 | 4.620e-07 |

The same-code reference repeat differs by at most **1.223e-08 m** at the terminal coordinates, with **0** differing discrete branch records. Derived initial norm/threshold values are checked to rtol 1e-14 (the sparse run differs by one ULP); solver input settings match exactly.

Maximum absolute CCD/step-length factor difference is **2.146e-07** for the GPU backends versus **3.433e-06** for the reference repeat. These differences occur at iteration 14 for the GPU runs and iteration 28 for the repeated reference.

The earlier `forward-003` 100 + 100 continuation follows a different path from the fresh reference: its first discrete difference is iteration **45**. At state 44 the fresh run rejects one additional shifted system for nonpositive curvature. Tiny prior numerical differences therefore change a discrete decision and amplify. This occurs in the original backend too, so it is not evidence of a GPU-backend policy change. The exact source of floating-point drift has not been isolated. Historical endpoint inversions: 62; fresh endpoint inversions: 68.

All fresh endpoints remain **nonconverged**, with gradient about **5.103e-07**, versus the **5.450e-08** threshold, **68 inverted tetrahedra**, and feasible soft-rigid contact. These results validate the backend comparison, not a physically valid neutral equilibrium. Selected-neutral assets remain unchanged.

![Trajectory comparison](../data/gpu-integration-review-003/trajectory-validation.png)

## Integration and cache contract

`src/liblaf/apple/forward/hessian` contains the promoted FEM BSR assembler, cached free-DOF GPU CSR assembler, GPU contact cache and `HessianProblem`. The adapter belongs to one solve and does not patch a model class or change implicit-adjoint execution. Numerical values refresh after state updates; the unshifted physical matrix is reused during PCG and shift retries. Energy, gradient, exact diagonal and CCD delegate to the original problem. Explicit backends fail visibly rather than falling back.

The promoted code checks free-DOF/topology mutation tokens. A global topology-cache regression was fixed so a new operator cannot inherit stale slots after Torch connectivity changes. Warp connectivity has no cheap mutation version and must remain immutable for the model lifetime. Unsupported FEM types fail explicitly. This validation covers the neutral model's StableNeoHookeanStress/StableNeoHookeanMembrane registry. The sparse profile predates the topology-cache guard fix; the full sparse run uses the final code. The fix does not change numerical assembly for fixed topology.

## Reproduction and evidence

Run from this experiment group with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYVISTA_OFF_SCREEN=true` and descriptive `CHERRIES_NAME`/`CHERRIES_TAGS`. Every run requires a fresh output directory.

```sh
.venv/bin/python src/10-run-neutral.py \
  --max-steps 200 --hessian-backend gpu_contact --output-dir data/new-forward
.venv/bin/python src/40-profile-forward.py \
  --hessian-backend gpu_contact --output-dir data/new-profile
```

Replace `gpu_contact` with `matrix_free` or `gpu_sparse` for the other arms. Forwards intentionally exit 1 after saving 200 steps when the force/geometry gate fails; profiles exit 0. All runs must complete Cherries shutdown. The early `profile-gpu-contact-001.stdout.log` records an enum spelling rejection before solver work; the successful contact profile is `002`.

Regression validation: `pytest -q tests/forward tests/solvers --no-cov` passed all 12 tests, including GPU contact cache behavior, sparse mapping/topology contracts, static forward mechanics and existing solver tests. Targeted Ruff checks passed. The first report attempt stopped because it treated continuous CCD factors as discrete branch labels; the corrected report preserves those differences as diagnostics and changes no solver output.

- Reference: [forward receipt](../data/forward-reference-200-001/summary.json), [profile receipt](../data/profile-reference-002/summary.json), [Python profile](../data/profile-reference-002/window-190/cprofile.txt), [timing tree](../data/profile-reference-002/window-190/synchronized.json).
- GPU contact: [forward receipt](../data/forward-contact-200-001/summary.json), [profile receipt](../data/profile-gpu-contact-002/summary.json), [Python profile](../data/profile-gpu-contact-002/window-190/cprofile.txt), [timing tree](../data/profile-gpu-contact-002/window-190/synchronized.json).
- GPU sparse: [forward receipt](../data/forward-sparse-200-001/summary.json), [profile receipt](../data/profile-gpu-sparse-001/summary.json), [Python profile](../data/profile-gpu-sparse-001/window-190/cprofile.txt), [timing tree](../data/profile-gpu-sparse-001/window-190/synchronized.json).
- [Reference repeat](../data/forward-reference-200-002/summary.json).
- [Independent audit](../data/gpu-backend-audit.json), [test log](../data/gpu-integration-tests.log), [source validation](../data/gpu-integration-code-validation.json). Library changes after benchmarking are formatting only, with identical parsed ASTs against the archived full-run sources.

[Machine-readable comparison](../data/gpu-integration-review-003/summary.json) retains hashes and numerical differences. This report/plot script performs zero forward solves. Source session: ‘What’s the main performance bottleneck? Can we improve performance? Related works include VBD, GIPC, etc.’ (`01a0c4dd-55f2-7270-959b-3852514f93e1`); original prototypes remain intact.
