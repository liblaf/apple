# cuDSS direct solves on exact free-space Hessians

CG remains faster for one RHS at the current 1e-3 tolerance. Cholesky costs about 1.19 s to factor, then 25.5 ms per reused solve, and gives much smaller residuals. For similarly costly RHS solves, factors pay off at roughly two RHS on the same matrix after symbolic analysis is amortized. This is a cost model from repeated solves of one RHS, not a measured independent multi-RHS workload.

| Frozen state | Free-CSR PCG | Cholesky factor + solve | LDLT factor + solve |
| --- | ---: | ---: | ---: |
| cold_loaded_neutral | 0.806 s | 1.215 s | 1.358 s |
| saved_smile | 1.163 s | 1.215 s | 1.358 s |

Times above exclude symbolic analysis and common free-matrix construction. All GPU tests ran sequentially on one RTX 4090, using float64 and 597,177 free DOFs. The saved state is already converged; its shifted system is a residual probe, not a necessary forward iteration.

This report compares scalar-diagonal PCG with cuDSS Cholesky and LDLT on the same assembled free FEM-plus-IPC systems at two frozen Smile states. It reports direct-solver device work and the separate CPU sparse merge/restriction cost. It does not claim an end-to-end Newton, forward, or inverse speed-up.

## Matched shifted Newton systems

| State | Mode | Status | Free CSR build s | PCG median s | Symbolic s | First factor s | Factor median s | Reuse solve median s | Factor + reuse solve s | Factor nnz | Symbolic estimate MiB | Sampled device peak MiB | Max true residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold_loaded_neutral | spd | success | 3.34 | 0.806 | 10.8 | 1.24 | 1.19 | 0.0255 | 1.22 | 4.01e+08 | 4050.1 | 7246.2 | 2.57e-13 |
| cold_loaded_neutral | symmetric | success | 3.34 | 0.806 | 11 | 1.33 | 1.33 | 0.0247 | 1.36 | 4.01e+08 | 4050.1 | 7248.2 | 2.62e-13 |
| saved_smile | spd | success | 3.32 | 1.16 | 11.2 | 1.19 | 1.19 | 0.0255 | 1.21 | 4.01e+08 | 4050.1 | 7248.2 | 3.83e-13 |
| saved_smile | symmetric | success | 3.32 | 1.16 | 10.7 | 1.33 | 1.33 | 0.0249 | 1.36 | 4.01e+08 | 4050.1 | 7248.2 | 3.83e-13 |

The `Free CSR build s` column is prominent because it includes CPU SciPy conversion, collision scatter, global addition, free-DOF restriction, and GPU upload. It is excluded from `Factor + reuse solve s`; adding it is necessary when the matrix must be rebuilt at a new Newton state.

The earlier assembled-BSR FEM plus GPU-contact benchmark measured PCG medians of **1.24 s** for loaded neutral and **1.77 s** for saved Smile. Those are the earlier split FEM/contact sparse-operator PCG times, not full direct-method totals. The new merged free-CSR PCG is faster after assembly, but its current CPU conversion takes about 3.3 s per state including FEM refresh, so this prototype does not yet improve total forward runtime. This points to caching the free-space structure and filling values directly on GPU as a separate optimization.

## Saved-state unshifted residual probe

| State | Mode | Status | Free CSR build s | PCG median s | Symbolic s | First factor s | Factor median s | Reuse solve median s | Factor + reuse solve s | Factor nnz | Symbolic estimate MiB | Sampled device peak MiB | Max true residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| saved_smile | symmetric | success | 3.18 | — | 10.8 | 1.33 | 1.33 | 0.0247 | 1.36 | 4.01e+08 | 4050.1 | 7248.2 | 1.67e-12 |

The saved Smile state is already a small-residual state. Its unshifted LDLT record is a separate numerical probe, not an adjoint or production Newton timing. LDLT reports inertia `[597177, 0]` (positive, negative) for both physical unshifted snapshots with pivot epsilon zero. The saved unshifted matrix was tested with LDLT only; Cholesky was tested on the cold unshifted and saved shifted matrices. These numerical diagnostics support positive definiteness of these snapshots, not every state in the nonlinear trajectory.

All 40 direct solves pass independent original-operator residual checks: cold/shifted relative residuals are below 3.84e-13, and the unshifted saved probe is below 1.68e-12. The direct acceptance threshold was 1e-7; PCG used 1e-3. The maximum assembled-operator relative error is below 3.66e-16. Thus the timing comparison uses the accuracy sufficient for our current forward solves; higher-accuracy PCG was not benchmarked here.

## Interpretation limits

Both matched Cholesky tests succeeded. LDLT success alone does not establish positive definiteness; the inertia diagnostics provide that evidence for these snapshots. `Factor nnz` is the cuDSS-reported factor nonzero count. The symbolic memory estimate is reported by cuDSS; sampled device peak is whole-device NVML usage sampled every 10 ms, so it can miss brief peaks and includes other allocations.

cuDSS reports 401,335,200 factor nonzeros versus 23,995,107 full free-CSR entries (about 16.7 times as many). Its estimated solver peak device requirement is 4,246,885,492 bytes (3.96 GiB). Observed total device usage peaked at about 7.08 GiB, including the model, full/lower matrices and solver workspaces. Full plus lower free-CSR GPU storage is about 563 MiB; cuDSS itself needs only the lower input, while the full matrix is retained for PCG. CPU process lifetime high-water RSS before/after conversion is in raw receipts; it is not a separately sampled merge-only peak.

Symbolic analysis costs 10.7–11.2 s and is excluded from the displayed numeric factor-plus-solve times. Numerical factorization uses the same analyzed pattern, while values are freshly factored on each repeat; this is not reuse of numerical factors for changing matrices. The first full-mesh Cholesky solve costs 80.9 ms versus about 25.5 ms for later solves. Cold FEM topology construction is also separate (26.75 s in this process).

All three tested free matrices have exactly the same CSR pattern hash, despite changed contact counts. This permits symbolic reuse for these patterns; full-mesh cross-state refactorization was not timed here. A small GPU check separately validated value-update/refactorization.

Symbolic analysis can only be reused when the free-system dimension, matrix type, CSR ordering/index base, and sparsity pattern are all identical. The benchmark analyzed each case independently; reuse at additional states requires verifying the free pattern again. Repeated numerical factorization and reused triangular solves are measured only after a successful analysis for that particular matrix pattern.

## Reproduction and implementation

The pinned `nvidia-cudss-cu13==0.7.0.20` wheel was installed into experiment-local `tmp/cudss-runtime`; existing environment packages and physical sources were unchanged. Wrapper ABI constants were checked against the installed header. GPU smoke checks cover SPD solves, numeric value updates/refactorization, symmetric-indefinite solves and inertia. Wheel/header hashes are saved in `tmp/cudss-install-provenance.json`.

Run from `${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/exp/2026/09/22/solver-performance`:

```bash
DEBUG=1 CUDSS_LIBRARY="$PWD/tmp/cudss-runtime/nvidia/cu13/lib/libcudss.so.0" CHERRIES_NAME="cuDSS frozen Smile comparison 001" CHERRIES_TAGS="solver-performance,cudss,factorization,smile,rtx4090" ${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python -u src/51-benchmark-cudss.py
```

Cherries finished successfully (`states_completed: 2`); debug retained local logs and disabled remote Comet recording. Each of the five matrix/method cases has three numeric factorizations and five further solves reusing factors, plus three post-factor solves. Matched PCG has three repeats per state. Terminal output is in `tmp/cudss-comparison-001-terminal.log`; the imported legacy helper emits an unrelated unused `simple-skin-forward` asset warning at shutdown.

The implementation uses [cuDSS analysis, factorization and solve phases](https://docs.nvidia.com/cuda/cudss/getting_started.html), with exact API details pinned by the archived 0.7 header. No production solver default was changed.

## Evidence

Source summary: `exp/2026/09/22/solver-performance/data/cudss-comparison-001/summary.json` (SHA-256 `24642613ccc0c2c6891e2f723f066401d80c0c557b67f7a8a219cf6866fae45e`). Figure assets: `cudss.png` and `cudss.svg`. Raw timing arrays, factor receipts, residual checks, and per-system matrix metadata are retained in [cudss-evidence.json](cudss-evidence.json).
