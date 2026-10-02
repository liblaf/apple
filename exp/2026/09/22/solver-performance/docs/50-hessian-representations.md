# Frozen Hessian representation microbenchmark

Assembled sparse FEM is fastest in both frozen-state tests: 2.24–2.37× faster than matrix-free FEM with the same GPU contact, including numeric matrix refresh plus PCG. Cache mesh topology once, rebuild numerical values at each Newton state, and reuse the matrix through CG and shift retries. This is a linear-system result; end-to-end forward and inverse speedup is not yet measured.

| State | Variant | HVP ms | CG s | Refresh s | Refresh + CG s | CG iters | Extra GPU MiB | Max relative error | HVP x CPU | HVP x GPU | GPU break-even applies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold_loaded_neutral | current FEM + CPU contact | 3.68 | 4.57 | 0 | 4.57 | 1088-1117 | 0 | 6.91e-17 | 1 | 0.729 | not faster |
| cold_loaded_neutral | current FEM + GPU contact | 2.68 | 3.3 | 0 | 3.3 | 1097-1105 | 0 | 6.68e-17 | 1.37 | 1 | not faster |
| cold_loaded_neutral | cached FEM + GPU contact | 2.52 | 2.94 | 0.00358 | 2.94 | 1077-1120 | 1.14e+03 | 8.41e-17 | 1.46 | 1.06 | 22 |
| cold_loaded_neutral | assembled BSR FEM + GPU contact | 0.86 | 1.24 | 0.228 | 1.47 | 1097 | 239 | 4.10e-16 | 4.28 | 3.12 | 125 |
| saved_smile | current FEM + CPU contact | 3.66 | 6.47 | 0 | 6.47 | 1592-1601 | 0 | 6.70e-17 | 1 | 0.733 | not faster |
| saved_smile | current FEM + GPU contact | 2.69 | 4.72 | 0 | 4.72 | 1582-1596 | 0 | 1.08e-16 | 1.36 | 1 | not faster |
| saved_smile | cached FEM + GPU contact | 2.36 | 4.23 | 0.00342 | 4.23 | 1582-1599 | 1.14e+03 | 1.11e-16 | 1.55 | 1.14 | 10.5 |
| saved_smile | assembled BSR FEM + GPU contact | 0.823 | 1.77 | 0.227 | 2 | 1590 | 239 | 3.53e-16 | 4.45 | 3.26 | 122 |

## Scope and timing

This is a frozen-Newton-system microbenchmark at two saved states. It measures exact Hessian-vector products and scalar diagonally preconditioned PCG with the same RHS, shift, diagonal and relative tolerance. It does not validate an end-to-end forward solve, an inverse update, or a converged-solution wall-time improvement.

`cached FEM` stores frozen bulk invariants, membrane derivatives, and copied material arrays. This prototype copies all three full-mesh bulk potential fields, so it uses more memory than the assembled matrix; this is not a fundamental memory lower bound for matrix-free methods. `assembled BSR FEM` stores the full assembled FEM block-sparse representation. Both retain the common GPU contact contribution. The CPU-contact current route is included as a control for the contact transfer path.

Construction/first-call timing may include JIT and topology setup. Numeric refresh is reported separately and is the setup term used in the break-even calculation. Break-even is undefined when a candidate does not improve on `current_gpu_contact`; the table says `not faster` rather than reporting a negative apply count.

## Startup cost and amortization

For `cold_loaded_neutral`, assembled BSR first construction cost 29.39 s (24.20 s topology; JIT may also be included). Including a fresh numerical matrix for each later state, that startup amortizes after about 16 comparable Newton linear solves against matrix-free FEM with GPU contact.

These startup figures must remain separate from numeric refresh and repeated apply timing. The per-state HVP break-even in the table uses numeric refresh only. The construction amortization above subtracts numerical refresh from the per-state PCG saving; it is conditional on similarly sized later Newton linear solves.

The reported extra GPU MiB is explicit persistent operator storage. Torch allocator telemetry excludes Warp/native allocations; CUDA memory telemetry covers the wider device allocation scope. CPU sparse-topology storage is not included in the GPU storage number. Exact relative accuracy is the largest checked relative L2 operator difference against the CPU-contact reference.

The shifted CG calibration and its `solution_relative_difference` receipt use `current_gpu_contact` as the solution reference. The CPU-contact route is the independent product-accuracy and true-residual reference. These are different comparisons.

`cold_loaded_neutral` force L2: 1.268e-05.
`saved_smile` force L2: 3.669e-09; it is already below the 1e-8 forward force target, so its CG result is an artificial residual probe rather than a necessary Newton iteration.

## Numerical validation and reproducibility

Three deterministic directions per state check exact unshifted HVP agreement against the original CPU-contact route; the maximum relative L2 difference is below 4.1e-16. All 24 measured PCG solves pass the original-operator true residual check at relative tolerance 1e-3. Cold state shift is zero. The saved Smile residual probe requires a common shift of 2.8459e-7 after three 2000-step calibration budgets; those calibration attempts are excluded from the one-successful-solve timing. This benchmark does not test unshifted implicit-adjoint solve speed.

One RTX 4090, float64, eight IPC threads, 30 synchronized HVP samples and three PCG repeats per state and variant. The same physical source hashes and checkpoint/input hashes match the prior cold-start experiment. Physical model: tissue and skin with active stress, fixed prestress, cranium, mandible and eyeball contact. Different iteration counts within roughly the same range reflect floating-point summation order, not different tolerances. For PNCG or very short CG solves, matrix rebuild cost may not amortize; the measured sparse crossover is about 122–125 HVP applications per state.

Run from `${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/exp/2026/09/22/solver-performance`:

```bash
DEBUG=1 CHERRIES_NAME="Frozen Hessian representations 001" CHERRIES_TAGS="solver-performance,hessian-cache,smile,rtx4090" ${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python src/49-benchmark-hessian-representations.py
```

Cherries completed successfully with `states_completed: 2`; `DEBUG=1` kept local logs and disabled remote Comet recording. Full terminal log: `tmp/hessian-representations-001-terminal.log`. Source archives and raw timing arrays are retained beside the summary. The imported legacy benchmark registered an unused missing `data/simple-skin-forward` asset, producing a shutdown warning; the requested benchmark outputs and completion receipt exist.

## Evidence

Source summary: `exp/2026/09/22/solver-performance/data/hessian-representations-001/summary.json` (`SHA-256 cd91f3ca21e473f3466013f4e1ec3b72b715f33d3c6ee88090e594bdc5d1fe27`). Figure assets: `hessian-representations.png` and `hessian-representations.svg`. The benchmark ran variants sequentially on one GPU; repeat medians are descriptive measurements, not a throughput estimate for concurrent solves.
