# Smile solver optimization results

22 September 2026. Sequential runs on the same Paratera RTX 4090, eight IPC threads.

The selected opt-in candidate is **adjoint relative tolerance `1e-4` with GPU contact products scoped to the adjoint linear solve**. It leaves the forward solver unchanged. It passed direct gradient and projected-Adam comparisons on two saved states. Newton shift reuse and GPU contact during forward solving remain experimental.

## Measured adjoint improvement

| Saved state | CPU contact, rtol 1e-7 | GPU adjoint only, rtol 1e-4 | Speedup |
| --- | --- | --- | --- |
| Regularized step 20 | 13.78 s | 5.85 s | 2.36× |
| Zero-smoothing step 19 | 29.51 s | 11.49 s | 2.57× |

These are single cold-start adjoint measurements with identical zero initial adjoints, excluding model construction and transfer. They are not whole inverse-update speedups or new 20-update fits. Existing production fits warm-start adjoints; their speedup may differ.

The CPU-only tolerance sweep already gave 1.67× and 1.80× at `1e-4` versus `1e-7`. Moving contact matvecs to the GPU gives the additional gain above. The complete five-point CPU and three-point whole-adjoint GPU sweep is plotted separately below.

## Direct accuracy against CPU rtol 1e-8

Each percentage is `100 * ||candidate-reference|| / ||reference||`. The reference and candidate use the identical saved displacement, active stress, material fields, boundary conditions and objective. No forward solve is run during these checks.

| Saved state | Data stress gradient | Total stress gradient | Jaw gradient | Projected Adam stress update |
| --- | --- | --- | --- | --- |
| Regularized step 20 | 0.00262% | 0.00001% | 0.00478% | 0.00041% |
| Zero-smoothing step 19 | 0.00264% | 0.00264% | 0.03136% | 0.00535% |

All values pass the provisional 0.1% engineering comparison gate. The next projected jaw update is exactly zero in both saved states because of its bound; raw jaw-gradient error is reported separately. This does not validate future jaw motion or accumulated trajectory error. The reference is tight differentiation at the saved approximate equilibrium, not a newly tightened primal equilibrium.

## Implementation and model

The physical adjoint solves the original unshifted Hessian system. The exact IPC contact Hessian is assembled on CPU once per owned state, uploaded as CSR, then applied on CUDA during CG/MINRES. CPU contact is restored before recomputing the true residual and fixed-boundary derivative. Installation and cleanup are exception-safe. This does not move broad phase, collision detection, CCD or all IPC operations onto the GPU.

The cached complete-model HVP microbenchmark fell from 3.779 ms to 2.938 ms (1.286×, five samples per route); first GPU upload plus HVP cost 11.564 ms. Three deterministic directions and a perturbed state agreed with CPU within 7e-17 relative L2. Cache refreshes were checked on state updates and state switches; the old-state replay tested upload invalidation, without a separate numerical oracle for that final replay.

The model remains active-stress volumetric tissue plus skin mechanics, with complete cranium, mandible and fixed eyeball collision. Collision surface counts: 34,245 soft vertices / 65,580 triangles; cranium 17,575 / 35,162; mandible 9,476 / 18,948; eyes 1,298 / 2,560. Contact is frictionless soft-versus-rigid IPC; soft-soft and rigid-rigid pairs are excluded. Forward force tolerance stays `1e-8`. Adam stays at 0.3, with no magnitude or jaw prior and no outer step rejection. The two saved states retain their respective original smoothing weights, including the zero-smoothing diagnostic.

## Why forward changes were not selected

The hybrid keeps PNCG until the switch threshold, then uses safeguarded Newton-CG with diagonal preconditioning, curvature-dependent shifts, CCD and Armijo. Shift reuse carries a normalized successful shift to the next Newton iteration; a revised guard resets it near convergence. The unchanged `reset` policy remains selected.

Easy forward replay: CPU reset 7.374 s; CPU reuse 7.775 s; GPU reset 5.755 s; GPU reuse 5.804 s. All passed the endpoint gate (skin RMS <= 0.001 mm and maximum full-node difference <= 0.01 mm).

Hard replay: CPU reset 75.933 s; CPU reuse 62.319 s; GPU reset 78.759 s; GPU reuse 76.065 s. Every solve passed force/contact/inversion checks, but every non-control variant failed endpoint agreement. Maximum node differences were 0.0311, 0.2260 and 0.1422 mm respectively. The earlier unguarded shift-reuse trial also failed; its apparent 2.44× speedup is not an accepted result.

An unchanged CPU-reset repeat took **51.875 s**, versus 75.933 s, with **0.0008562 mm skin RMS** and **0.0242778 mm maximum-node** difference. It too exceeded the maximum-node gate. PNCG coarse steps changed from 271 to 213 under identical parameters. Thus hard-case failures and timing differences cannot be assigned solely to the new variants. Sensitivity to floating-point reductions/contact branching is a plausible explanation, not a proven root cause. Restricting GPU changes to the adjoint preserves the existing forward implementation.

## Bottleneck and practical use

Late regularized-fit updates previously spent about 59.5% of their time in the adjoint, so this is a useful target. The zero-smoothing diagnostic spent about 85.7% in forward solving; its main bottleneck remains nonlinear equilibrium/contact behavior. A forward convergence and repeatability study is more valuable there than claiming a speedup from one trajectory.

Use these explicit options with `src/20-fit-smile.py`:

```sh
--adjoint-rtol 1e-4 --gpu-contact true --gpu-contact-scope adjoint --newton-shift-policy reset
```

Historical defaults remain unchanged (`1e-7`, GPU off, shift reset). No new full inverse trajectory has been run with the candidate settings.

## Commands and evidence

Run from `exp/2026/09/22/solver-performance`. Remote numerical runs used `${EXPERIMENT_WORKSPACE}/codex-apple-performance/apple/.venv/bin/python`, `DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8`, readable `CHERRIES_NAME`, and task tags. All GPU jobs were serialized. Local debug Cherries receipts were retained; there is no remote Comet run URL.

```sh
# Fixed-state sweep; set CHECKPOINT and OUTPUT to an unused run directory.
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=adjoint-scope CHERRIES_TAGS=smile,adjoint,scoped-gpu \
  python src/30-adjoint-tolerance-sweep.py --checkpoint "$CHECKPOINT" --output-dir "$OUTPUT" \
  --tolerances 1e-8,1e-4 --reference-rtol 1e-8 --forward-atol 1e-8 --ipc-threads 8 \
  --gpu-contact true --gpu-contact-scope adjoint
```

CPU reference runs used `--tolerances 1e-8 --gpu-contact false`. The earlier CPU ladder was `1e-8,1e-7,1e-6,1e-5,1e-4`; whole-adjoint GPU sweeps used `1e-8,1e-7,1e-4`. Forward replay used `src/32-replay-forward-optimizations.py` on each step-15 checkpoint; the unchanged repeat used `--variants reset_cpu`. Complete configurations, input/source hashes and raw vectors are retained per run.

Evidence directories under `data/`: `adjoint-tolerance-*-001`, `adjoint-tolerance-*-gpu-002`, `adjoint-reference-*-cpu-003`, `adjoint-scope-*-gpu-003`, `adjoint-scope-validation-001`, `gpu-contact-benchmark-002`, `forward-optimizations-easy-002`, `forward-optimizations-hard-001`, `forward-control-hard-repeat-002`, and `solver-optimization-visuals-001`. Failed `gpu-contact-benchmark-001` (driver state-selection error, corrected) and `shift-reuse-easy-001` remain preserved.

Validation: compilation and Ruff; CPU SPD/nonconvex solver checks; adapter install/uninstall, cache and exception-cleanup checks; actual GPU HVP equivalence; fixed-state gradient/Adam comparisons; forward force, collision and inversion checks. The changes are experiment-local and retain the existing dirty workspace. No commit or publication outside the temporary tailnet report was made.
