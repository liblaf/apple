# One cold-start forward solve: Newton-CG-only versus hybrid

At least one arm did not reach a valid equilibrium within its declared budget. No matched-convergence speedup ratio is claimed.

| Method | Forward time | PNCG counter | Newton steps | Last recorded force | Valid equilibrium | HVP calls |
| --- | --- | --- | --- | --- | --- | --- |
| newton_diag | 204.12 s | 0 | 100 | 7.2515651e-08 | No | 46,352 |
| hybrid_diag | 600.12 s | 2289 | 0 | 1.6619475e-05 | No | 0 |

On this case, direct Newton made substantially more force-residual progress in less elapsed time. Hybrid spent its entire budget in coarse PNCG and never reached Newton. This revises the expectation that the PNCG phase necessarily helps a difficult cold start. The stopping causes differ, so these durations do not measure time to the same converged solution. More Newton iterations could change its outcome, but that extension was not run.

## Fixed problem and timing

Both arms use the saved zero-smoothing Smile stress at accepted update 16, with jaw angle zero. They start from the same contact-valid, loaded neutral displacement from update 0. There is no new Adam proposal, expression displacement warm start, adjoint solve, load continuation or outer optimization update. This is a substantial activation jump from neutral, not the earlier small first-proposal probe.

The initial free-force norms were 1.267878263e-05 and 1.267878263e-05. Activation, jaw and neutral-seed tensor hashes are identical. The physical model retains fixed neutral prestress, skin membrane and complete cranium, mandible and eyeball contact.

Newton-CG-only ran first, followed by hybrid, sequentially on the same Paratera RTX 4090 with eight IPC threads. Both use force tolerance `1e-8`, CG relative tolerance `1e-3`, scalar diagonal preconditioning, shift reset, CCD and Armijo. Hybrid switches at `max(1e-8, 0.001 * initial_force, 1e-7)`. Each arm has a 600-second forward wall budget and a 100-Newton-step limit; hybrid additionally performs coarse PNCG. A failed arm is retained and never silently replaced.

This is a cold displacement start with prewarmed operators, not a cold process. Model/input construction, fixed-state kernel prewarm and post-solve metrics/audits are excluded. Complete forward solves are CUDA-synchronized at their boundaries. Each primal reconstructs its own contact state after prewarm. This is one observation per method, in fixed order, without a confidence interval.

## Solver work and endpoint checks

- **newton_diag**: 99 shifted Newton steps; 558 linear retries costing 100.99 s; successful linear solves cost 92.95 s. These times are included in total forward time. There were 0 PNCG curvature evaluations and 100 inner CCD queries, excluding the initial boundary query.
  Failure: `Newton iteration budget exhausted`. This is failure within the declared budget, not proof that the method cannot converge.
- **hybrid_diag**: 0 shifted Newton steps; 0 linear retries costing 0.00 s; successful linear solves cost 0.00 s. These times are included in total forward time. There were 2290 PNCG curvature evaluations and 2290 inner CCD queries, excluding the initial boundary query.
  Failure: `declared forward wall budget exhausted`. This is failure within the declared budget, not proof that the method cannot converge.

Comparison receipt: `{"comparable": false, "reason": "one_or_both_forwards_failed"}`.

For an interrupted PNCG solve, the force and iteration counter come from the last computed optimizer gradient in its failure receipt. They are not an independent force recomputation on a saved terminal displacement. Failed arms do not produce validated endpoint geometry or adjoint gradients. PNCG force norms need not decrease monotonically even when its energy line search succeeds.

## Reproduction and evidence

Run from `exp/2026/09/22/solver-performance` on the compute host, using a new output directory:

```sh
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=cold-smile-newton-vs-hybrid CHERRIES_TAGS=smile,performance,cold-forward,newton,hybrid \
  python -u src/43-compare-cold-forward.py \
  --checkpoint data/inverse-duration-hard-001/zero_smoothing/historical/expressions/Smile/latest.pt \
  --neutral-checkpoint data/smile-fit-adam03-no-smoothness-005/arms/hybrid_diag/expressions/Smile/initial.pt \
  --output-dir data/NEW-RUN
```

Raw outputs, full traces, source archives and input hashes: `data/cold-forward-comparison-001`. Local Cherries logging was enabled with remote Comet disabled. The process can complete successfully while a solver arm fails its declared convergence budget; read the per-arm results. No production solver or physics setting was changed.

All 282 copied compute-host files passed SHA-256 verification. The log also contains three missing legacy `data/simple-skin-forward` asset-registration warnings; the explicit numerical/source receipts are present. These logging warnings were not the solver stopping causes.
