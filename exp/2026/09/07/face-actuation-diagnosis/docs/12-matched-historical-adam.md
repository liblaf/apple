# Matched historical Adam comparison

This runner isolates the effect of spatial smoothing while preserving the June no-skin inverse setup. Raw6 and Raw6-S start from rest and zero activation, use the same mesh, target, fixed boundary, active domain, materials, solvers, Adam settings, and 200-step budget. Their only method difference is the Raw6-S smoothness weight.

The initial executions were interrupted without terminal summaries after recorded steps 54 and 60. Their shared step-50 checkpoints are the inputs to the [controlled continuation](18-interruption-and-continuation.md), which applies an explicit Adam reset to both methods. The original traces remain archived; their declared 200-step budgets were not completed.

## Shared contract

| Setting | Value |
| --- | ---: |
| Volume vertices / tetrahedra | 228,660 / 1,146,517 |
| Original fixed vertices | 27,036 |
| Target vertices | 15,302 finite `IsFace` vertices |
| Active tetrahedra / muscle labels | 288,235 / 103 |
| Scalar controls | 1,729,410 symmetric-tensor components |
| Skin energy | 0 |
| Fat | E=0.003 MPa, nu=0.49, Stable Neo-Hookean |
| Muscle | E=0.03 MPa, nu=0.49, active Stable Neo-Hookean |
| Aponeurosis | E=0.1 MPa, nu=0.35, Stable Neo-Hookean |
| Volume Lamé convention | classical lambda passed directly to Stable energies, as in June |
| Data objective | uniform mean squared residual over vertices and Cartesian components, multiplied by 1e6 |
| Adam | lr=0.3, eps=0.01 |
| Forward solve | June PNCG, max 5,000, rtol=5e-4, atol=1e-10 |
| Adjoint solve | CG then MINRES, max 10,000 each, rtol=5e-4 |
| Budget / checkpoints | 200 Adam opportunities; checkpoint every 10 evaluations |

No SPD or deformation-Jacobian gate rejects an endpoint. `det(F)` and activation eigenvalues remain recorded diagnostics. Every checkpoint stores `forward_success`, `adjoint_success`, and `solver_valid`; `solver-receipts.jsonl` gives the full per-evaluation mapping. A viewer must omit frames whose `solver_valid` value is false.

The exported best endpoint requires successful forward and adjoint receipts. This is deliberately stricter than the June `BestState`, which required a successful forward solve and finite objective. Isolated finite unsuccessful solver evaluations remain visible in the trace and can still advance Adam, following the June mandatory-baseline behavior. Three consecutive unsuccessful evaluations restore the best valid endpoint, halve the learning rate, clear Adam moments, recompute its gradient, and skip that update. The final status distinguishes a completed fixed budget, unresolved trailing solver failures, a numerical stop, and interruption; none claims stationarity.

## Raw6-S penalty

Raw6-S uses the shared-face finite-volume energy

`L^2 sum_e[(area/distance) harmonic(MuscleFraction) ||H_i-H_j||_F^2 / 1.5] / [sum_t(volume MuscleFraction) (-log(0.8))^2]`.

Edges cross one shared tetrahedral face and connect cells only when their `MuscleId` values are identical. The graph has 501,409 edges. `L=0.005 m`; the packed symmetric tensor uses Frobenius weights `(1,1,1,2,2,2)`. At the saved June endpoint, the dimensionless graph energy is 124.804588. The candidate weight `5e-4 mm^2` contributes 0.0624023 mm^2, or 46.6% of that endpoint's 0.133797 mm^2 data objective. This is a training-target tuning heuristic, not an independently validated regularization weight.

## Verified CPU preflight

Both preflights passed without a visible CUDA device. Their receipts are `data/30-historical-adam-raw6-preflight.json` and `data/31-historical-adam-raw6-s-preflight.json`. They validate fixture cardinalities, material and solver constants, control count, graph size, regularization scale, and hashes without constructing a CUDA model or running an equilibrium solve.

## GPU commands

Run these from the experiment directory with the repository virtual-environment interpreter. Use separate, empty output directories. The recorded pair was launched concurrently on one GPU, along with the remaining current-fixture diagnostic. Its wall times therefore are not independent performance measurements. Sequential reproduction is also possible with the commands below.

```bash
cd exp/2026/09/07/face-actuation-diagnosis
env CUDA_VISIBLE_DEVICES=0 COMET_AUTO_LOG_GIT_METADATA=false COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_ENV_DETAILS=false CHERRIES_NAME='Historical matched Raw6 Adam no-skin' CHERRIES_TAGS='gpu,face,inverse,historical-matched,raw6,adam,no-skin' .venv/bin/python src/30-run-historical-adam.py --output-dir data/30-historical-adam-raw6 --smoothness-weight 0
```

```bash
cd exp/2026/09/07/face-actuation-diagnosis
env CUDA_VISIBLE_DEVICES=0 COMET_AUTO_LOG_GIT_METADATA=false COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_ENV_DETAILS=false CHERRIES_NAME='Historical matched Raw6-S Adam no-skin' CHERRIES_TAGS='gpu,face,inverse,historical-matched,raw6-s,adam,no-skin' .venv/bin/python src/30-run-historical-adam.py --output-dir data/31-historical-adam-raw6-s --smoothness-weight 0.0005
```

Each nonempty output directory is rejected before a run starts. A completed directory contains the resolved config, archived source files and hashes, trace, solver receipts, optimizer events when applicable, ten-step checkpoints, `latest.npz`, the best-valid `final.npz` and `final.vtu`, and an honest `summary.json`.
