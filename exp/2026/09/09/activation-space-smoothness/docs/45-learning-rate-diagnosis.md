# Learning rate and distorted intermediate states

The learned-axis rate of **7.86 is a strong suspect in the unstable Axis-off trajectory**, and the short calibration did not establish that it was suitable for a long run. The existing evidence does not isolate learning rate as the only cause of distortion.

## What the calibration actually checked

The revised calibration chose the largest tested rate because it gave the most fitting progress over 16 updates. Its `stable` flag required successful finite inner solves, positive fitting progress, and final loss within 1% of the best post-update loss. **It did not require inversion-free geometry.** The selected pilot already had 225 inverted tetrahedra.

| Fit-only pilot rate | Fit RMS at update 16 (mm) | Motion RMS (mm) | Inverted tetrahedra | First inverted update |
| ---: | ---: | ---: | ---: | ---: |
| 0.982 | 5.302 | 0.015 | 0 | None through 16 |
| 1.964 | 5.233 | 0.155 | 0 | None through 16 |
| 3.929 | 4.376 | 1.760 | 1 | 14 |
| **7.858, selected** | **3.028** | **4.210** | **225** | **8** |

![Four learning-rate calibration pilots](../data/44-report-figures-v2/learning-rate-calibration-trajectories.png)

All four start from the same learned-axis controls and fresh Adam state. The slower pilots also reach much less motion. Their lower inversion counts at update 16 therefore do not prove that they would remain undistorted at comparable fit or deformation. No lower-rate primary run was continued to that comparison point.

The half-rate pilot is particularly informative: its first inversion occurs at update 14 with **1.122 mm motion RMS**, versus update 8 with **1.167 mm** at the selected rate. The corresponding fit RMS values are 4.724 and 4.698 mm. Halving the rate delays the first inversion in update count, but both encounter it at a similar attained deformation. Lowering the rate alone has not been shown to remove the distortion.

## What happened in the primary run

| Axis-off accepted update | Fit RMS (mm) | Motion RMS (mm) | Inverted tetrahedra |
| ---: | ---: | ---: | ---: |
| 8 | 4.698 | 1.167 | 1 |
| 15, best fit | 3.539 | 4.122 | 541 |
| 16 | 3.601 | 4.472 | 1,035 |
| 28, last accepted | 6.038 | 8.302 | 12,657 |

The fit worsens after its best state while distortion grows. Attempted update 29 fails the fixed forward-solver limit. These are optimizer iterates, not a simulated time sequence of facial motion. The displayed intermediate shapes are exact saved equilibria; the distortion is not produced by plot interpolation or camera settings.

The on-run uses the same rate and reaches 128 updates, but retains 94 inverted tetrahedra. Its better outcome does not make the chosen rate or resulting geometry mechanically validated. The separate calibration-to-primary audit also found substantial trajectory sensitivity despite matching archived implementation and settings.

## Why Adam and the prior do not prevent this

The outer update is `Δq = −η m̂ / (√ŝ + ε)`, with fixed η, β = (0.9, 0.999), and ε = 0.01. Adam normalizes using gradient history; this runner has no outer loss-decrease test, backtracking, step-size cap, or rejection based on tetrahedron inversion. The inner equilibrium solver's line search works at fixed activation and does not select the outer activation step.

Learned-axis controls are nonlinear: C = vvᵀ and Z = (2 + ‖v‖²)vvᵀ. A fixed change in v can produce a much larger change in Z as v grows. S(C) penalizes spatial differences, not activation magnitude or det(F). Spatially uniform but excessive contraction can therefore evade this prior.

For a causal check, a new run should change only the rate while preserving initialization, Adam reset, physics, objective, and solver policy, then compare attained fit, motion, and inversion counts. That check has **not** been run here. The present evidence supports revisiting the rate and calibration criterion, rather than attributing the entire distorted trajectory to one confirmed cause.

## Evidence

The [pilot endpoint CSV](../data/45-learning-rate-evidence/pilot-endpoints.csv), [primary milestones](../data/45-learning-rate-evidence/primary-milestones.csv), and [extraction receipt](../data/45-learning-rate-evidence/summary.json) are derived from existing traces only. No forward or adjoint solve was rerun. See [calibration revision](15-calibration-revision.md), [calibration-to-primary audit](../data/18-calibration-main-divergence-audit/summary.json), and [extraction source](../src/45-extract-learning-rate-evidence.py).

Run from this experiment directory:

```bash
CHERRIES_NAME='Learning rate and distortion evidence from saved traces' \
CHERRIES_TAGS='activation-space,learning-rate,distortion,read-only,report' \
.venv/bin/python src/45-extract-learning-rate-evidence.py
```

The completed extraction is recorded in [Comet](https://www.comet.com/liblaf/apple/b513847b3f5a4e4a86f1af9c580c642f).
