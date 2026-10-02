# Free activation: all four heights at 1,200 Adam updates

[16:9 PNG, 3840 × 2160](../data/190-completed-height-figures/free-activation-height-comparison-16x9.png) · [Vector SVG](../data/190-completed-height-figures/free-activation-height-comparison-16x9.svg) · [Vector PDF](../data/190-completed-height-figures/free-activation-height-comparison-16x9.pdf) · [Verified metrics](../data/190-completed-height-figures/comparison.json)

![Completed height comparison](../data/190-completed-height-figures/free-activation-height-comparison-16x9-preview.png)

## Request and continuation policy

The user requested continuing through occasional forward failures instead of terminating the height comparison early. All four heights now reach the original total budget of **1,200 Adam updates**. This figure supersedes the [early-stopped comparison](170-free-height-comparison.md).

The h = 0.05 endpoint already completed this budget. The other three resume their original checkpoints at steps 515, 38, and 262, preserving activation, displacement, Adam moments/variance, global update counter, and learning-rate decay. The first resumed proposal reproduces the originally failed proposal **bit-for-bit** for every height.

The established rest-reset continuation policy is applied consistently: a recoverable forward failure returns the last assembled finite Newton iterate; the outer loop computes an approximate off-equilibrium adjoint and advances Adam. The next forward solve starts from zero displacement after a failure, or from the previous displacement after a converged solve. There is no early outer stop on recoverable forward failure, a small positive physical determinant, or a numerical plateau. Nonfinite states/gradients, inverted accepted states, or unusable adjoints still fail visibly. No rejected Armijo trial is used.

The original 100 × 10 layered mesh, muscle band, materials, physical J = det(F), unrestricted symmetric activation, zero smoothness, L2 objective, Adam settings, forward tolerance 1e−10, and 250-iteration forward limit remain fixed. Here h is peak target vertical displacement in y_target = 0.1 + 4 h x (1 − x). The relaxed failure policy changes the trajectories after their first failed forward evaluation.

## Final results

| h | Adam update | L2/h² | Failed forward solves | Rest-seeded solves | Final force residual | Final forward status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 1200 | 0.441198 | 0 | 0 | 5.35e−16 | Equilibrium |
| 0.10 | 1200 | 0.547075† | 685 | 684 | 3.02e−3 | Not converged |
| 0.15 | 1200 | 0.467659† | 1125 | 1124 | 7.46e−4 | Not converged |
| 0.20 | 1200 | 0.436305 | 7 | 7 | 1.56e−15 | Unstable equilibrium |

† Loss is evaluated at the final returned off-equilibrium iterate. These two numbers are not equilibrium-fit scores. Budget completion is distinct from forward convergence and inverse convergence.

For h = 0.10, the original iteration-limit failure occurs at 516, followed by line-search failures through 1200. For h = 0.15, the continuation contains 38 converged evaluations, one iteration-limit failure, and 1,124 line-search failures. For h = 0.20, failures at 263–269 are followed by converged solves through 1200. All per-step failure records are preserved in [forward-failures.json](../data/190-completed-height-figures/forward-failures.json) and the numerical traces. The final scheduled rest reset is not consumed when the last evaluation itself fails, explaining the one-count difference between failures and rest-seeded solves for h = 0.10 and 0.15.

| h | Minimum physical J | Inverted triangles | Smallest displacement-Hessian eigenvalue |
| ---: | ---: | ---: | ---: |
| 0.05 | 0.466046 | 0 | +1.47601e−5 |
| 0.10 | 1.000013e−8 | 0 | −0.206683 |
| 0.15 | 1.000277e−8 | 0 | −0.027987 |
| 0.20 | 0.0167591 | 0 | −0.00768899 |

The h = 0.10 and 0.15 states lie near the forward solver's positive-J floor; repeated rest resets did not recover equilibrium. Their Hessian values describe off-equilibrium states and are not equilibrium-stability diagnoses. The h = 0.20 endpoint is force-balanced but has negative Hessian curvature, so it is labeled an unstable equilibrium. It reproduces the previously verified rest-reset endpoint exactly. No inverse-convergence claim is made for any height, and the losses do not establish a ranking of attainable physical fit.

## Verification and artifacts

The renderer independently reconstructs B from controls, reassembles each endpoint's force residual and physical J, recomputes L2, and checks the minimum symmetric Hessian eigenvalue. It verifies contiguous trace indices, every failure/reset transition, checkpoint/history equality, and all four final update indices. The h = 0.20 controls, moments, variance, displacement, and B arrays are bitwise identical to the previous `140-reset-continuation` endpoint.

The PNG is exactly 3840 × 2160 pixels. All eight panels retain equal physical scales and bounds x ∈ [−0.02, 1.02], y ∈ [−0.01, 0.31]. Glyphs show both eigenmodes of B − I transported by F, with fixed length 0.0168995 and a common signed color range ±3.831307. The SVG has no embedded raster images. The preview was visually inspected; headings, annotations, and footer fit the slide. [Delivery checks](../data/190-completed-height-figures/delivery-checks.json) retain these results. Ruff and formatting checks pass for the changed/new scripts.

## Reproduction and receipts

Working directory: `exp/2026/09/15/activation-direction-smoothness`.

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Complete free activation at every target height' \
CHERRIES_TAGS='2d,free-activation,height-sweep,smoothness-off,continuation,forward-reset' \
uv run python src/180-continue-free-heights.py > logs/180-continue-free-heights-terminal.log 2>&1

COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Completed 1200-update height comparison slide' \
CHERRIES_TAGS='2d,free-activation,height-sweep,smoothness-off,1200-updates,16x9,figures' \
uv run python src/190-render-completed-heights.py > logs/190-render-completed-heights-terminal.log 2>&1
```

Both processes completed with exit 0 after Cherries shutdown. Each numerical case has its own protocol, checkpoint hash, source snapshot, trace, history, and summary under `data/180-free-height-continuation/h100`, `h150`, or `h200`. Output directories are protected against overwriting; choose a fresh output for reproduction. Rendering defaults reference this recorded continuation output. Existing unrelated working-tree changes were preserved, and automatic Git commits were disabled.

- Numerical run: [Comet](https://www.comet.com/liblaf/apple/567e5f041f4543e2a63672ff190f3009), [Comet.ml Experiment Summary](../data/180-free-height-continuation/comet-summary.txt), [terminal log](../logs/180-continue-free-heights-terminal.log).
- Figure run: [Comet](https://www.comet.com/liblaf/apple/3a3fcbddd7f74fe599d7865351027458), [Comet.ml Experiment Summary](../data/190-completed-height-figures/comet-summary.txt), [terminal log](../logs/190-render-completed-heights-terminal.log).
