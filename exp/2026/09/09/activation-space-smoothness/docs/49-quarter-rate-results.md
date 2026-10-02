# Quarter-rate results

Reducing the learned-axis Adam rate from 7.857986 to 1.964496 made physical updates smaller at the selected fit-and-motion matched states, but did not remove inversion at comparable surface motion RMS. The smoothed arm still developed a late increase in fit error and severe local inversion. This test supports rate sensitivity, not a general claim that a smaller fixed rate produces better geometry.

The [frozen protocol](47-conservative-rate-plan.md) changed only the rate within each arm. It retained the original initialization, fresh Adam state, epsilon, betas, objective, smoothness coefficient, fixture, and numerical implementation. Both arms completed 64 updates. Runtime did not support extending both arms under this pair's frozen cutoff. See the [execution record](48-conservative-rate-execution.md) for commands and run metadata. The subsequent user-requested [rate-0.3 test](54-rate-03-plan.md) is separate.

## First inversion

| Rate | Arm | First inverted update | Fit RMS (mm) | Motion RMS (mm) |
| --- | --- | ---: | ---: | ---: |
| 7.857986 | Off | 8 | 4.697652 | 1.166806 |
| 7.857986 | On | 8 | 4.697891 | 1.166659 |
| 1.964496 | Off | 25 | 4.713890 | 1.143371 |
| 1.964496 | On | 25 | 4.714298 | 1.142112 |

Inversion is delayed by 17 optimizer updates, but is first sampled near the same surface motion RMS. The 16-update quarter-rate pilot had not reached this amount of motion.

## Endpoints and best fit

| Arm | Selection | Update | Fit RMS (mm) | Motion RMS (mm) | Inverted tetrahedra | Minimum det(F) | Surface residual HP (mm) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | Endpoint and best fit | 64 | 2.587802 | 4.358874 | 409 | −1.946184 | 0.218834 |
| On | Best fit | 45 | 3.378381 | 3.674459 | 231 | −1.891935 | 0.237660 |
| On | Endpoint | 64 | 5.152102 | 7.308119 | 2,189 | −4.186610 | 0.609685 |

The smoothed arm's most negative det(F) was −27.583275 at update 63. All recorded forward and adjoint solves passed their configured success checks, while outer fit and total objective later increased. These facts must be reported together: solver completion does not certify acceptable geometry or outer convergence.

![Quarter-rate and original-rate trajectories](../data/52-conservative-rate-comparison/optimizer-trajectories.png)

## Matched saved states

Each row below compares actual saved states within 0.05 mm in both fit and motion, excluding initialization. Selection uses the lowest mean fit, then normalized mismatch. No interpolation is used. A positive surface-score change means the right-hand state is worse.

| Comparison, left → right | Updates | Fit difference (mm) | Motion difference (mm) | Surface residual HP (mm), left → right | Relative HP change | Inversions, left → right |
| --- | --- | ---: | ---: | --- | ---: | --- |
| Quarter off → quarter on | 41 → 45 | 0.022033 | 0.045608 | 0.232430 → 0.237660 | +2.25% | 90 → 231 |
| Original off → quarter off | 11 → 34 | 0.007985 | 0.037243 | 0.246860 → 0.257027 | +4.12% | 19 → 18 |
| Original on → quarter on | 13 → 44 | 0.023933 | 0.009415 | 0.231606 → 0.235492 | +1.68% | 81 → 214 |

The quarter-rate off/on match does not meet the predeclared 10% surface improvement threshold. At these states, S(C) is also higher on (967.821) than off (858.363); the on trajectory has followed a different path by this stage. This is an observed matched-state result, not a statement that the regularizer has been omitted or its coefficient changed.

At the cross-rate off match, the most recent face displacement update decreases from 0.868 to 0.279 mm RMS; at the on match, it decreases from 0.463 to 0.245 mm. The corresponding Frobenius-RMS changes in Z decrease from 24.85 to 8.60 off and from 38.16 to 17.73 on. Thus the nominal rate reduction also produces smaller physical updates at these states. It does not produce a better measured surface residual in either matched comparison.

![Surface score and inversions at attained fit and motion](../data/52-conservative-rate-comparison/attained-state-comparison.png)

![Exact quarter-rate off/on skin sections](../data/52-conservative-rate-comparison/sections/conservative-off-vs-on.png)

## Saved endpoint geometry

The views below use the unchanged skin mesh, frozen cameras, flat shading, and exact saved displacements without exaggeration. Endpoints have different fit, motion, and budgets; these images are descriptive and do not substitute for the matched-state table.

![Original and quarter-rate off endpoints](../data/52-conservative-rate-comparison/geometry/unmatched-off-endpoints.png)

![Original and quarter-rate on endpoints](../data/52-conservative-rate-comparison/geometry/unmatched-on-endpoints.png)

## Evidence and command

[Comparison receipt](../data/52-conservative-rate-comparison/summary.json), [selected-state CSV](../data/52-conservative-rate-comparison/selected-states.csv), [verification receipt](../data/53-conservative-rate-verification/summary.json). Independent verification passes evidence integrity for 130 evaluated states and 12 full checkpoints, including initialization, source hashes, Adam state, and the first-step equivalence gates. Physical inversions remain explicitly recorded.

The comparison completed at 11:46:59, [Comet 8c321150225240f8b96d1a656716b2e9](https://www.comet.com/liblaf/apple/8c321150225240f8b96d1a656716b2e9). Its summary SHA-256 is `649e73f395b1c6946da6c6e4aeb06430c6d3ed53721e6a7cb7c26806a9e697cf`. The figures were visually inspected and their source/output hashes checked.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='Conservative learning-rate saved-state comparison' \
CHERRIES_TAGS='activation-space-smoothness,learning-rate,saved-state,cpu-analysis' \
.venv/bin/python3 src/52-compare-conservative-rate.py \
  --off-dir data/48-axis-off-lr-quarter-64 \
  --on-dir data/49-axis-on-lr-quarter-64 \
  --output-dir data/52-conservative-rate-comparison
```
