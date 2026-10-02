# Learning rate 0.3 results

Rate 0.3 is a more conservative choice for the discrete updates in this test, but it does not resolve the inversion problem. It completed 256 updates with 2.876797 mm fit error and 73 inverted tetrahedra. It avoided the sharp late deterioration seen at rate 1.964496 through the tested budgets. Against the original rate, however, its selected global-fit/motion match has a 9.35% higher surface residual, so the result is not a general improvement in surface quality.

The targeted smoothed-arm run tests the user's proposed Adam rate of 0.3 with the original learned-axis initialization and objective. It is a rate-only comparison against the original 7.857986 and quarter-rate 1.964496 smoothed arms. There is no unsmoothed rate-0.3 arm in this follow-up, so it cannot establish the effect of adding smoothness at rate 0.3.

The [frozen protocol](54-rate-03-plan.md) retains the fixture, target, materials, epsilon 0.01, betas (0.9, 0.999), initial strength 0.001, seed 20260909, lambda_C = 0.0004450069704277614, and the complete numerical implementation. Initial q/C/Z arrays are bitwise identical to the original smoothed arm. The [execution record](55-rate-03-execution.md) lists commands, Comet records, and the exact saved-Adam continuation.

## Inversion onset

Rate 0.3 still inverts. Its first inverted state occurs at update 111 with 1.000948 mm motion, compared with update 8 at the original rate and update 25 at the quarter rate. The adjacent non-inverted states matter because the rates sample deformation at different step sizes.

| Smoothed-arm rate | Last non-inverted → first inverted update | Motion RMS interval (mm) | Fit RMS at first inversion (mm) | Maximum commanded axial shortening at first inversion |
| --- | --- | --- | ---: | ---: |
| 7.857986 | 7 → 8 | 0.707708 → 1.166659 | 4.697891 | 97.51% |
| 1.964496 | 24 → 25 | 0.981400 → 1.142112 | 4.714298 | 97.74% |
| 0.3 | 110 → 111 | 0.966093 → 1.000948 | 4.793328 | 97.09% |

These sampled motion intervals overlap. The smaller rate delays inversion in update count and resolves its onset more finely; the results do not show an inversion-free path to larger deformation. They also do not establish an exact universal motion threshold, because the optimized control fields and paths differ.

The shortening column is a control-space diagnostic, not measured tissue strain. For s = ‖v‖², B = I + vvᵀ and A = B⁻¹, so the preferred axial stretch in A is 1/(1+s), and commanded shortening is s/(1+s). The reported maximum is the largest value over active cells. It does not use the observed deformation gradient F. Thus a roughly 1 mm global motion RMS can coexist with an extreme command in a small part of the mesh. The scalar summaries do not identify whether the most strongly commanded cell is the inverted cell and do not establish a causal mechanism.

This interpretation follows the frozen [metric definition](../data/25-learned-axis-smooth/sources/experiment/study_runner.py) and [control mapping](../data/25-learned-axis-smooth/sources/experiment/activation_controls.py). The unchanged optimizer has no outer inversion rejection, magnitude cap, or physical-step limit. Lowering the scalar rate changes the update size, but does not add any of those constraints.

## Endpoints, budgets, and physical steps

The first 128 updates completed normally, then the run continued to 256 from its saved controls, displacement seed, gradient, Adam moments, and counter. The endpoint is also the best-fit state. No extension to 384 was launched: measured runtime no longer supported another 128 updates before the frozen fitting cutoff. There was no administrative interruption or numerical solver failure.

| Smoothed-arm rate | Endpoint update | Fit RMS (mm) | Motion RMS (mm) | Inverted tetrahedra | Minimum det(F) | Surface residual HP (mm) | S(C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7.857986 | 128 | 0.681094 | 5.179578 | 94 | -1.528030 | 0.166520 | 285.582 |
| 1.964496 | 64 | 5.152102 | 7.308119 | 2,189 | -4.186610 | 0.609685 | 38451.179 |
| 0.3 | 256 | 2.876797 | 4.047514 | 73 | -1.026526 | 0.204550 | 462.514 |

These endpoints have different budgets, fit, and motion; they are descriptive and cannot provide a controlled ranking by inversion count or surface residual. The original-rate smoothed run reached a substantially better fit in fewer updates, so reducing the rate also has an efficiency cost. The quarter-rate smoothed run's best fit was 3.378381 mm at update 45 before its later deterioration. Rate 0.3's worst observed minimum det(F) was −1.360181 at update 216.

| Rate | Face update median (mm RMS) | Face update maximum (mm RMS) | Cumulative face path (mm) | Z-update median | Z-update maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7.857986 | 0.079109 | 0.878159 | 20.858548 | 1.759709 | 221.559220 |
| 1.964496 | 0.226140 | 1.355003 | 17.142793 | 6.165770 | 972.814486 |
| 0.3 | 0.030987 | 0.239977 | 7.942627 | 0.409414 | 1.194759 |

These summaries exclude initialization and describe each observed path; Z changes use Frobenius RMS. They do not imply convergence or compare equal iteration budgets. The nominal rate reduction does produce much smaller physical increments for rate 0.3, including at the matched states below. In contrast, the quarter-rate smoothed trajectory eventually develops large updates, so “smaller nominal rate” is not a general bound on realized step size.

![Three-rate trajectories with readable scale ranges](../data/59-rate03-report-figures/optimizer-trajectories.png)

## Matches at global fit and motion

Each comparison selects actual saved states within 0.05 mm in both global fit RMS and motion RMS, excluding initialization. Among eligible pairs, choose the lowest mean fit, then normalized mismatch and earliest steps. No interpolation or tolerance relaxation is used. Positive surface HP change means the right-hand state is worse.

| Comparison, left → right | Updates | Fit gap (mm) | Motion gap (mm) | Inversions | Minimum det(F) | Surface HP change | Low-frequency projection ratio |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| Original → 0.3 | 11 → 158 | 0.040135 | 0.027378 | 18 → 11 | -0.552169 → -0.634472 | +9.35% | 0.796 |
| Quarter → 0.3 | 43 → 184 | 0.048882 | 0.027298 | 205 → 20 | -1.986782 → -0.764653 | -0.66% | 0.686 |
| Original → quarter | 13 → 44 | 0.023933 | 0.009415 | 81 → 214 | -2.351025 → -1.927207 | +1.68% | 1.180 |

At the original-rate match, rate 0.3 reduces the latest face update from 0.878159 to 0.034376 mm RMS and the Z update from 21.147712 to 0.974342. Its inversion count is lower (18 → 11), but minimum det(F) is slightly more negative and surface residual HP rises from 0.245876 to 0.268860 mm (+9.35%).

At the quarter-rate match, the face update decreases from 0.261393 to 0.040463 mm RMS and the Z update from 14.830701 to 0.623530. Inversions decrease from 205 to 20, minimum det(F) is less negative, and residual HP falls from 0.231362 to 0.229828 mm (0.66% improvement).

The low-frequency normal target-projection ratios are 0.796 and 0.686 for the two rate-0.3 matches, both below 0.90. Matching two global RMS measures therefore does not establish the same local expression structure. The small HP reduction against the quarter rate should be read with that change in motion pattern, and the higher HP against the original rate rules out a blanket surface-quality improvement claim. These are rate comparisons within the smoothed arm; there is no rate-0.3 off/on result to assess against the original smoothness-effect criterion.

![Surface and inversion diagnostics at global fit and motion](../data/59-rate03-report-figures/global-fit-motion-comparison.png)

![Original versus rate-0.3 exact matched sections](../data/58-rate03-comparison/sections/original-vs-rate03.png)

![Quarter versus rate-0.3 exact matched sections](../data/58-rate03-comparison/sections/quarter-vs-rate03.png)

## Exact saved endpoint geometry

These views use the unchanged skin mesh, fixed cameras, flat shading, and exact saved displacement without exaggeration. The endpoints are unmatched, as labeled.

![All three smoothed endpoints and target](../data/58-rate03-comparison/geometry/unmatched-smoothed-endpoints.png)

## Interpretation and evidence

A rate of 0.3 is a useful conservative baseline for further diagnostics because it reduces realized update sizes and avoided the quarter-rate run's sharp late deterioration through 256 updates. It is not an inversion remedy or a demonstrated best rate. The original smoothed run still achieves lower error, and the surface comparison depends on which rate it is matched against. Further work should investigate activation magnitude and the acceptance of inverted states explicitly instead of treating another scalar rate reduction as a sufficient fix. No such changes were made in this test.

This is one seeded trajectory per rate, one mesh and target, and finite budgets. It does not establish stationarity, a common continuous inversion threshold, seed robustness, or physiological validity.

The [final verification receipt](../data/57-rate-03-verification/summary.json) passes 257 trace/solver/surface states, 18 full checkpoints, initial equivalence, unchanged sources, and the exact saved-Adam continuation. Physical inversion remains a recorded defect. [Verification Comet record](https://www.comet.com/liblaf/apple/56d9466adb454e0480d33702b51e54ec). Receipt SHA-256: `c94df13cc7fabdf4f6d105e451630dabec5baf4bc8a2aca551769b1ec81d079c`.

The [comparison receipt](../data/58-rate03-comparison/summary.json) and [selected-state CSV](../data/58-rate03-comparison/selected-states.csv) retain all exact values and source hashes. [Comparison Comet record](https://www.comet.com/liblaf/apple/ed737411e00c40f3b7b3df1211cdcf13), completed at 12:24:40. Summary SHA-256: `57e3c6ebfa73606d57895d1e983cea9c434fb5b1ac84e2a32ebe85cfc98c664a`. All 43 comparison-input and 15 output hashes passed; saved endpoint and section figures were visually inspected. The report plots are rendered separately to make their scale ranges readable; they do not change the analysis or source data. The [plot receipt](../data/59-rate03-report-figures/summary.json) records the explicit symmetric-log scales, unchanged input hashes, and output hashes. [Plot Comet record](https://www.comet.com/liblaf/apple/5d22fdf27df94c1aa33fc43de15ff5f3). Its summary SHA-256 is `fcdc9e90ec4ee60d8e0bc8dc021ebd2d076bae2312e320b4aaef909000fe3ddb`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='Smoothed learned-axis three-rate saved-state comparison' \
CHERRIES_TAGS='activation-space-smoothness,learning-rate,rate-03,saved-state,cpu-analysis' \
.venv/bin/python3 src/58-compare-rate-03.py \
  --quarter-dir data/49-axis-on-lr-quarter-64 \
  --rate03-dir data/56-axis-on-lr03-256 \
  --output-dir data/58-rate03-comparison
```
