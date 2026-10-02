# From active stress to the physical-volume correction

**All experiments in this report have no skin energy.** The visible face surface is geometry used for fitting and display. The baseline and active-stress comparisons have no inverse regularization; spatial regularization is introduced only in the final section.

![Target shape in the same oblique view](../data/101-oblique-target/target.png)

## 1. Active stress first produced a much smoother result

The historical six-parameter activation baseline fitted the target closely but produced pronounced surface bumps and distorted muscle elements. PSD active stress produced a visibly smoother face without adding a spatial regularizer. This was the initial observation that motivated the analysis below.

The saved endpoints are the old baseline's best update **194** and active stress at **1024**. Their different fit errors and optimization histories mean this initial comparison alone cannot explain the cause of the improvement.

![Old baseline and active stress: surface comparison](../data/97-baseline-oblique/old-vs-psd.png)

## 2. Testing the expected equivalence exposed a volume-energy bug

Write **B = A⁻¹** for the activation multiplier used by the code, so the activated gradient is **G = FB**. Let **W₀(F)** be the passive energy and **Q = μ(BBᵀ − I)**. With physical-volume terms evaluated at **J = det F**, the corrected activation energy satisfies

$$
W_{\mathrm{activation}}(F,B)
=W_0(F)+\frac12 Q:(F^\mathsf{T}F)
=W_{\mathrm{stress}}(F,Q)+\frac12\operatorname{tr}Q.
$$

The last term is independent of deformation: mapped controls therefore give the same equilibrium forces and deformation Hessian. The historical implementation instead used **det(FB)** in its determinant terms. Because `det(FB) = det(F) det(B)`, activation could compensate the penalized determinant while the actual tissue volume changed substantially.

The correction keeps the activated norm term and uses physical volume in both determinant terms:

$$
W=\frac{\mu}{2}(\|FB\|_F^2-3)
-\mu(J-1)+\frac{\lambda_0}{2}(J-1)^2,
\qquad J=\det F.
$$

This is forward equivalence under a control mapping. The tested spaces are not identical: unrestricted Raw6 can produce negative effective modes, PSD active stress excludes them, and learned axis is rank one. Their optimization trajectories also differ.

## 3. The corrected baseline also became much smoother

We reran the original six-parameter activation model with the corrected volume energy, zero initialization, no inverse regularization, and the same 200-update budget. It already removed much of the historical bumpiness. Its surface and muscle deformation are close to the active-stress result.

![Old baseline, corrected baseline, and PSD active stress: oblique face overview](../data/97-baseline-oblique/comparison.png)

![Location of the 694-cell muscle neighborhood on the rest face](../data/107-muscle-location/overview.png)

The box locates the fixed neighborhood shown in the close-ups below.

![Rest, old baseline, corrected baseline, and PSD: the same focused muscle neighborhood](../data/93-focused-muscle-patch/cell-patch.png)

This is the fixed neighborhood from the earlier tetrahedron analysis: **694 muscle-containing cells within 6 mm** of cell 27306's rest centroid. The same cell IDs, world positions, camera, and scale are used across all four panels. The gold tetrahedron identifies cell 27306.

| Measurement | Old baseline | Corrected baseline | Active stress |
| --- | ---: | ---: | ---: |
| Saved update | 194 | 200 | 1024 |
| Area-weighted fit RMS, mm | 0.654 | 1.836 | 1.610 |
| Area-weighted motion RMS, mm | 4.974 | 4.151 | 4.126 |
| Muscle-volume-weighted RMS(det F − 1) | 0.432 | 0.015 | 0.023 |
| Inverted pure-muscle tetrahedra | 33 | 0 | 0 |

In the old baseline, the muscle-weighted RMS deviation of **det(FB)** from one was **0.145**, while that of physical **det(F)** was **0.432**. The compensated determinant was substantially closer to one, but it did not preserve actual volume.

![Tracked pure-muscle tetrahedron 27306, with a common scale](../data/89-baseline-story/cell27306-shapes.png)

For the old tracked cell, **det(FB) = 1.073**, while **det(F) = 2.662**. For the same cell across the three results, physical volume ratios are **2.662 / 1.009 / 0.982**, and largest principal stretches are **4.346 / 1.591 / 1.564**. The corrected baseline and active stress substantially reduce the historical distortion. These are actual saved deformations with fixed topology and cameras, without smoothing or amplification.

The determinant correction is therefore a major confound in the original active-stress comparison. The corrected baseline still has one inverted mixed cell containing 99.9023% fat, and some surface irregularities remain. Neither result is a matched-fit or fully converged superiority claim.

## 4. Uniaxial contraction and activation smoothness added little improvement

We then restricted activation to a learned contraction axis, `B = I + vvᵀ`, and tested a same-muscle spatial smoothness penalty on `C = B − I`. Skin and activation figures use overview cameras; full-size images can be opened for zooming. Shape and activation views use the same cameras and saved deformed coordinates. Each activation line shows the strongest contractile mode; its length increases with contraction on one common scale.

The selected learned-axis result fits more closely than smoothed Raw6 (**0.681 versus 1.405 mm**), but has **94 versus 4** inverted tetrahedra and more localized corrugation. It does not establish a further improvement in geometric quality. This is an achieved-result comparison with different optimization histories.

For the controlled Raw6 off/on comparison at update 200, the shape and dominant activation field look nearly unchanged. The penalty lowers **S(C) by 11.0%**, yet the local surface residual improves only **0.20%**. The matched learned-axis off/on snapshots similarly improve that residual by only **0.40%**. Under the tested settings, smoother controls did not yield a substantial additional surface improvement.

![Smoothed Raw6 activation overview on the saved deformed face](../data/83-idea-activation/geometry/raw6-refit-on-200/side-context.png)

![Smoothed learned-axis activation overview on the saved deformed face](../data/83-idea-activation/geometry/axis-on-128/side-context.png)

The interactive shape and activation gallery (private preview omitted) provides the full pairs, overview and mouth cameras, standalone figures, and a companion map of tensor content omitted by a single line. A line represents the full learned-axis tensor but only the dominant positive mode of Raw6 or PSD.

## Evidence

The canonical correction report (historical worktree file: `exp/2026/09/08/physical-volume-baseline/docs/20-baseline-report.md`), [revised interpretation](../../../07/tensor-active-stress/docs/114-physical-volume-correction.md), and [implementation comparison](46-implementation-comparison.md) retain the model and execution details. The [shape/activation methods](84-shape-activation-comparison.md) and [independent verification](../data/85-idea-verification/summary.json) cover all six later fields and 38 standalone images. The original learning-rate study remains supplementary evidence rather than the main report narrative.
