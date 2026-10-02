# Selected results: learned axis, spatial smoothness, active stress

Use three comparisons, each answering a distinct question. Keep the learning-rate sweep as supporting material rather than mixing it into this presentation. These selections use existing saved states; no new fit or rendering was performed.

## 1. Learned axis: achieved fit and remaining local distortion

Select **corrected Raw6 + smoothness at update 200** and **learned axis + smoothness at update 128**. Both have spatial regularization, so the labels make the model distinction clear.

| Saved result | Fit RMS (mm) | Motion RMS (mm) | Surface residual HP (mm) | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| Raw6 + smoothness, 200 | 1.404671 | 4.807193 | 0.179513 | 4 |
| Learned axis + smoothness, 128 | 0.681094 | 5.179578 | 0.166520 | 94 |

Show the target, each deformed face, and the same mouth-corner and muscle-section views. The message is: **the selected learned-axis solution fits the target more closely, while retaining substantially more inverted cells. One contraction axis per tetrahedron does not guarantee acceptable geometry.**

This is an achieved-result comparison. Different starts, parameter scaling, rates, smoothness coefficients, and update budgets prevent attributing the numerical difference solely to the learned-axis restriction. The learned direction is per tetrahedron, not one fixed anatomical fiber per muscle.

Useful existing assets: [Raw6-on mouth](../data/40-comparison/geometry/raw6-matched/raw6-on/region1-mouth-corner.png), [learned-axis-on mouth](../data/42-axis-on-endpoint-v2/geometry/region1-mouth-corner/target-axis-on-0128.png), and [learned-axis endpoint sections](../data/44-report-figures-v2/sections/axis-on-endpoint-0128.png). These are source assets, not a finished common-camera cross-model plate.

As supporting evidence, use **learned axis without smoothness at its best-fit update 15**: fit 3.539459 mm, motion 4.121976 mm, and 541 inverted tetrahedra. The full controls and deformation exist in `data/24-learned-axis/best.npz`, verified to contain step 15. Prefer this best-fit state over the badly deteriorated update 28 when showing the representative unsmoothed result. Later failure at attempted update 29 belongs in the trajectory caption.

## 2. Spatial smoothness: smoother controls with little surface change

Select **corrected Raw6 off/on at update 200**. This is the clearest existing equal-start, equal-budget pair, and it meets the frozen 0.05 mm tolerances in both fit and motion.

| Quantity | Smoothness off | Smoothness on |
| --- | ---: | ---: |
| Fit RMS (mm) | 1.397767 | 1.404671 |
| Motion RMS (mm) | 4.818188 | 4.807193 |
| S(C) | 9.402596 | 8.369652 |
| S(Z) | 189.799006 | 163.243365 |
| Surface residual HP (mm) | 0.179875 | 0.179513 |
| Inverted tetrahedra | 4 | 4 |

Show the [paired mouth geometry](../data/40-comparison/geometry/raw6-matched/region1-mouth-corner-target-off-on.png), the two [off](../data/40-comparison/highpass/raw6-matched/raw6-off-residual.png)/[on](../data/40-comparison/highpass/raw6-matched/raw6-on-residual.png) residual maps, and the [activation variation plot](../data/44-report-figures-v2/new-pairs-activation-variation.png). The geometry pair was visually inspected and is nearly unchanged.

The message is: **S(C) falls by 11.0%, but the surface residual improves by only 0.20%.** This shows a difference between smooth activation and smooth reconstructed geometry.

The learned-axis off/on match at **11/11** is supporting evidence: S(C) falls by 9.17%, while surface residual improves by 0.40%. Its [paired mouth view](../data/40-comparison/geometry/learned-axis-matched/region1-mouth-corner-target-off-on.png) is available. Those exact step-11 files are surface snapshots, so do not claim full activation glyphs for them without a corresponding full control checkpoint.

## 3. Active stress: compare against the corrected physical-volume baseline

Select the **rest-start corrected Raw6 baseline at update 200** and **unsmoothed PSD active stress at global update 1024**. This Raw6 run is the earlier physical-volume correction experiment, not the later 200-update Raw6 re-fit used above.

| Quantity | Corrected Raw6, 200 | PSD active stress, 1024 |
| --- | ---: | ---: |
| Area-weighted fit RMS (mm) | 1.835697 | 1.610246 |
| Area-weighted motion RMS (mm) | 4.150593 | 4.125968 |
| Active-muscle volume-weighted RMS of J − 1 | 0.015434 | 0.022861 |
| Inverted tetrahedra in the whole volume | 1 | 0 |

The one corrected-Raw6 inversion is a mixed cell containing 99.9023% fat; both have zero inverted pure-muscle cells. The motion differs by approximately 0.6%, making this the most useful existing visual pair for active stress. The first two comparison groups use uniform fitted-face vector RMS; this table uses the canonical comparison's area-weighted metrics. Do not combine the two metric conventions into one ranking.

Use the corrected-baseline and PSD columns from the canonical mouth view (historical worktree file: `exp/2026/09/08/physical-volume-baseline/data/30-comparison/skin-mouth.png`) and muscle section (historical worktree file: `exp/2026/09/08/physical-volume-baseline/data/30-comparison/muscle-mouth.png`). The mouth plate was visually inspected. In its current three-column form, the left column is the historical activation-dependent-volume baseline; compare the middle and right columns for the selected pair. Later layouts should use separate assets for these two states.

The message is: **both corrected Raw6 and PSD produce relatively regular geometry without a spatial smoothness term; the dramatic historical difference largely disappears after correcting the physical-volume formulation.** The selected PSD state has zero inversions, but its additional benefit is not isolated at matched fit and optimizer history. Avoid using the old baseline's extreme distortion as proof that active stress alone caused the improvement.

## Evidence and display choices

Exact checkpoint paths, stored steps, available arrays, sizes, and SHA-256 hashes are recorded in [selection.json](../data/80-idea-result-selection/selection.json). The states were checked directly. Existing numerical evidence comes from [current comparison results](40-results.md), [selected-state metrics](../data/40-comparison/selected-states.csv), and the [physical-volume correction report](../../../07/tensor-active-stress/docs/114-physical-volume-correction.md).

Use the same target, camera, deformed coordinates, scalar range, and section locations within each newly prepared comparison. Keep figure assets separate for later layout. For cross-model activation magnitudes and variation, use the common effective field Z or stress Q=μZ. A single learned-axis line describes the rank-one model; it does not fully describe a general-rank Raw6 or PSD tensor. Surface residual HP here is the frozen local high-frequency target-residual score, not whole-face roughness.
