# Residual bumps and the missing nasolabial fold

The target contains a clear, continuous nose-to-mouth fold on the same surface topology used to render the corrected baseline. The step-200 result has localized mouth-corner puckering and dispersed cheek/jaw irregularities instead. It has therefore not reproduced this target feature. The renders establish geometric representability on this mesh; they do not establish that the target is reachable as an equilibrium of the current forward model.

All comparisons show **rest geometry, corrected baseline step 200, target geometry**, with identical cameras, lights, triangle connectivity, flat shading, and actual deformation. The cameras approximate the three anatomical regions marked in the supplied screenshot; they are not an exact registration of its screen-space boxes.

| View | Observation |
| --- | --- |
| [Region 1: mouth corner](../data/20-regions/region1-mouth-corner.png) | The result concentrates radial folds around the corner. The target has a more coherent crease continuing upward beside the nose. |
| [Region 2: lateral cheek](../data/20-regions/region2-lateral-cheek.png) | Short irregular ridges appear in the result where rest and target are substantially smoother. |
| [Region 3: lower cheek/jaw](../data/20-regions/region3-lower-cheek.png) | The result adds irregular relief above the jaw boundary. The mesh boundary itself is jagged in all three states and should not be counted as an optimization artifact. |
| [Nasolabial region](../data/20-regions/nasolabial-region.png) | The continuous target groove remains weak and fragmented in the result. |
| [Oblique face context](../data/20-regions/side-context.png) | Shows where these local differences sit within the overall smile. |

![Rest, corrected result, and target at the nasolabial region](../data/20-regions/nasolabial-region.png)

## What is verified

The inverse objective reads the existing `Smile` vectors directly from the volume fixture and includes all 15,302 finite `IsFace` target vertices. This rerun does not downsample or transfer the target. The 15,299-point, 29,899-triangle display surface is a separate extraction of its exterior geometry. The target fold is visibly present on this display surface.

The optimized loss is uniform Cartesian displacement MSE:

$$
\mathcal L(q)=\frac{10^6}{3N}\sum_{i=1}^{N}
\|u_i(q)-u_i^*\|^2,\qquad N=15302.
$$

This loss includes the fold's vertices. It has no explicit normal, curvature, crease, or regional weighting term. If it reached zero, those vertices would match and the target fold would be reproduced; the absence of a curvature term does not make that impossible. At a finite residual, however, displacement MSE does not directly control the sharpness or continuity of a geometric line feature.

An independent CPU recomputation gives global vector RMS **1.763031 mm**. Exploratory target-space boxes give:

| Patch | Vertices | Fit RMS, mm | Share of total squared residual |
| --- | ---: | ---: | ---: |
| Nose to mouth | 630 | 2.536021 | 8.52% |
| Mouth corner | 436 | 1.658858 | 2.52% |
| Lateral cheek | 720 | 1.749542 | 4.63% |
| Lower cheek/jaw | 557 | 2.152271 | 5.42% |

These boxes overlap and are not exact traces of the fold or screenshot rectangles. Their shares must not be added. The nose-to-mouth region contributes appreciably to the current loss; it is inaccurate to say the optimizer ignores it. Its residual is simply still substantial.

All inner forward/adjoint evaluations succeeded, but the outer inverse run stopped at its prescribed 200-update budget. The saved result is explicitly not certified stationary. Consequently, the present evidence cannot separate incomplete optimization from limited mechanical reachability of the target fold.

The [exact horizontal sections](../data/21-diagnostics/nasolabial-horizontal-sections.png) provide a lighting-independent comparison at fixed world-space heights 2.170, 2.180, and 2.190 m. They show different local profiles in the result and target. These are spatial sections, not tracked material curves or a measured fold-depth statistic. [Raw segment endpoints](../data/21-diagnostics/nasolabial-horizontal-segments.csv) are retained.

## Why volume preservation does not remove all bumps

The physical determinant correction prevents activation from directly compensating the volume penalty. It does not prescribe the surface shape. For example,

$$
F=I+\gamma e_1e_2^\mathsf T
\quad\Longrightarrow\quad\det F=1
$$

for any shear amplitude \(\gamma\). The remaining norm term still resists shear, but the determinant terms alone cannot distinguish smooth deformation from uneven local shear.

The current inverse has six independent symmetric controls per active tetrahedron: 1,729,410 scalars. There is no spatial activation penalty or fiber-aligned control restriction in this run. This allows cell-to-cell activation variation without an explicit spatial penalty and is a plausible source of the dispersed relief. A matched control-basis or regularization experiment is needed to establish that causal explanation. Flat shading also displays ordinary triangle facets; the stronger evidence here is the difference between three states rendered on the same topology.

## Mechanical mechanisms still to test

There is skin geometry for weighting and visualization, but no skin energy in the executed baseline. Fat, muscle, and aponeurosis use one conforming tetrahedral mesh and one continuous displacement field, with spatially varying material fractions. There are no independent layer-sliding/separation/contact degrees of freedom. The only explicit kinematic attachment is zero displacement at the inherited fixed vertices; the model does not specify separate muscle origin/insertion or dermal tether laws. Relevant muscle labels are present, including upper-lip elevators and zygomatic muscles.

These details matter because a nasolabial fold is a regionally organized deformation. Cadaver traction experiments associate different fold portions with different upper-lip elevators. [Pessa and Brown, 1992](https://pubmed.ncbi.nlm.nih.gov/1570780/). Dissections describe medial/lateral differences in fat thickness and dermal muscle attachments. [Fu et al., 1999](https://pubmed.ncbi.nlm.nih.gov/11501147/). MRI during smiling shows muscle shortening together with redistribution of cheek fat. [Gosain et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8773684/).

This anatomical evidence motivates testing regional force transfer and skin/tissue mechanics. It does not prove that adding a skin shell, an interface law, or a different activation model is necessary or sufficient in this case.

The next controlled question is whether further progress with the corrected baseline reduces the nose-to-mouth residual and recovers the target section shapes. A local plateau despite verified inverse convergence would motivate a reachability or anatomical-mechanics test. No such continuation or mechanics ablation was run for this report.

## Evidence and reproduction

The baseline runner (historical worktree file: `exp/2026/09/08/physical-volume-baseline/src/20-run-baseline.py:257`) defines the loss and fixed budget. The forward helper (historical worktree file: `exp/2026/09/08/physical-volume-baseline/src/baseline_physics.py:195`) defines the target, shared mesh, material phases, and zero-skin constraint. The completed run summary (historical worktree file: `exp/2026/09/08/physical-volume-baseline/data/20-baseline/summary.json`) records its endpoint and status.

The [render receipt](../data/20-regions/summary.json), [render validation](../data/20-regions/validation.json), and [diagnostic receipt](../data/21-diagnostics/summary.json) record exact inputs, source hashes, cameras, region boxes, metrics, and output hashes. Fifteen render-source/output hashes passed verification; exported rest and target coordinates match the fixture exactly. Five final comparison PNGs are 3,000 × 1,130 pixels. The earlier [individual close-ups](10-closeup-views.md) remain available.

Run from `exp/2026/09/08/physical-volume-closeups`:

```bash
DEBUG=1 CHERRIES_NAME='Marked regions and nasolabial fold comparison' \
CHERRIES_TAGS='physical-volume,regions,nasolabial,target,postprocessing' \
.venv/bin/python src/20-regional-comparison.py

DEBUG=1 CHERRIES_NAME='Physical volume regional diagnostics' \
CHERRIES_TAGS='physical-volume,regional-diagnostics,nasolabial' \
.venv/bin/python src/21-regional-diagnostics.py
```

Use a fresh `--output-dir` to repeat either stage. Both are local postprocessing and completed with exit 0. Cherries subsequently emitted missing-log errors from its local snapshot-copy hook; canonical outputs were verified directly. The regional renders used a local profile after the earlier individual-view run encountered Comet shutdown failures. No forward solve, adjoint, optimization update, source-material change, or geometry smoothing was performed for this diagnosis.
