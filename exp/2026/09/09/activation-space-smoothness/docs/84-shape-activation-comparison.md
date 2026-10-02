# Shape and activation comparisons

The comparison presents three pairs selected from existing saved results. The interactive gallery (private preview omitted) places the deformed shape and activation field beside one another, with controls for the comparison, camera, and displayed quantity. Each figure is also a separate 1,800 × 1,800 PNG for later slide layout.

No inverse fitting or mechanical solve is needed to produce these figures. The exact checkpoint paths, stored steps, and hashes are recorded in [selection.json](../data/80-idea-result-selection/selection.json).

## Selected pairs

| Idea | Left state | Right state | Interpretation |
| --- | --- | --- | --- |
| Learned axis | Raw6 + smoothness, update 200 | Learned axis + smoothness, update 128 | Compare the achieved fit, local distortion, and the change from a general tensor to one contractile mode. Starts, rates, weights, and budgets differ. |
| Spatial smoothness | Corrected Raw6, smoothness off, update 200 | Corrected Raw6, smoothness on, update 200 | Same start and budget, with fit and motion inside the frozen matching tolerances. |
| Active stress | Corrected Raw6, rest-start baseline, update 200 | PSD active stress, update 1024 | Both unsmoothed and nearly equal in motion. This Raw6 run is different from the re-fit used in the first two pairs. |

The best unsmoothed learned-axis state, update 15, is included as supporting material. The exact update-11 matched learned-axis files contain surface displacements only, so they are not used for a paired full-field visualization.

## Shape display

Every saved shape uses its actual nodal coordinates `x = X + u` at deformation scale 1. The skin uses the fixture's verified `GlobalPointId` mapping. Target panels use the supplied target displacement in the same camera. Shapes use flat shading without smoothing, decimation, or interpolation between optimizer updates.

The overview and mouth-corner cameras are frozen in the [physical-volume close-up camera receipt](../../../08/physical-volume-closeups/data/20-regions/summary.json). The selected pair uses identical camera position, focal point, view-up vector, orthographic scale, resolution, and illumination. Saved inverted tetrahedra remain present; the figures report their counts rather than repairing geometry.

## A common activation field

Use the dimensionless effective tensor `Z` for every model:

- Raw6: `Z = B Bᵀ − I`, with `B = Ainv = I + sym(q)` in the historical Raw6 coordinate convention.
- Learned axis: `Z = (2 + ‖v‖²) vvᵀ`.
- PSD active stress: `Z = Q / μ`, where `μ = 0.010067114093959731 MPa`, verified from `E = 0.03 MPa`, `ν = 0.49`, and the saved stress convention.

The historical PSD six-coordinate vector is scaled by `QREF = 3 μ`; it is not directly interchangeable with the Raw6 vector. The renderer uses the saved physical `Q` and verifies its reconstruction. All checkpoints carrying `rest_points` match the fixture exactly. The PSD checkpoint does not carry rest points, so its documented fixture supplies `X`.

## One line per tetrahedron

Each full field contains one centered line for every active tetrahedron (`MuscleFraction > 0`), with no spatial sampling. For the common tensor, let `λ₁` be its largest eigenvalue and `n` the corresponding unsigned rest eigenvector. The displayed positive mode has amplitude

`a = 1 − 1 / sqrt(1 + max(λ₁, 0))`.

Its center is the mean of its four saved deformed vertices. With `F = Ds Dm⁻¹`, its spatial direction is `normalize(F n)`. Endpoints are `center ± 0.00225 a normalize(F n)`, in metres. Thus **line length is 4.5 mm × a** for every cell and state. Color uses the same linear 0–100% scale. There is no minimum line length, cell-size normalization, or normalization per state. Transport changes the direction; it does not multiply display length by tissue stretch.

For learned axis this is exactly the existing commanded shortening `‖v‖² / (1 + ‖v‖²)`. For Raw6 and PSD it is a display-equivalent contraction of the strongest positive effective mode. A line does not describe negative or secondary positive modes. A cell with no positive mode has zero display length. The unsigned line does not identify an anatomical fiber or measured tissue strain.

Visibility is recomputed from each saved deformed volume. At each projected centroid, the renderer retains a tetrahedron when the frontmost activation-region label matches its own. This omits other muscles behind the visible muscle while retaining interior tetrahedra throughout that muscle's depth. The full exported line field remains unfiltered. No muscle outlines, silhouettes, or muscle surfaces are drawn in the glyph panels; faint deformed skin supplies context. The centroid rule can leave endpoints crossing a region boundary and permits overlap inside the same visible region.

## What the line omits

The companion map shows the fraction of squared tensor magnitude outside the displayed positive mode:

`r = 1 − max(λ₁, 0)² / ‖Z‖F²`.

For an exactly zero tensor, define `r = 0`. The map uses one fixed 0–100% range on actual deformed muscle surface cells, without edge or silhouette overlays. Negative and secondary positive eigenmodes both contribute. This is a fraction of squared tensor magnitude, not mechanical energy.

The [independent audit](../data/82-idea-activation-audit-v2/summary.json) gives these averages over the full active field, weighted by reference tetrahedron volume times muscle fraction:

| Saved state | Mean omitted squared-magnitude fraction | Cells with a material negative mode |
| --- | ---: | ---: |
| Raw6 re-fit, off, 200 | 43.7819% | 99.9781% |
| Raw6 re-fit, on, 200 | 43.7745% | 99.9899% |
| Learned axis, on, 128 | 0% | 0% |
| Learned axis, off, best 15 | 0% | 0% |
| Corrected Raw6 rest-start, 200 | 44.7375% | 99.9781% |
| PSD active stress, 1024 | 7.1992% | 0% |

The first column is a weighted mean of per-cell fractions, not a ratio of globally summed tensor norms. A material negative mode satisfies `λ < −64 ε max(‖Z‖₂, 1)` for that tensor. The receipt retains exact minimum eigenvalues and tolerance ranges. This distinguishes floating-point eigensolver noise in the algebraically PSD learned-axis fields from Raw6's substantial negative modes. The exported tensor values are not clipped.

## How to interpret the comparisons

The learned-axis endpoint has a smaller fit residual than the selected smoothed Raw6 endpoint (0.681 versus 1.405 mm), but more inverted cells (94 versus 4). Its rank-one activation alone does not ensure acceptable deformed geometry. The model comparison is descriptive because optimization histories differ.

In the Raw6 off/on pair, the activation prior reduces `S(C)` by 11.0% and `S(Z)` by 14.0%, while the surface residual score decreases only 0.20%. The mean omitted-mode fraction barely changes. Spatial regularization reduces variation without removing the extra signed tensor modes available to Raw6.

The corrected Raw6 and PSD pair has area-weighted motion 4.151 and 4.126 mm, with one and zero whole-volume inversions. Both have zero inverted pure-muscle cells; the Raw6 inversion is in a mixed cell containing 99.9023% fat. The principal field and omitted-mode map expose a difference that the relatively regular exterior shapes may conceal. The old activation-dependent-volume baseline is excluded because it would exaggerate the model difference.

The first two pairs use uniform fitted-face vector RMS. The active-stress pair uses the canonical comparison's area-weighted RMS; do not combine these conventions into a ranking. Surface residual HP is the frozen local high-frequency target-residual score, not whole-face roughness. All results come from one fixture and target with the recorded optimization settings.

## Execution and verification

Normal Cherries/Comet recording used `ProfileCometNoCommit`, with no Git commit or push. All rendering and verification processes completed successfully. The shape run produced 14 PNGs and seven exact skin VTPs; the activation run produced 24 PNGs and six complete line fields. The independent checker passed all six states, all 12 visibility masks, and all 38 image dimensions, including source/input/output hashes, tensor reconstruction, affine transport, line lengths, and exact skin coordinates.

| Stage | Comet run | Recorded completion (Asia/Shanghai) |
| --- | --- | --- |
| Shape rendering | [Selected idea shape comparison](https://www.comet.com/liblaf/apple/be937d4dcb64476db80650f1f54f37df) | 2026-09-09 14:33:48 |
| Tensor audit | [Idea activation tolerance audit v2](https://www.comet.com/liblaf/apple/a426ac889e134fd6ab6293cdd446949d) | 2026-09-09 14:28:58 |
| Activation rendering | [83-idea-activation](https://www.comet.com/liblaf/apple/1d0ddfe2f5024369b0eec553eb04abc6) | 2026-09-09 14:42:14 |
| Independent artifact checks | [Idea comparison artifact verification](https://www.comet.com/liblaf/apple/5633fb50b32646a78abdbf9cc874b85d) | 2026-09-09 14:44:04 |

Run the numbered source scripts from this experiment directory, preserving `CHERRIES_NAME` and `CHERRIES_TAGS`. For example, the shape execution was:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='Selected idea shape comparison' CHERRIES_TAGS='shape,comparison,selected-results,report' \
uv run python src/81-render-idea-shapes.py --output-dir data/81-idea-shapes
```

The corresponding activation command was `uv run python src/83-render-idea-activation.py`; the independent check was `.venv/bin/python src/85-verify-idea-comparison.py` using the repository interpreter. Sources, input identities, and output hashes are recorded in [the shape receipt](../data/81-idea-shapes/summary.json), [activation receipt](../data/83-idea-activation/summary.json), and [independent verification](../data/85-idea-verification/summary.json). Use fresh output directories to reproduce rather than overwriting these retained outputs.
