# Four-stage smile figures

**Current revision: `52-four-stage-figures-010`.** The L2 + normal chain has completed all four 200-update stages. Its final stage has fit RMS 1.463404 mm and normal RMS 3.937855 degrees. The L2 chain was still running when captured at stage 4, update 149; that panel is labeled as a snapshot. Both figures use the same logarithmic normalization for activation color and line length, as requested. The power-6 length rule has been removed. The date labels are omitted from both images; capture timestamps remain available in the receipts.

## Outputs

- [Completed L2 + normal figure](../data/52-four-stage-figures-010/l2-normal-four-stages-16x9.png) and [preview](../data/52-four-stage-figures-010/l2-normal-four-stages-preview.png).
- [Updated L2 figure](../data/52-four-stage-figures-010/l2-four-stages-16x9.png) and [preview](../data/52-four-stage-figures-010/l2-four-stages-preview.png).
- [Render receipt](../data/52-four-stage-figures-010/summary.json) and [independent coordinate and display audit](../data/52-four-stage-figures-010/coordinate-audit.json).
- [Captured states](../data/51-visualization-checkpoints-002/), [L2 manifest](../data/51-visualization-checkpoints-002/l2/manifest.json), and [L2 + normal manifest](../data/51-visualization-checkpoints-002/l2-normal/manifest.json).
- [Renderer](../src/52-render-four-stage-figures.py), [shape helper](../src/shape_scene.py), [activation helper](../src/activation_scene.py), and [capture tool](../src/51-capture-visualization.py).

Both final PNGs are 10240 × 5760 (16:9), with 1920 × 1080 previews. Shape panels are freshly rendered at 2460 × 2000; activation panels at 4428 × 3600 before Lanczos reduction to the same layout size. Text and legends are drawn directly at final resolution. Four stages appear as columns, with full deformed exterior and static anatomy above and activation below.

## Displayed checkpoints

The capture tool double-reads remote endpoints and requires checkpoint, summary and final trace row to agree. All eight hashes and steps were verified. The previous capture `001` was preserved. Source-freeze manifests agree across both remote chains; all material settings match, and protocol differences are the intended loss settings and runtime identifiers.

Fit is reference-surface area-weighted position RMS in millimeters, independently recomputed from the captured displacement. Normal is reference-triangle-area-weighted normal-angle RMS in degrees from the matching trace row.

| Loss | Stage | Update | Fit RMS (mm) | Normal RMS (degrees) |
| --- | --- | ---: | ---: | ---: |
| L2 | Unrestricted, 6 DoF | 200 | 1.493860 | 13.106150 |
| L2 | Contraction only, 6 DoF | 200 | 1.309650 | 10.660379 |
| L2 | Fixed axis, 1 DoF | 200 | 0.828937 | 10.361209 |
| L2 | Released axis, 3 DoF | 149 | 1.222826 | 11.442495 |
| L2 + normal | Unrestricted, 6 DoF | 200 | 2.077406 | 4.244041 |
| L2 + normal | Contraction only, 6 DoF | 200 | 1.097580 | 3.318942 |
| L2 + normal | Fixed axis, 1 DoF | 200 | 0.966839 | 2.296162 |
| L2 + normal | Released axis, 3 DoF | 200 | 1.463404 | 3.937855 |

The sequence is unrestricted symmetric strain → positive-semidefinite contraction → fixed principal axis → released single axis, using projected warm starts. Both branches use Adam learning rate 0.05, epsilon 1e-8, fresh moments per stage, and smoothness coefficient 7.2e-7. The normal coefficient is approximately 1, calibrated so a 2 mm vector RMS error matches a 5-degree normal error. There is no skin membrane energy; activation does not enter the physical volume terms.

## Logarithmic activation color and line length

The saved field is `B = I + S`. The renderer takes the largest algebraic eigenpair `(z, n)` of `Z = B Bᵀ - I`. Each reference axis is transported by the physical deformation gradient and normalized, `F n / ||F n||`, at its deformed tetrahedron centroid. Only unique, positive principal modes in the visible muscle region are drawn.

The colorbar now reports **dimensionless principal amplitude**, replacing the earlier bounded percentage:

```text
a = sqrt(1 + z) - 1 = sigma_max(B) - 1
color_fraction = log(1 + a) / log(1 + 60)
colorbar_ticks = [0, 1, 3, 10, 30, 60]
```

All eight panels use the same range 0–60 and neutral-to-red palette. `log1p` includes zero exactly, without an arbitrary epsilon. The largest plotted amplitude is 54.668344, so no values are clipped. Tick positions follow the logarithmic mapping and their labels are raw amplitudes. For example, `a=1` is the former 50% display value, and `a=3` is the former 75%. The old percentage can be recovered as `p=100*a/(1+a)` for these positive modes.

For positive-semidefinite contraction stages, `a` is the largest eigenvalue of `S`. For unrestricted tensors, it is an effective singular amplitude because `B Bᵀ` loses eigenvalue signs. It is not measured physical tissue shortening. The full tensor drives the saved shape; the line field only shows one principal mode.

Color and line length now use the same normalized logarithm:

```text
fraction = log(1 + a) / log(61)
color = neutral_to_red(fraction)
line_length = 4.5 mm * fraction
```

The mapping is common to all stages and both branches. An amplitude of 0 gives zero length, 1 gives 0.759 mm, 3 gives 1.518 mm, 10 gives 2.625 mm, 30 gives 3.759 mm, and 60 gives 4.5 mm. There is no additional exponent, spatial thinning or per-panel normalization. The actual line-length distribution and raw amplitude quantiles are recorded separately for each stage. The figure footer shows the exact length formula.

## Geometry and camera

The reference volume has 228,660 points and 1,146,517 tetrahedra. Its full exterior contains 64,042 points and 128,172 triangles, including all 15,299 fitting-surface vertices. Every displayed point is exactly `X + u`, transferred through original volume point IDs. There is no displacement exaggeration, smoothing, remeshing or cropping.

Cranium and mandible are the registered melon `13-cranium.ply` and `13-mandible.ply` assets. Eyes are the registered `20-eye.ply` geometry preserved in `rigid-eyes-001/eyes.vtp`. They retain their reference poses. Anatomy paths/hashes appear in the render receipt; lineage is recorded in [verified-static-context.json](../tmp/viz-geometry/verified-static-context.json).

All 16 panels use one orthographic camera looking from direction `(0.65, 0.03, 1)`, fitted to the union of all eight shapes and anatomy. Both eyes remain visible in this three-quarter view. Activation uses the same deformed fitting-skin surface at 6% opacity as the earlier clean figure, with 2-pixel non-tube lines in a 3600-pixel-high native render. The shape row retains full exterior, cranium, mandible and eyes.

The remote mesh archives differ in unused `active_volume_weights`. All arrays consumed by rendering and the recomputed fit RMS agree across hosts: see [shared-mesh-identity.json](../data/51-visualization-checkpoints-002/shared-mesh-identity.json).

## Comparison with previous materials

The older comparison used muscle Young's modulus 30 kPa, fat 3 kPa and aponeurosis 100 kPa. Both new branches use 12, 11.2 and 1,693 kPa, respectively. Muscle/fat Poisson's ratios remain 0.49; aponeurosis changes from 0.35 to 0.49. The old volume coefficient was classical lambda; the new one is classical lambda plus mu. Both plotted lineages use physical `det(F)` in volume terms and have no skin membrane energy. Consequently, comparing inferred activations across the old and new studies does not isolate an optimization or regularization effect.

The earlier principal-field source is `exp/2026/09/14/dominant-activation-ablation/src/96-render-aligned-activation-comparison.py`, verified against its frozen copy. Before visual rescaling, old and new glyph functions produced exactly equal endpoint coordinates, original lengths and original percentages for the same 288,235-cell checkpoint. Larger original lines reflected larger saved fields.

## Numerical status and revision history

The captures preserve finite approximate states under the user's continuation policy. All eight states record `solver_valid=false`; all-volume inverted-cell counts are 36, 79, 120, 109 for L2 and 46, 91, 91, 103 for L2 + normal. These diagnostics are retained in the receipt. Completion means the normal chain finished its configured update budget. The L2 stage-4 snapshot has fewer updates, so the two final-stage panels do not have matched budgets. Capture and rendering did not alter any numerical process.

Revisions 001/002 are superseded due to a shallow-VTK-copy bug that added the preceding displacement to later shape panels. Revision 003 introduced immutable reference points, deep per-state volume copies and exact before/after `X+u` assertions. Revision 004 restored fine glyph styling; 005 removed negative glyphs; 006 added native 10k resolution; 007 applied shared power-6 color and length. Revision 008 refreshed the completed normal chain, advanced the L2 snapshot and replaced color with `log1p(a)` while retaining the power-6 length mapping. Revision 009 applied the same normalized `log1p(a)` to line length and color. Revision 010 removes the capture date/time labels from both image footers. Earlier outputs remain available for provenance.

## Reproduce

Run from `exp/2026/09/21/stress-activation-loss` using a fresh output directory:

```bash
CHERRIES_NAME='Four stage smile figures, date labels removed' \
CHERRIES_TAGS='active-strain,visualization,four-stage,smile,slide,native-10k,log1p,log-length,no-date' \
LIBGL_ALWAYS_SOFTWARE=1 CUDA_VISIBLE_DEVICES='' \
uv run python src/52-render-four-stage-figures.py \
  --source 51-visualization-checkpoints-002 \
  --output 52-four-stage-figures-011
```

The renderer verifies checkpoint hashes, geometry correspondence and the shared display range. A later capture needs a fresh capture directory; credentials remain in private secret files. This rendering completed in about 38 seconds with software OpenGL, including normal Cherries shutdown. No commit was made; the working tree was already modified.

[Comet run](https://www.comet.com/liblaf/apple/7762611728fb4625adc4e5fb28ca7cb7) and [Cherries log](../logs/52-render-four-stage-figures.log).

```text
Comet.ml Experiment Summary
  name: Four stage smile figures, date labels removed
  url: https://www.comet.com/liblaf/apple/7762611728fb4625adc4e5fb28ca7cb7
  cherries/entrypoint: exp/2026/09/21/stress-activation-loss/src/52-render-four-stage-figures.py
  cherries/exp_dir: exp/2026/09/21/stress-activation-loss
  cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
  source: 51-visualization-checkpoints-002
  output: 52-four-stage-figures-010
```

## Verification

- Both final outputs are exactly 10240 × 5760, with native geometry/glyph renders and 1920 × 1080 previews.
- All eight checkpoint and renderer/helper hashes match the receipt. L2 + normal has four completed step-200 endpoints.
- An independent audit reloaded all eight rendered-surface arrays and compared each with the corresponding capture's `X+u`; maximum coordinate error is exactly zero.
- All 16 screen-projection matrices agree within an absolute tolerance of 1e-12.
- Independently recomputed amplitude, logarithmic colors and logarithmic line-length quantiles match the receipts. Zero maps exactly to zero; all values lie within 0–60 without clipping.
- Ruff checks passed. Previously passed shared-VTK-points regression and activation-helper tests cover the unchanged geometry/activation helpers.
- Both previews were independently inspected for readable headings, completion/snapshot labels, colorbar tick positions and shared logarithmic color/length formulas. No clipping or overlap was found.
