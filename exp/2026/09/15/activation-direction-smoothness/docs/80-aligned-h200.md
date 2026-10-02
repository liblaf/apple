# Aligned final shape and activation at h = 0.20

[Combined overview](../data/80-aligned-h200/aligned-shape-activation-h200.png) · [PDF](../data/80-aligned-h200/aligned-shape-activation-h200.pdf) · [Gallery with larger off/on sheets](../data/80-aligned-h200/index.html)

[16:9 slide-ready PNG (3840 × 2160)](../data/91-slide-h200/aligned-shape-activation-h200-16x9.png) · [16:9 PDF](../data/91-slide-h200/aligned-shape-activation-h200-16x9.pdf)

[16K PNG](../data/92-high-resolution-h200/aligned-shape-activation-h200-16x9.png) · [Vector SVG](../data/92-high-resolution-h200/aligned-shape-activation-h200-16x9.svg) · [Vector PDF](../data/92-high-resolution-h200/aligned-shape-activation-h200-16x9.pdf)

## Layout and interpretation

The four rows are free activation, contraction-only with free directions, contraction-only with a learned direction, and contraction-only with fixed x-direction. The four columns pair final shape and activation for smoothness off, then final shape and activation for smoothness on (α=1). All panels show the whole domain on identical physical axes, x∈[−0.02,1.02] and y∈[−0.01,0.31], with equal x/y scale. The dashed target top surface appears in both views.

Activation uses the same saved deformed triangle centers and F n / ||F n|| directions as the preceding render. Glyphs have fixed full length 0.0168995 model units and width 0.45 pt. Color encodes signed eigenvalues of B−I on the shared range ±4.430356. This is transported material activation, not a spatial stress eigensystem.

The free/off panels show the last valid state at step 262, before a forward solve failure at proposal 263. Every other displayed case is at step 1200. These are saved finite-budget results. The failed run is visibly marked in both its shape and activation panels.

![Aligned overview](../data/80-aligned-h200/aligned-shape-activation-h200.png)

## Verification

All eight cases match the previous glyph geometry arrays exactly. The renderer checks that every deformed mesh and target lies inside the shared bounds, panel pixel dimensions match, paired panels align vertically, and x/y physical scales are equal. Maximum floating-point panel-size spread is below 5e−13 pixels. All three PNG/PDF pairs were opened or decoded, and the figure layouts were visually inspected. See [alignment checks](../data/80-aligned-h200/alignment-checks.json) and [delivery checks](../data/80-aligned-h200/delivery-checks.json).

## Reproduction and run receipt

Run from `exp/2026/09/15/activation-direction-smoothness`:

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CHERRIES_NAME='Aligned final shape and activation h020' \
CHERRIES_TAGS='2d,activation,deformed-glyphs,aligned,figures' \
uv run python src/80-render-aligned.py \
  > logs/render-aligned-h200-terminal.log 2>&1
```

The normal Cherries run exited 0. [Comet run](https://www.comet.com/liblaf/apple/bb0839f9a91d47c5bf96aed7b473e30e) · [Summary](../data/80-aligned-h200/comet-summary.txt) · [Terminal log](../logs/render-aligned-h200-terminal.log). The terminal log contains no Local-plugin failure. The source snapshots include this renderer and the shared drawing functions from `30-render.py`. Numerical fits were read from the existing off/on checkpoints. Git mutations were disabled; unrelated working-tree changes remain present.

## 16:9 slide export

The full overview has been reflowed onto an exact 16 × 9-inch canvas, exported at 240 dpi as **3840 × 2160 pixels**. The matching PDF page is 1152 × 648 points. Full-canvas export preserves this ratio; no tight bounding-box crop is applied. Larger headings, abbreviated column labels, row spacing, and reserved colorbar/footer space make the entire figure suitable for a widescreen slide. All 16 panels keep the same data bounds, equal physical x/y scale, and eight saved solutions.

In a 16:9 PowerPoint slide, insert the PNG at the slide's full width and height with aspect ratio locked. The white margins are already included. [Export checks](../data/91-slide-h200/delivery-checks.json) verify exact dimensions, source snapshots, and pixel identity with the visually inspected preview.

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CHERRIES_NAME='Final 16 by 9 shape and activation h020' \
CHERRIES_TAGS='2d,activation,aligned,16x9,figures' \
uv run python src/80-render-aligned.py --slide-format true --output 91-slide-h200 \
  > logs/render-slide-final-h200-terminal.log 2>&1
```

The final Cherries render exited 0. [Comet run](https://www.comet.com/liblaf/apple/05cbb9514d5e4f5fb84fb0e91b2813e5) · [Summary](../data/91-slide-h200/comet-summary.txt) · [Terminal log](../logs/render-slide-final-h200-terminal.log). Source checks and formatting pass.

## 16K and true-vector export

The full 16:9 composition is rendered directly from the saved mesh and activation arrays at **15360 × 8640 pixels** (960 dpi on the 16 × 9-inch canvas). This is four times the preceding width and height, or 16 times its pixel count. All panels are rendered at the requested output resolution.

The matching SVG and PDF contain vector geometry for the meshes, activation glyphs, curves, text, and colorbar. Every rasterized artist flag is cleared before export. Verification found **zero embedded images** in both formats and 36,197 SVG paths. The one-page PDF is 1152 × 648 points. Geometry, physical scales, color range, and failure markings match the preceding figure. A 600 dpi crop from the vector PDF was rendered for zoom inspection. [Delivery checks](../data/92-high-resolution-h200/delivery-checks.json) include source and prior glyph-archive hashes.

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CHERRIES_NAME='16K and vector shape activation h020' \
CHERRIES_TAGS='2d,activation,aligned,16x9,16k,vector,figures' \
uv run python src/80-render-aligned.py --slide-format true --slide-dpi 960 \
  --vector-export true --output 92-high-resolution-h200 \
  > logs/render-high-resolution-h200-terminal.log 2>&1
```

The normal Cherries run exited 0. [Comet run](https://www.comet.com/liblaf/apple/5567ef68943a44fe900f514d1738b876) · [Summary](../data/92-high-resolution-h200/comet-summary.txt) · [Terminal log](../logs/render-high-resolution-h200-terminal.log).
