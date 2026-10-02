# Aligned deformed shapes and point-to-point target errors

[One-page 16:9 PowerPoint](../data/99-aligned-error-slide/aligned-errors-five-way-16x9.pptx) · [10,240 × 5,760 PNG](../data/99-aligned-error-slide/aligned-errors-five-way-16x9-10k.png)

![Slide preview](../data/99-aligned-error-slide/aligned-errors-five-way-16x9-preview.png)

The page retains the five saved states and top-row shape images from the [principal-activation comparison](96-aligned-five-way.md). The lower row shows point-to-point target error on each saved deformed skin. Every panel uses the same orthographic camera and physical scale. All error maps use the common linear viridis range **0–10 mm**, with no clipping. The PowerPoint contains the ten original **1800 × 1800** PNG panels at full source resolution and disables automatic image compression. Each panel spans 1952 × 1952 pixels in the 10K page export.

## Definition and domain

For skin vertex i, `GlobalPointId` identifies its corresponding volume vertex. The displayed distance is

$$
e_i = 1000\,\|(X_i+u_i)-(X_i+\mathrm{Smile}_i)\|_2
    = 1000\,\|u_i-\mathrm{Smile}_i\|_2\quad\mathrm{mm}.
$$

`Smile` stores the target displacement. Both positions use the same reference coordinates and correspondence. No registration or nearest-surface matching is applied. The error maps contain all **15,299 skin vertices**, each with a finite target and `IsFace=true`. The fit RMS printed above each column uses all **15,302 finite IsFace volume vertices**. The three additional objective vertices are GlobalPointIds 24364, 68209 and 71018, so skin RMS and the reported fit RMS differ slightly.

The renderer linearly interpolates the vertex error scalars over each triangle before color mapping. It does not recompute the norm at triangle interiors. Lighting is disabled on the error maps to preserve the scalar colors; the upper row retains the shape shading. Only the visible skin surfaces appear in a given camera view.

## Results

All distances below are in millimeters. Inversion counts refer to the complete volume mesh.

| State | Face fit RMS | Skin RMS | Skin p95 | Skin maximum | Inversions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Free activation | 1.763031 | 1.763202 | 3.895218 | 6.097381 | 1 |
| Dominant only | 3.318048 | 3.318370 | 6.766848 | 9.179898 | 0 |
| Fixed-axis refit | 2.070724 | 2.070924 | 4.369724 | 5.947751 | 0 |
| Released-axis continuation | 1.092945 | 1.093047 | 2.297113 | 4.283849 | 16 |
| Learned-axis from scratch | 0.681094 | 0.681155 | 1.407832 | 3.387863 | 94 |

The largest error is **9.179898 mm** in the dominant-only state, which fits inside the common 0–10 mm range. Dominant-only has a broad visible error region around the cheek and mouth. Fixed-axis refitting reduces that error, and releasing the axes reduces it further. The scratch state has the lowest RMS among these saved endpoints but has 94 inverted tetrahedra and visible surface irregularity.

These are the same corrected physical-volume results as the prior comparison. No forward or inverse mechanics was rerun. Optimization budgets, initialization and regularization differ across the saved states, so the page is a descriptive comparison. The scratch state is the original smoothness-on run at update 128, not a controlled optimizer match to the released continuation. The shape and inversion diagnostics remain relevant when interpreting the smaller errors.

## Assets and reproducibility

- [Error renderer](../src/98-render-aligned-errors.py), [slide builder](../src/99-build-aligned-error-slide.mjs).
- [Numerical summary](../data/98-aligned-errors/summary.json) records source hashes, scalar statistics, correspondence domain and cameras.
- `data/98-aligned-errors` contains ten 1800 × 1800 error panels: whole-face and mouth-corner views for each state. It also contains per-state NPZ error values and deformed VTP surfaces with `PointToPointErrorMm` and `TargetPosition` arrays.
- [Complete process log](../data/98-aligned-errors/run.log) and [Comet run](https://www.comet.com/liblaf/apple/247baf930f7a49e59a08d3dcaf69893f).

The renderer completed with exit code 0. The Comet summary records the human-readable name, 30 error/fit metrics, configuration paths, source and Git metadata at base commit `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. Current experiment sources and outputs are untracked working-tree additions. The executed source snapshots are under `data/98-aligned-errors/sources/`. Cherries Local emitted the existing nonfatal internal log-copy error during shutdown; the terminal log was copied manually, and the standalone local assets are the verified output. An initial attempt failed before rendering because the output directory had not been created; the source was corrected before the completed run.

The computation independently reproduces all previous face RMS values, verifies the skin correspondence and input hashes, checks every error lies within the shared range, and checks exact equality of the x/y/w screen-projection rows with the prior shape panels. An independent package audit confirms all ten embedded PNGs match the source bytes exactly. Package and layout checks passed for the one-slide PowerPoint; its final imported artifact supplied the PNG exports. The preview and a crop of the 10K image were visually inspected. No native PowerPoint execution was used.

Working directory: `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`.

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
LIBGL_ALWAYS_SOFTWARE=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Aligned five-state corresponding-vertex target errors' \
CHERRIES_TAGS='face,activation,point-to-point-error,comparison,physical-volume,visualization' \
.venv/bin/python src/98-render-aligned-errors.py
```

The renderer requires an empty output directory. Preserve completed outputs and pass a different `--output-dir` for any new run. For slide reproduction, copy the ES module to a private build directory with a `node_modules` symlink to the bundled runtime packages, then run it with the bundled Node executable. The source defines the input/output paths, font, finalization checks and 8× PNG export scale.
