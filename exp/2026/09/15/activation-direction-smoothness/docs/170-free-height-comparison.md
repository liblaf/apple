# Free activation with smoothness off: four target heights

Updated after the request to continue through forward failures: [all four heights at 1,200 Adam updates](190-completed-height-comparison.md). The early-stopped results below are retained as the original comparison.

[16:9 PNG, 3840 × 2160](../data/170-free-height-figures/free-activation-height-comparison-16x9.png) · [Vector SVG](../data/170-free-height-figures/free-activation-height-comparison-16x9.svg) · [Vector PDF](../data/170-free-height-figures/free-activation-height-comparison-16x9.pdf) · [Metrics](../data/170-free-height-figures/comparison.json)

![Height comparison](../data/170-free-height-figures/free-activation-height-comparison-16x9-preview.png)

## Purpose and protocol

Extend the experiment in the supplied h = 0.20 slide to h = 0.05, 0.10, 0.15, and 0.20 for free activation with smoothness off. Here h is the peak **target vertical displacement**, with target top surface y = 0.1 + 4 h x (1 − x); h does not vary mesh spacing or specimen thickness.

The original h = 0.05 and 0.20 results come from `data/tune-w0/`. The two missing heights were run independently from B = I and zero displacement using the original runner, mesh, physics, and optimizer. SHA-256 checks verified every computational source listed in the original protocol before running. No continuation results were substituted.

All cases retain the 100 × 10 mesh, 400 active triangles in y ∈ [0.04, 0.06], fixed bottom/side boundaries, E_muscle = 0.03, E_fat = 0.003, and ν = 0.49. The experiment-local Stable Neo-Hookean energy uses physical J = det(F). Free activation is unrestricted symmetric B = I + S, with three controls per active triangle and no amplitude or positive-definiteness projection. Smoothness weight is zero. The objective is mean squared vector displacement error over free top nodes; the reported normalized loss is L2/h².

The common optimizer budget is 1,200 Adam updates, initial learning rate 0.03, decay 0.99 per update, betas (0.9, 0.999), and epsilon 1e−8. Forward tolerance is 1e−10 with at most 250 Newton iterations. Each forward failure ends its trajectory and preserves the last accepted checkpoint.

## Endpoint results

| h | Last accepted update | L2/h² | Fit RMS | Activation neighbor RMS | Minimum physical J | Stop |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.05 | 1200 | 0.441198 | 0.033211 | 0.697183 | 0.466046 | Budget completed |
| 0.10 | 515 | 0.428426 | 0.065454 | 0.914443 | 0.000025985 | Forward iteration limit at 516 |
| 0.15 | 38 | 0.484745 | 0.104435 | 0.446768 | 0.669840 | Forward iteration limit at 39 |
| 0.20 | 262 | 0.442281 | 0.133008 | 1.007558 | 0.001764 | Forward line search failed at 263 |

All displayed checkpoints satisfy the original force tolerance and have zero inverted triangles. These checks do not establish mechanical stability; endpoint Hessian eigenvalues were not tested in this extension. Free activation may contain singular or indefinite B because the original parameterization is unconstrained.

The h = 0.10 failed proposal had force residual 4.703e−8. The h = 0.15 failed proposal had residual 1.032e−10, only about 3.2% above the strict tolerance after 250 iterations. Its early stop should not be interpreted as an intrinsically worse target height or a geometric breakdown. The original h = 0.20 failure residual was 3.191e−6. Failed proposal states are excluded from the figure.

The plotted endpoints have different optimization durations. The lower h = 0.10 normalized loss is an observed trajectory result, not a converged ranking of attainable fit. The h = 0.15 shape is visibly less developed because its solver stopped much earlier. Near-zero minimum J at h = 0.10 also limits any favorable interpretation based on fitting loss alone.

## Comparison at a common update

All four traces contain update 38, the last accepted update of the shortest run. These values are read directly from the per-update traces, without interpolation. The latest displacement snapshot available at an identical update across all four histories is update 30.

| h | L2/h² at update 38 | Activation neighbor RMS at update 38 |
| ---: | ---: | ---: |
| 0.05 | 0.472785 | 0.198059 |
| 0.10 | 0.483323 | 0.345186 |
| 0.15 | 0.484745 | 0.446768 |
| 0.20 | 0.487283 | 0.479755 |

At this common update, normalized fit error and activation variation both increase with target height. This is an early-trajectory observation under the stated Adam settings. Equal update counts do not imply equal forward-solver work or inverse convergence.

## Figure and verification

The PNG is exactly 3840 × 2160 pixels on a 16 × 9-inch canvas. All eight panels use x ∈ [−0.02, 1.02], y ∈ [−0.01, 0.31], equal physical x/y scales, and equal panel dimensions. Dashed curves are the targets. Activation glyphs show both eigenmodes of B − I, transported to the deformed configuration using F n / ‖F n‖. Their full length is 0.0168995, and the common signed color scale is ±3.653420. This color range is selected across the four displayed cases and differs from the original eight-case slide's range.

Independent assembly from each checkpoint verifies its force residual, physical determinant, and fitting loss. The renderer also checks mesh identity, control-to-B reconstruction, saved displacement agreement, panel bounds/scales, PNG dimensions, and vector SVG export with no embedded raster images. The preview was visually inspected for legibility and clipping. The new scripts pass Ruff. [Checks](../data/170-free-height-figures/delivery-checks.json) and saved glyph arrays retain the rendering evidence.

## Commands and run receipts

Run from `exp/2026/09/15/activation-direction-smoothness`:

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Free activation height comparison h010 h015' \
CHERRIES_TAGS='2d,free-activation,height-sweep,smoothness-off' \
uv run python src/160-run-free-heights.py > logs/160-run-free-heights-terminal.log 2>&1

COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME='Free activation four-height 16 by 9 slide' \
CHERRIES_TAGS='2d,free-activation,height-sweep,smoothness-off,16x9,figures' \
uv run python src/170-render-free-heights.py > logs/170-render-free-heights-terminal.log 2>&1
```

Both Cherries processes exited 0 after shutdown. Numerical forward failures are recorded within the successfully completed sweep process.

- Numerical run: [Comet](https://www.comet.com/liblaf/apple/f32ee44a147a4caba2567310a4fa2956), [Comet.ml Experiment Summary](../data/160-free-height-sweep/comet-summary.txt), [terminal log](../logs/160-run-free-heights-terminal.log), [protocol and source hashes](../data/160-free-height-sweep/protocol.json).
- Figure run: [Comet](https://www.comet.com/liblaf/apple/53f3d2944abe47e59868b0878c8a52db), [Comet.ml Experiment Summary](../data/170-free-height-figures/comet-summary.txt), [terminal log](../logs/170-render-free-heights-terminal.log).

The working tree already contained unrelated changes. Runs used the repository environment and disabled automatic Git commits. Computational sources were unchanged from the original recorded hashes; source snapshots record the executed files. Output directories must be new when rerunning; use the `--output` option for a new destination.
