# Toy Forward Activation

## Purpose

This run completes the forward-physics part of the unreachable-inverse experiment group. It uses the same toy geometry, material fractions, material properties, and boundary conditions as the stretch/squash inverse runs:

- full body: `box(0, 1, 0, 0.1, 0, 1)`
- SMAS layer: `box(0, 1, 0.04, 0.06, 0, 1)`
- muscle region: `box(0, 0.5, 0.04, 0.06, 0.4, 0.6)`
- fixed boundary: bottom surface plus the four side surfaces
- muscle: `E = 1e2`, `nu = 0.49`, with activation
- aponeurosis: `E = 1e2`, `nu = 0.49`, no activation
- fat: `E = 1`, `nu = 0.49`

Each tetra has the requested material split:

- `MuscleFraction`
- `AponeurosisFraction = max(0, SmasFraction - MuscleFraction)`
- `FatFraction = 1 - AponeurosisFraction - MuscleFraction`

The active muscle tetrahedra are contracted along the global x-axis with additive activation `(-0.5, 0, 0, 0, 0, 0)`. The corresponding stored `ActivationInv` value is `(1, 0, 0, 0, 0, 0)` on active muscle tetrahedra.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-inverse-physics
```

Command:

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="toy forward x-axis muscle contraction" \
CHERRIES_TAGS="unreachable-inverse,toy,forward,activation,nu049,x-contraction" \
uv run python src/40-toy-forward-activation.py
```

Comet run: <https://www.comet.com/liblaf/apple/ebe3188b59544d4a9d51a6673e5f8f8e>

Cherries summary fields from the run log:

- `cherries/entrypoint`: `exp/2026/06/10/unreachable-inverse-physics/src/40-toy-forward-activation.py`
- `cherries/exp_dir`: `exp/2026/06/10/unreachable-inverse-physics`
- `cherries/git/sha`: `d52c909b97b7b0abf77f5e24229284b22119a2d2`
- `cherries/start_time`: `2026-06-10 15:19:57.573556+08:00`
- `cherries/end_time`: `2026-06-10 15:20:02.771635+08:00`

## Outputs

- `data/40-toy-forward-activation-summary.json`
- `data/40-toy-forward-activation-cases.csv`
- `data/40-toy-forward-activation-table.md`
- `data/40-toy-forward-activation-coarse-input.vtu`
- `data/40-toy-forward-activation-coarse.vtu`
- `data/40-toy-forward-activation-medium-input.vtu`
- `data/40-toy-forward-activation-medium.vtu`
- `data/40-toy-forward-activation-fine-input.vtu`
- `data/40-toy-forward-activation-fine.vtu`
- `logs/40-toy-forward-activation.log`

The output VTUs remain in rest coordinates and store `Displacement`, `DisplacementNorm`, `DeformedPoint`, activation arrays, and per-tetra volume diagnostics. The forward-named fields include:

- `VolumeForward`
- `VolumeForwardRelChange`
- `SignedVolumeForward`
- `SignedVolumeForwardRelChange`

## Results

| case | points | tets | active tets | result | steps | signed volume change | surface area change | top area change | top y mean | top y std | top edge RMS | displacement RMS | displacement max |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse` | 567 | 2304 | 96 | `primary_success` | 109 | -1.7708% | 0.1121% | -0.8641% | -0.001675 | 0.007787 | 0.007586 | 0.026652 | 0.109190 |
| `medium` | 2475 | 11760 | 224 | `primary_success` | 250 | -1.0488% | 0.3328% | 0.7053% | -0.000863 | 0.007236 | 0.006925 | 0.021395 | 0.107098 |
| `fine` | 4851 | 24000 | 480 | `primary_success` | 175 | -1.2809% | 0.1489% | 0.1150% | -0.000909 | 0.006235 | 0.003361 | 0.021540 | 0.102680 |

All three forward solves reached `primary_success`. No tetrahedra inverted in the saved forward solutions.

## Analysis

The x-axis contraction produces a localized deformation whose largest point displacement is about `0.10` to `0.11`, concentrated near the active muscle patch. The global signed-volume change stays small and negative, between `-1.05%` and `-1.77%`, which is consistent with the nearly incompressible `nu = 0.49` material resisting large volume change.

The top-surface response is not a uniform lift or squash. Its mean y displacement is slightly negative on all resolutions, but the top y range is much larger than the mean: `0.0383` on coarse, `0.0558` on medium, and `0.0325` on fine. This provides a forward baseline for the bumpy inverse observations: even a valid forward contraction from the small active region creates a nonuniform top-surface displacement.

Compared with the inverse stretch/squash targets, the forward contraction is much less volume-changing. The inverse targets asked for roughly `+15%` to `+18%` or `-15%` to `-18%` signed volume change, while this forward activation lands near `-1%`. That supports the earlier conclusion that the prescribed uniform top motion is far outside the volume response naturally available from the active patch under high Poisson ratio and fixed boundaries.

## Reproducibility

The script imports the shared toy helpers from `src/20-toy-unreachable-inverse.py`, so the geometry, fraction definitions, boundary masks, and forward material construction are shared with the inverse toy runs. The run used CUDA through Warp/Torch on `cuda:0` and Cherries recorded a Comet experiment. `uv run python -m py_compile` and `uv run ruff check` passed for the script before the report-worthy run.
