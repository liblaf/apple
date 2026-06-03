# Forward Face 3152k Zygomaticus Major

## Purpose

Create a slim forward-physics experiment for the 3152k human face mesh, activate only Zygomaticus major, and manually choose a local activation that gives a visible but not exaggerated deformation.

The selected source mesh is:

`/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu`

This mesh was used instead of `41-expression-3152k.vtu` because it contains both `Expression000` and `MuscleOrientation`.

## Commands

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/05/27/forward-face
```

Prep:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k prep" \
  CHERRIES_TAGS="forward-face,3152k,prep,zygomaticus-major" \
  uv run python src/10-prepare-forward-face.py
```

Selected activation probe:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k zygomaticus a087" \
  CHERRIES_TAGS="forward-face,3152k,zygomaticus-major,activation-search" \
  uv run python src/20-forward-face.py \
    --activation-local '[-0.87,0.65,0.65,0,0,0]' \
    --output-stem 20-forward-face-3152k-a087
```

Final canonical run:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k zygomaticus final" \
  CHERRIES_TAGS="forward-face,3152k,zygomaticus-major,final" \
  uv run python src/20-forward-face.py
```

## Cherries And Comet

Prep Comet URL:

`https://www.comet.com/liblaf/apple/7e59a3006fb340fcaeaebf398c13e5cc`

Selected probe Comet URL:

`https://www.comet.com/liblaf/apple/ca28940801904f2f8fe388117596a9f1`

Final Comet URL:

`https://www.comet.com/liblaf/apple/9e4c4457d5004d40b4b497940bd9496e`

The final Cherries run exited with code 0 and printed a full local Comet summary. Comet also warned during shutdown that online logging failed, so the local files in `data/` and `logs/` are the authoritative evidence for this report.

## Setup

The prep script extracts only tetrahedra where `InFaceConvex` is true. The resulting face-convex mesh has:

| Quantity | Value |
| --- | ---: |
| Points | 225052 |
| Tetrahedra | 1127541 |
| Zygomaticus-major activation tets | 2522 |
| Face target points | 17582 |
| Fixed cranium points | 18902 |
| Fixed mandible points | 7287 |
| Fixed points total | 26189 |

Zygomaticus major is represented by `MuscleId` 46 and 47, corresponding to `Zygomaticus_major001_00` and `Zygomaticus_major001_01`.

Each tet uses three material fractions:

| Component | Fraction | Activation | E | nu |
| --- | --- | --- | ---: | ---: |
| Fat/background | `1 - max(SmasFraction, MuscleFraction)` | none | 1.0 | 0.49 |
| Muscle | `MuscleFraction` | active model | 1.0 | 0.49 |
| SMAS-only | `max(SmasFraction - MuscleFraction, 0)` | none | 1.0 | 0.49 |

The forward solve uses the new `liblaf.apple.forward` `ModelBuilder`/`Forward` path with stable neo-Hookean potentials, no collisions, and PNCG defaults: `rtol = 5e-4`, `atol = 0`, `max_steps = 10000`.

## Activation Search

The initial suggested activation `(-0.5, 0.2, 0.1, 0, 0, 0)` was too small:

| Run | Face RMS | Face Max | Lip Top RMS | Lip Top Max | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| start | 0.005038 cm | 0.053917 cm | 0.003319 cm | 0.014037 cm | too small |
| selected `a087` | 0.125168 cm | 0.818799 cm | 0.226329 cm | 0.447926 cm | reasonable |
| final canonical | 0.126191 cm | 0.826099 cm | 0.229352 cm | 0.456870 cm | selected |

The selected activation local delta is:

```text
(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)
```

The script treats this as an additive local delta, forms `I + delta_local`, rotates it by each tet's muscle orientation, and stores both `Activation` and `ActivationInv` in cell data.

## Final Results

Final forward summary:

| Metric | Value |
| --- | ---: |
| Forward result | `primary_success` |
| PNCG steps | 619 |
| Relative gradient norm | 0.000497563 |
| Face RMS displacement | 0.126191 cm |
| Face max displacement | 0.826099 cm |
| Face RMS ratio to `Expression000` | 0.411616 |
| Lip top RMS displacement | 0.229352 cm |
| Lip top max displacement | 0.456870 cm |
| Lip bottom RMS displacement | 0.123553 cm |
| Lip bottom max displacement | 0.394535 cm |

The final snapshot shows localized cheek and mouth-corner deformation. It is clearly visible and roughly 40% of the full `Expression000` face RMS, which is appropriate for a single Zygomaticus-major activation rather than the complete expression.

## Outputs

Prep outputs:

- `data/10-forward-face-3152k-input.vtu`
- `data/10-forward-face-3152k-target.vtu`

Selected probe outputs:

- `data/20-forward-face-3152k-a087-input.vtu`
- `data/20-forward-face-3152k-a087.vtu`
- `data/20-forward-face-3152k-a087.png`
- `data/20-forward-face-3152k-a087-summary.json`

Final canonical outputs:

- `data/20-forward-face-3152k-input.vtu`
- `data/20-forward-face-3152k.vtu`
- `data/20-forward-face-3152k.png`
- `data/20-forward-face-3152k-summary.json`

The final VTU keeps the mesh in rest coordinates and stores the deformation in `point_data["Displacement"]`. It also includes `DeformedPoint`, `TargetDisplacement`, `TargetPoint`, and displacement norm/error arrays for inspection.

## Verification

Validation checks passed:

- `uv run ruff check exp/2026/05/27/forward-face/src/10-prepare-forward-face.py exp/2026/05/27/forward-face/src/20-forward-face.py`
- `uv run python -m py_compile exp/2026/05/27/forward-face/src/10-prepare-forward-face.py exp/2026/05/27/forward-face/src/20-forward-face.py`
- PyVista sanity check confirmed the final result mesh has rest-shape points equal to the prep input, required point/cell arrays are present, 2522 activation tets are selected, and inactive tets have zero activation.

## Reproducibility Notes

Final Cherries Git SHA: `9f01c05f65ab4b8257cc873bfbf20d048f7d92ee`.

At report time, `main` is ahead of `origin/main` by 7 commits from Cherries experiment commits. The final canonical run is commit `9f01c05f` with message `chore(exp): forward-face 3152k zygomaticus final`. This report and the final worklog notes are uncommitted on top of that Cherries commit.

Limitations:

- Comet printed a URL and local summary, but the final shutdown reported online logging failure. Use local `data/20-forward-face-3152k-summary.json`, `data/20-forward-face-3152k.vtu`, and `logs/20-forward-face.log` as authoritative.
- No collision handling was enabled, by request.
- The activation was manually bracketed for a reasonable single-muscle deformation, not optimized to match all of `Expression000`.
