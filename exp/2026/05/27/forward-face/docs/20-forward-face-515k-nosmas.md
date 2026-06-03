# Forward Face 515k No-SMAS Follow-Up

## Purpose

Run the same selected Zygomaticus-major activation from the 3152k forward-face experiment on the 515k mesh, without SMAS stiffness.

Selected activation local delta:

```text
(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)
```

Source mesh:

`/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-515k.vtu`

## Commands

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/05/27/forward-face
```

Prep:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 515k nosmas prep" \
  CHERRIES_TAGS="forward-face,515k,prep,zygomaticus-major,nosmas" \
  uv run python src/10-prepare-forward-face.py \
    --source /home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-515k.vtu \
    --output-stem 10-forward-face-515k-nosmas \
    --use-smas false
```

Forward:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 515k nosmas zygomaticus" \
  CHERRIES_TAGS="forward-face,515k,zygomaticus-major,nosmas" \
  uv run python src/20-forward-face.py \
    --input data/10-forward-face-515k-nosmas-input.vtu \
    --target data/10-forward-face-515k-nosmas-target.vtu \
    --output-stem 20-forward-face-515k-nosmas \
    --use-smas false
```

## Cherries And Comet

Prep Comet URL:

`https://www.comet.com/liblaf/apple/8bca34d7cd094bc3ab7420f76eaadd1d`

Forward Comet URL:

`https://www.comet.com/liblaf/apple/c119ee752e624b7aaca41dbe28cc54d3`

The local outputs were written before the Cherries upload tail finished. Local `data/` and `logs/` files are the authoritative evidence.

## Setup

The prep extracts only `InFaceConvex` tetrahedra from the 515k orientation mesh.

| Quantity | Value |
| --- | ---: |
| Points | 58651 |
| Tetrahedra | 253876 |
| Zygomaticus-major activation tets | 420 |
| Face target points | 6787 |
| Fixed cranium points | 9078 |
| Fixed mandible points | 4808 |
| Fixed points total | 13886 |
| SMAS stiffness fraction volume | 0.0 |

No-SMAS mode sets:

| Component | Fraction | Activation | E | nu |
| --- | --- | --- | ---: | ---: |
| Fat/background | `1 - MuscleFraction` | none | 1.0 | 0.49 |
| Muscle | `MuscleFraction` | active model | 1.0 | 0.49 |
| SMAS | `0` | none | disabled | disabled |

The forward solve used stable neo-Hookean material, no collisions, and PNCG with `rtol = 5e-4`, `atol = 0`, `max_steps = 10000`.

## Results

| Metric | Value |
| --- | ---: |
| Forward result | `primary_success` |
| PNCG steps | 322 |
| Relative gradient norm | 0.000499942 |
| Face RMS displacement | 0.071006 cm |
| Face max displacement | 0.599423 cm |
| Face RMS ratio to `Expression000` | 0.202021 |
| Lip top RMS displacement | 0.096671 cm |
| Lip top max displacement | 0.257022 cm |
| Lip bottom RMS displacement | 0.052231 cm |
| Lip bottom max displacement | 0.184981 cm |

The deformation is visible and localized, but smaller than the 3152k SMAS-enabled final run. On the 515k no-SMAS setup, the same activation produced about 20% of the 515k `Expression000` face RMS.

## Outputs

Prep outputs:

- `data/10-forward-face-515k-nosmas-input.vtu`
- `data/10-forward-face-515k-nosmas-target.vtu`

Forward outputs:

- `data/20-forward-face-515k-nosmas-input.vtu`
- `data/20-forward-face-515k-nosmas.vtu`
- `data/20-forward-face-515k-nosmas.png`
- `data/20-forward-face-515k-nosmas-summary.json`

The result VTU keeps rest coordinates and stores deformation in `point_data["Displacement"]`.

## Verification

Validation checks passed:

- `uv run ruff check exp/2026/05/27/forward-face/src/10-prepare-forward-face.py exp/2026/05/27/forward-face/src/20-forward-face.py`
- `uv run python -m py_compile exp/2026/05/27/forward-face/src/10-prepare-forward-face.py exp/2026/05/27/forward-face/src/20-forward-face.py`
- PyVista sanity check confirmed rest-shape points are preserved, required point/cell arrays are present, 420 activation tets are selected, inactive tets have zero activation, and `SmasStiffnessFraction` sums to zero.
