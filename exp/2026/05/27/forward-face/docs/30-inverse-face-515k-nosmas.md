# Forward Face 515k No-SMAS Inverse

## Purpose

Use the 515k no-SMAS forward solution as the target deformation and recover a Zygomaticus-major activation with inverse physics.

This run optimizes one shared local six-component `ActivationInv` delta over the active Zygomaticus-major tetrahedra. It is full 6-DoF in the symmetric activation tensor components, but not a per-tet activation field.

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/05/27/forward-face
```

Run command:

```bash
CHERRIES_NAME="forward-face 515k inverse nosmas activation-inv full6 series" \
CHERRIES_TAGS="forward-face,515k,inverse,zygomaticus-major,nosmas,activation-inv,full6,series" \
uv run python src/30-inverse-face-515k.py --max-point-error-cm 0.08
```

Comet URL:

`https://www.comet.com/liblaf/apple/2a616c45e54b4557ac0f7a2c21f959da`

## Setup

Inputs:

- `data/10-forward-face-515k-nosmas-input.vtu`
- `data/20-forward-face-515k-nosmas.vtu`

The target is the previous 515k no-SMAS forward result. The solve uses stable neo-Hookean material with `E = 1.0`, `nu = 0.49`, no SMAS potential, no collisions, and the default PNCG forward optimizer with `rtol = 5e-4`, `atol = 0`, `max_steps = 10000`.

The inverse optimizer is Adam over six local `activation_inv` parameters:

```text
(xx, yy, zz, xy, xz, yz)
```

## Results

The inverse solve stopped by `max_point_error_tol` at step 93 and passed the requested tolerance.

| Metric | Value |
| --- | ---: |
| Passed | true |
| Stop reason | `max_point_error_tol` |
| Best step | 93 |
| Trace length | 94 |
| Target points | 6787 |
| Active tets | 420 |
| Target displacement RMS | 0.071006 cm |
| Target displacement max | 0.599423 cm |
| Recovered face RMS error | 0.016721 cm |
| Recovered face max error | 0.076498 cm |
| All-point RMS error | 0.009969 cm |
| All-point max error | 0.085630 cm |
| Total wall time | 385.990 s |

Recovered local activation delta:

```text
(-0.855972, 0.179139, 0.965528, 0.011184, -0.028203, -0.067857)
```

Recovered local `ActivationInv` delta:

```text
(5.966956, -0.149711, -0.488843, -0.060450, 0.097881, 0.028487)
```

Compared with the target activation embedded in the forward result, the recovered `ActivationInv` has RMS error `0.249177` and max active-tet norm error `0.614189`. The deformation match is good enough for this tolerance, but the recovered activation is not an exact parameter recovery.

## Visualization Outputs

Yes: this run saves every evaluated inverse step as a ParaView series.

- `data/30-inverse-face-515k-nosmas.vtu.series`
- `data/30-inverse-face-515k-nosmas.vtu.d/30-inverse-face-515k-nosmas_000000.vtu`
- ...
- `data/30-inverse-face-515k-nosmas.vtu.d/30-inverse-face-515k-nosmas_000093.vtu`

The series manifest contains 94 frames, with times `0.0` through `93.0`. The frame directory is about `1.9G`.

Each frame includes displacement, target displacement, displacement error, recovered `Activation`, recovered `ActivationInv`, target `ActivationInv` when present, `ActivationInvError`, inverse masks, and scalar step metrics in field data.

Other outputs:

- `data/30-inverse-face-515k-nosmas-input.vtu`
- `data/30-inverse-face-515k-nosmas-target.vtu`
- `data/30-inverse-face-515k-nosmas.vtu`
- `data/30-inverse-face-515k-nosmas.png`
- `data/30-inverse-face-515k-nosmas-summary.json`
- `data/30-inverse-face-515k-nosmas-checkpoint.npz`
- `logs/30-inverse-face-515k.log`

## Verification

Validation checks passed before the Cherries run:

- `uv run ruff check exp/2026/05/27/forward-face/src/30-inverse-face-515k.py`
- `uv run python -m py_compile exp/2026/05/27/forward-face/src/30-inverse-face-515k.py`

Post-run checks:

- The `.vtu.series` manifest exists and lists 94 files.
- The frame directory contains 94 `.vtu` files.
- `data/30-inverse-face-515k-nosmas-summary.json` reports `passed = true`.
- Final Cherries run commit: `4fc323fd`.

## Notes

The first inverse attempt wrote a checkpoint but did not save a `.vtu.series`. I stopped relying on that attempt, patched `src/30-inverse-face-515k.py` to append a result mesh after every evaluated inverse step, and reran the inverse solve.

The local artifacts were written before the Cherries upload/git-patch tail completed. The local `data/`, `logs/`, and Cherries commit are the evidence used for this report.
