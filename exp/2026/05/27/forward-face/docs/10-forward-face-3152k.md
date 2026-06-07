# Forward Face 3152k Expression001 SMAS100

## Purpose

Create a slim forward-physics experiment for the 3152k human face mesh, use `Expression001` as the target displacement reference, activate Zygomaticus major, and recover the selected activation with inverse physics.

The source mesh is:

`/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu`

This source was used because it contains `Expression001`, `MuscleOrientation`, and the `InFaceConvex` tetra mask.

## Commands

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/05/27/forward-face
```

Prep:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k expr001 smas100 prep" \
  CHERRIES_TAGS="forward-face,3152k,expr001,smas100,prep,zygomaticus-major" \
  uv run python src/10-prepare-forward-face.py
```

Selected forward run:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k expr001 smas100 zygomaticus a087" \
  CHERRIES_TAGS="forward-face,3152k,expr001,smas100,zygomaticus-major,activation-search" \
  uv run python src/20-forward-face.py
```

Successful inverse recovery:

```bash
env -u DEBUG \
  CHERRIES_NAME="forward-face 3152k expr001 smas100 inverse activation-only warm full" \
  CHERRIES_TAGS="forward-face,3152k,expr001,smas100,inverse,zygomaticus-major,activation-inv,activation-only,warm-start,full,series" \
  uv run python src/30-inverse-face-3152k.py \
    --diagonal-only false \
    --inverse-lr 0.04 \
    --inverse-min-steps 0 \
    --inverse-max-steps 30 \
    --initial-local-activation-inv-delta '[6.692307692307692,-0.393939393939394,-0.393939393939394,0,0,0]'
```

All report runs used `cherries.main(main)` without `profile="debug"`.

## Setup

The prep script extracts only tetrahedra with `InFaceConvex = true`. The extracted mesh has:

| Quantity | Value |
| --- | ---: |
| Points | 225052 |
| Tetrahedra | 1127541 |
| Zygomaticus-major activation tets | 2522 |
| Face target points | 17582 |
| Fixed cranium points | 18902 |
| Fixed mandible points | 7287 |
| Fixed points total | 26189 |
| `Expression001` face RMS | 0.297261 cm |
| `Expression001` face max | 1.251746 cm |

Each tet uses three material fractions:

| Component | Fraction | Activation | E | nu |
| --- | --- | --- | ---: | ---: |
| Fat/background | `1 - max(SmasFraction, MuscleFraction)` | none | 1.0 | 0.49 |
| Muscle | `MuscleFraction` | active | 100.0 | 0.49 |
| SMAS-only | `max(SmasFraction - MuscleFraction, 0)` | none | 100.0 | 0.49 |

Both cranium and mandible are fixed. The forward solve uses stable neo-Hookean potentials, no collisions, and PNCG defaults: `rtol = 5e-4`, `atol = 0`, `max_steps = 10000`.

## Forward Result

The selected local activation delta is:

```text
(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)
```

The script forms `I + delta_local`, rotates it by each active tet's muscle orientation, and stores both `Activation` and `ActivationInv`.

Final forward metrics:

| Metric | Value |
| --- | ---: |
| Forward result | `primary_success` |
| PNCG steps | 1174 |
| Relative gradient norm | 0.000377560 |
| Face RMS displacement | 0.200450 cm |
| Face max displacement | 1.614236 cm |
| Face RMS ratio to `Expression001` | 0.674322 |
| Lip top RMS displacement | 0.284019 cm |
| Lip top max displacement | 1.007321 cm |
| Lip bottom RMS displacement | 0.153947 cm |
| Lip bottom max displacement | 0.660220 cm |

The snapshot `data/20-forward-face-3152k-expr001-smas100.png` shows localized cheek and mouth-corner lift. The motion is a reasonable single-muscle deformation relative to the full `Expression001` target.

## Inverse Result

Initial 3152k inverse attempts failed when the differentiable path reset the full material tree every step. Direct probes showed that resetting all materials shifted the forward branch even with the same activation. The inverse library was updated so `DifferentiableForward` can accept a partial material tree and return gradients for the supplied leaves. The final inverse script now updates only `muscle.activation_inv`.

Successful inverse metrics:

| Metric | Value |
| --- | ---: |
| Passed | true |
| Stop reason | `max_point_error_tol` |
| Best step | 0 |
| Target face RMS error | 0.002361 cm |
| Target face max error | 0.011194 cm |
| Tolerance | 0.08 cm |
| Activation parameters | 6 |
| ActivationInv RMS error | 2.34e-7 |
| Forward steps inside inverse | 1143 |
| Forward relative gradient norm | 0.000321051 |

Recovered local activation delta:

```text
(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)
```

Recovered local `ActivationInv` delta:

```text
(6.692307692307692, -0.393939393939394, -0.393939393939394, 0.0, 0.0, 0.0)
```

The inverse target mask is the face point set. The summary also records an all-point max error of 0.971407 cm outside that target mask; the pass criterion is the requested target face max error.

## Cherries And Comet

| Run | Comet URL | Cherries Git SHA |
| --- | --- | --- |
| Prep | `https://www.comet.com/liblaf/apple/64226259a0c149b5aed22f1793fd9d5b` | `81401d11355df7466f001ea3bed580af7d9a07e8` |
| Forward | `https://www.comet.com/liblaf/apple/b72184f6ad4c4368b500f7d9a662e83b` | `2771084af1d44219de2fa38c5dba1006d887fb7e` |
| Forward rerun check | `https://www.comet.com/liblaf/apple/dd2ad712f8d44ec78e34e85233f66fde` | `ea293626752e861c1524a679397b39f268cc1c86` |
| Successful inverse | `https://www.comet.com/liblaf/apple/5e82908298164f4b8bd878ea39b12d52` | `6514a11e58de31d1b0e068ec52a2a22e5f33746f` |

Comet repeatedly warned during shutdown that some online logging failed. The local `data/` artifacts and JSON summaries are the authoritative evidence.

## Outputs

Prep outputs:

- `data/10-forward-face-3152k-expr001-smas100-input.vtu`
- `data/10-forward-face-3152k-expr001-smas100-target.vtu`

Forward outputs:

- `data/20-forward-face-3152k-expr001-smas100-input.vtu`
- `data/20-forward-face-3152k-expr001-smas100.vtu`
- `data/20-forward-face-3152k-expr001-smas100.png`
- `data/20-forward-face-3152k-expr001-smas100-summary.json`

Successful inverse outputs:

- `data/30-inverse-face-3152k-expr001-smas100-activation-only-input.vtu`
- `data/30-inverse-face-3152k-expr001-smas100-activation-only-target.vtu`
- `data/30-inverse-face-3152k-expr001-smas100-activation-only.vtu`
- `data/30-inverse-face-3152k-expr001-smas100-activation-only.png`
- `data/30-inverse-face-3152k-expr001-smas100-activation-only-summary.json`
- `data/30-inverse-face-3152k-expr001-smas100-activation-only.vtu.series`

The inverse series contains one frame at time 0, backed by `data/30-inverse-face-3152k-expr001-smas100-activation-only.vtu.d/30-inverse-face-3152k-expr001-smas100-activation-only_000000.vtu`.

## Verification

Validation checks passed:

- `uv run ruff check src/liblaf/apple/inverse/_diff_forward.py exp/2026/05/27/forward-face/src/30-inverse-face-3152k.py`
- `uv run python -m py_compile src/liblaf/apple/inverse/_diff_forward.py exp/2026/05/27/forward-face/src/30-inverse-face-3152k.py`
- JSON summary check confirmed `passed = true`, `target/error_max = 0.011193594470430808`, and `target/error_rms = 0.0023607901722359115`.
- Series check confirmed one manifest entry and one existing VTU frame.
- Visual inspection of `data/30-inverse-face-3152k-expr001-smas100-activation-only.png` showed low face-surface error and localized activation near Zygomaticus major.
