# Corrected 515k/3152k Transfer Forward Matrix

## Purpose

Run the transferred `activation_inv` forward test across both source mesh
resolutions and both SMAS stiffness settings after correcting the material
fractions to be disjoint:

- muscle fraction: `MuscleFraction`, active material, `E = smas_stiffness_ratio * E`
- SMAS-only fraction: `max(SmasFraction - MuscleFraction, 0)`, passive material,
  `E = smas_stiffness_ratio * E`
- background fraction: `1 - max(SmasFraction, MuscleFraction)`, passive material,
  `E = 1`

## Command

```bash
cd exp/2026/05/20/inverse-face
DEBUG=1 CHERRIES_NAME="$stem" CHERRIES_TAGS="inverse-face,transfer-activation,forward,matrix" uv run python src/30-transfer-activation-3152k.py --source "$source" --smas-stiffness-ratio "$ratio" --output-input "data/${stem}-input.vtu" --output "data/${stem}.vtu" --output-series "data/${stem}.vtu.series" --output-summary "data/${stem}-summary.json"
```

The four concrete cases were:

- `40-transfer-activation-515k-smas100`
- `41-transfer-activation-515k-smas1`
- `42-transfer-activation-3152k-smas100`
- `43-transfer-activation-3152k-smas1`

## Results

| case | cells | active tets | steps | target mean cm | target RMS cm | target max cm | all RMS cm | all max cm | forward s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 515k, SMAS 100 | 253876 | 58494 | 421 | 0.422151 | 0.475055 | 1.297573 | 0.368954 | 9.127733 | 3.825 |
| 515k, SMAS 1 | 253876 | 58494 | 244 | 0.274502 | 0.318913 | 0.942866 | 0.243835 | 1.195187 | 3.633 |
| 3152k, SMAS 100 | 1127541 | 283391 | 82 | 0.242941 | 0.309361 | 1.033624 | 0.200444 | 3.558038 | 4.520 |
| 3152k, SMAS 1 | 1127541 | 283391 | 183 | 0.232794 | 0.283870 | 0.895340 | 0.198317 | 1.029727 | 6.647 |

All four forward solves returned `primary_success`.

## Assets

- `data/40-transfer-activation-515k-smas100.vtu`
- `data/40-transfer-activation-515k-smas100-summary.json`
- `data/40-transfer-activation-515k-smas100.vtu.series`
- `data/41-transfer-activation-515k-smas1.vtu`
- `data/41-transfer-activation-515k-smas1-summary.json`
- `data/41-transfer-activation-515k-smas1.vtu.series`
- `data/42-transfer-activation-3152k-smas100.vtu`
- `data/42-transfer-activation-3152k-smas100-summary.json`
- `data/42-transfer-activation-3152k-smas100.vtu.series`
- `data/43-transfer-activation-3152k-smas1.vtu`
- `data/43-transfer-activation-3152k-smas1-summary.json`
- `data/43-transfer-activation-3152k-smas1.vtu.series`

## Notes

The 3152k extracted face mesh has 214,828 cells where muscle and SMAS both have
positive fraction and 94,206 cells where muscle exceeds SMAS. With the corrected
formula, `background + muscle + smas_only` stays in `[0.9999999999999999, 1.0]`
on the 3152k extracted mesh.

These runs used `DEBUG=1`, so there are no Comet URLs and no automatic
experiment commits. Because the Cherries path helpers queue default paths before
CLI overrides are applied, the Cherries summary block may list the script's
default `30-...` outputs even though the actual files were saved under the
explicit `40`-`43` stems above.
