# 3152k Activation Transfer Forward Run, SMAS Ratio 1

## Purpose

Repeat the 3152k transferred-activation forward test with no extra SMAS
stiffening, using `smas_stiffness_ratio=1.0`.

## Command

```bash
cd exp/2026/05/20/inverse-face
DEBUG=1 CHERRIES_NAME="transfer activation 3152k smas1 forward" CHERRIES_TAGS="inverse-face,transfer-activation,3152k,forward,smas1" uv run python src/30-transfer-activation-3152k.py --smas-stiffness-ratio 1.0 --output-input data/31-transfer-activation-3152k-smas1-input.vtu --output data/31-transfer-activation-3152k-smas1.vtu --output-series data/31-transfer-activation-3152k-smas1.vtu.series --output-summary data/31-transfer-activation-3152k-smas1-summary.json
```

The run used the last frame of
`data/20-inverse-face-smooth-w1-lr003.vtu.series` and the same transfer
settings as the SMAS ratio 100 run: inverse-distance weighting over active
cell centers with `k=4` and `power=2`.

## Results

- Extracted high-resolution face mesh: 225,052 points and 1,127,541 tetrahedra.
- Active target tetrahedra: 283,391.
- Source active tetrahedra: 58,494.
- Forward result: `primary_success`.
- Forward steps: 188.
- Forward relative gradient norm: 0.00042317457894796293.
- Target error: mean 0.21057998403163483 cm, RMS 0.26809221274006007 cm, max 0.9064648852118059 cm.
- All-point error: RMS 0.18009951096673188 cm, max 1.0347772454701907 cm.
- Forward solve time: 6.737611743999878 s.
- Total script time: 14.016527194995433 s.

Compared with the SMAS ratio 100 run, `smas_stiffness_ratio=1.0` reduced the
forward step count from 462 to 188, reduced target RMS error from
0.2834989958453261 cm to 0.26809221274006007 cm, and reduced target max error
from 0.925671095808897 cm to 0.9064648852118059 cm. Target mean error increased
from 0.19756966452709088 cm to 0.21057998403163483 cm, and all-point RMS error
increased from 0.13328656562825245 cm to 0.18009951096673188 cm.

## Assets

- Prepared transferred input: `data/31-transfer-activation-3152k-smas1-input.vtu`
- Forward output: `data/31-transfer-activation-3152k-smas1.vtu`
- Single-frame output series: `data/31-transfer-activation-3152k-smas1.vtu.series`
- Summary JSON: `data/31-transfer-activation-3152k-smas1-summary.json`
- Run log: `logs/30-transfer-activation-3152k.log`

## Notes

This was run with `DEBUG=1`, so there is no Comet run URL and no automatic
experiment commit. Because `cherries.output(...)` records the script defaults
before CLI overrides are applied, the Cherries summary block lists the default
`30-...` outputs even though the actual saved files are the `31-...-smas1`
paths above.

Cherries Experiment Summary:

```yaml
---
name: transfer activation 3152k smas1 forward
exp_dir: exp/2026/05/20/inverse-face
cwd: exp/2026/05/20/inverse-face
cmd: /home/liblaf/github/liblaf/apple/.venv/bin/python3 src/30-transfer-activation-3152k.py
  --smas-stiffness-ratio 1.0 --output-input data/31-transfer-activation-3152k-smas1-input.vtu
  --output data/31-transfer-activation-3152k-smas1.vtu --output-series data/31-transfer-activation-3152k-smas1.vtu.series
  --output-summary data/31-transfer-activation-3152k-smas1-summary.json
params:
  E: 1.0
  active_fraction_tol: 0.001
  expression: Expression000
  fixed_point_mask: IsCranium
  forward_atol: 0.0
  forward_max_steps: 10000
  forward_rtol: 0.0005
  nu: 0.49
  output: data/31-transfer-activation-3152k-smas1.vtu
  output_input: data/31-transfer-activation-3152k-smas1-input.vtu
  output_series: data/31-transfer-activation-3152k-smas1.vtu.series
  output_summary: data/31-transfer-activation-3152k-smas1-summary.json
  smas_stiffness_ratio: 1.0
  solved: /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face-smooth-w1-lr003.vtu.series
  solved_frame_index: -1
  source: /home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/41-expression-3152k.vtu
  target_point_mask: IsFace
  target_scale: 1.0
  transfer_chunk_size: 200000
  transfer_k: 4
  transfer_power: 2.0
inputs:
  - /home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/41-expression-3152k.vtu
  - exp/2026/05/20/inverse-face/data/20-inverse-face-smooth-w1-lr003.vtu.series
outputs:
  - exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k-input.vtu
  - exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k.vtu
  - exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k.vtu.series
  - exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k-summary.json
---
```
