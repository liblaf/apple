# 3152k Activation Transfer Forward Run

## Purpose

Transfer the solved 515k per-active-tet `activation_inv` field from
`20-inverse-face-smooth-w1-lr003.vtu.series` to the 3152k anatomy mesh and run
one high-resolution forward solve with the transferred active material field.

## Command

```bash
cd exp/2026/05/20/inverse-face
DEBUG=false CHERRIES_NAME="transfer activation 3152k forward" CHERRIES_TAGS="inverse-face,transfer-activation,3152k,forward" uv run python src/30-transfer-activation-3152k.py
```

The default `solved_frame_index=-1` selected the last frame of the 515k solved
series: `20-inverse-face-smooth-w1-lr003_000256.vtu`.

## Method

The script extracts `InFaceConvex` tetrahedra from
`41-expression-3152k.vtu`, rebuilds the same material and fixed-cranium fields
used by the 515k inverse-face experiment, then transfers cell-centered
`RecoveredActivationInv` by inverse-distance weighting over active-cell centers
with `k=4` and `power=2`. Inactive target cells keep zero activation.

The transferred full `(n_cells, 6)` tensor is installed as
`materials["muscle"]["activation_inv"]`, and the regular `Forward.step()` path
runs to equilibrium with `forward_rtol=5e-4`.

## Results

- Extracted high-resolution face mesh: 225,052 points and 1,127,541 tetrahedra.
- Active target tetrahedra: 283,391.
- Source active tetrahedra: 58,494.
- Transfer nearest active-center distance max: 1.167946444926288.
- Transferred active activation RMS: 0.5188147365068417.
- Forward result: `primary_success`.
- Forward steps: 462.
- Forward relative gradient norm: 0.0003410798327373236.
- Target error: mean 0.19756966452709088 cm, RMS 0.2834989958453261 cm, max 0.925671095808897 cm.
- Forward solve time: 16.038381466001738 s.
- Total script time: 26.93065482401289 s.

## Assets

- Prepared transferred input: `data/30-transfer-activation-3152k-input.vtu`
- Forward output: `data/30-transfer-activation-3152k.vtu`
- Single-frame output series: `data/30-transfer-activation-3152k.vtu.series`
- Summary JSON: `data/30-transfer-activation-3152k-summary.json`
- Run log: `logs/30-transfer-activation-3152k.log`
- Comet run: <https://www.comet.com/liblaf/apple/1708acd3c5f34abf8276c32b4d73351a>

## Notes

The final Comet git-patch logging phase stalled on large VTU diffs and was
interrupted after outputs and the Cherries summary were written. Comet still
recorded the main metrics and source metadata, but the log ends with
`Failed to log run in comet.com`.

Cherries Experiment Summary:

```yaml
---
name: transfer activation 3152k forward
url: https://www.comet.com/liblaf/apple/1708acd3c5f34abf8276c32b4d73351a
exp_dir: exp/2026/05/20/inverse-face
cwd: exp/2026/05/20/inverse-face
cmd: /home/liblaf/github/liblaf/apple/.venv/bin/python3 src/30-transfer-activation-3152k.py
params:
  E: 1.0
  active_fraction_tol: 0.001
  expression: Expression000
  fixed_point_mask: IsCranium
  forward_atol: 0.0
  forward_max_steps: 10000
  forward_rtol: 0.0005
  nu: 0.49
  output: /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k.vtu
  output_input: /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k-input.vtu
  output_series: /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k.vtu.series
  output_summary: /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/30-transfer-activation-3152k-summary.json
  smas_stiffness_ratio: 100.0
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
