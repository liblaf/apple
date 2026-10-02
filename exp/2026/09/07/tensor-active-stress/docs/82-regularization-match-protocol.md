# Regularization match and conditional rank decision

`src/82-match-regularization.py` is a CPU-only post-processing stage. It does
not run an optimizer, alter a checkpoint, or choose a best state. It consumes
three completed 16-update source-74 regularization arms forked from the same
fit-only endpoint: control, weak smoothness, and full smoothness.

Create `docs/82-regularization-match-manifest.json` only after all three arms
complete. The manifest must use this shape, with paths relative to it:

```json
{
  "schema_version": 1,
  "title": "PSD regularization match",
  "output_dir": "../data/82-regularization-match",
  "surface_protocol": {
    "fixture_vtu": "../../face-actuation-diagnosis/data/12-historical-fixture/volume.vtu",
    "skin_vtp": "../../face-actuation-diagnosis/data/12-historical-fixture/skin.vtp",
    "scales_mm": [2, 5, 10],
    "primary_scale_mm": 5,
    "mouth_radius_mm": 10
  },
  "arms": {
    "control": {"id": "fit-control", "label": "Fit-only control", "path": "../data/..."},
    "weak": {"id": "weak-smooth", "label": "Weak smoothness", "path": "../data/..."},
    "full": {"id": "full-smooth", "label": "Full smoothness", "path": "../data/..."}
  }
}
```

The three directories must be successful source-74 regularization runs with
identical parent checkpoint, controls, seed displacement, optimizer settings,
physics provenance, and 16-update trace layout. Their only permitted loss
change is smoothness: control `0`, weak `1.4762928671047126`, full
`5.9051714684188505`; rank and magnitude weights remain zero.

For each smoothness strength, the stage compares every pair of solver-valid
saved states with positive local step. It excludes local step zero. A pair is
admissible when absolute area-fit difference is at most
`max(0.02 mm, 0.005 * control fit)` and absolute area-motion difference is at
most `max(0.02 mm, 0.01 * control motion)`. It selects the admissible pair with
least squared distance after division by those tolerances, breaking ties by the
earlier smooth step and then the earlier control step. Equal-step rows remain
in the inventory but do not replace cross-step matching.

For the selected pair, the stage recomputes the frozen source-40 full-face
5-mm normal-displacement high-pass ratio using source-80's exact helper path.
It uses the trace's normalized-coordinate geometric tensor variation (`smoothness`) and
records its ratio to the matched control. A smoothness arm qualifies for the
rank branch only if both the variation and the defined full-face normalized
high-pass ratio decrease by at least 10 percent and its matched
`rank_mixing_fraction` exceeds `0.05`. If both qualify, the weak arm is chosen;
otherwise the sole qualifying arm is chosen. If neither qualifies, the output
records why rank is skipped. The receipt names the exact selected smooth NPZ, VTU, and
`optimizer-step-<global>.pt` checkpoint plus its paired control checkpoint,
without starting a rank run. Each saved state must have that complete checkpoint
binding; the final `optimizer-latest.pt` cannot stand in for an intermediate
selected state.

Run only after all inputs exist:

```bash
cd exp/2026/09/07/tensor-active-stress
env COMET_AUTO_LOG_GIT_METADATA=false COMET_AUTO_LOG_GIT_PATCH=false \
  COMET_AUTO_LOG_ENV_DETAILS=false \
  CHERRIES_NAME='Tensor active-stress regularization match' \
  CHERRIES_TAGS='cpu,face,tensor-active-stress,regularization,postprocess' \
  .venv/bin/python \
  src/82-match-regularization.py --manifest docs/82-regularization-match-manifest.json
```
