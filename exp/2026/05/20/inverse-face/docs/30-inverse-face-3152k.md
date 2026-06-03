# 3152k Inverse Face Solve

## Purpose

Solve inverse physics for `Expression000` on the 3152k human-face model using
the new forward and inverse libraries. The solve optimizes six activation
degrees of freedom per active muscle tetrahedron and measures error only on
points where `IsFace` is true.

The final acceptance target is max face-point displacement error below
`0.2 cm`.

## Commands

Prepare the face-only tetrahedral problem:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face
DEBUG=false CHERRIES_NAME='prepare inverse face 3152k' CHERRIES_TAGS='inverse-face,prepare,3152k,Expression000' uv run python src/10-prepare-inverse-face.py
```

Run Adam on the 3152k inverse problem:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face
env -u DEBUG COMET_AUTO_LOG_GIT_PATCH=false CHERRIES_NAME='inverse face 3152k smas100 fresh' CHERRIES_TAGS='inverse-face,inverse,3152k,smas100,Expression000,stable-neo-hookean,adam' uv run python src/20-inverse-face.py --inverse-lr 0.03 --adam-beta1 0.3 --adam-beta2 0.9 --activation-smooth-weight 0.001 --activation-l2-weight 1e-5 --inverse-max-steps 2000 --inverse-min-steps 80 --stagnation-patience 300 --stagnation-rel-tol 1e-5 --stagnation-abs-tol 1e-8
```

Finalize and verify the saved best checkpoint without overwriting the
69-frame optimization series:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face
env -u DEBUG COMET_AUTO_LOG_GIT_PATCH=false CHERRIES_NAME='inverse face 3152k smas100 best checkpoint finalize' CHERRIES_TAGS='inverse-face,inverse,3152k,smas100,Expression000,stable-neo-hookean,adam,finalize' uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-3152k-checkpoint.npz --inverse-max-steps 0 --inverse-min-steps 0 --output-series data/20-inverse-face-3152k-final.vtu.series --checkpoint data/20-inverse-face-3152k-final-checkpoint.npz
```

## Setup

- Source mesh:
  `/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/41-expression-3152k.vtu`
- Cell selection: `InFaceConvex`
- Target displacement: `Expression000`
- Target point mask: `IsFace`
- Constitutive model: stable neo-Hookean, `nu = 0.49`
- Collision handling: none
- Fixed points: cranium and mandible
- Forward solver: PNCG, `rtol = 5e-4`, `atol = 0`, `max_steps = 10000`
- Adjoint solver: CG with MinRes fallback, `rtol = 5e-4`, `atol = 0`,
  `max_steps = 10000`

The prepared mesh has `225052` points, `1127541` tetrahedra, `17582` face
target points, and `283391` active muscle tetrahedra. The inverse parameter
count is `1700346`.

The material fractions were checked directly:

- fat/background: `1 - max(SmasFraction, MuscleFraction)`, `E = 1`
- muscle: `MuscleFraction`, `E = 100`, active
- SMAS-only: `max(SmasFraction - MuscleFraction, 0)`, `E = 100`, passive

All three fraction identities had max absolute difference `0.0`, and the sum
range was `0.9999999999999999` to `1.0`.

## Results

The long Adam run reached a best checkpoint at step `58`:

- best max `IsFace` point error: `0.17138371517410103 cm`
- Comet: `https://www.comet.com/liblaf/apple/8047b10f59b643cba772f95915fea744`
- optimization series: `data/20-inverse-face-3152k.vtu.series`
- frames in optimization series: `69`

The learning rate was intentionally large. It found a good basin quickly, then
the current iterate wandered upward after the best checkpoint. Because the
script reports and saves the best state, the stop condition was corrected to
use `best_max_error` for the tolerance stop.

The checkpoint verification run produced the final accepted artifacts:

- Comet: `https://www.comet.com/liblaf/apple/93f9113c7c3945818164f70a5ca7be3a`
- summary: `data/20-inverse-face-3152k-summary.json`
- result mesh: `data/20-inverse-face-3152k.vtu`
- snapshot: `data/20-inverse-face-3152k.png`
- final checkpoint: `data/20-inverse-face-3152k-final-checkpoint.npz`
- one-frame verification series: `data/20-inverse-face-3152k-final.vtu.series`

Final metrics:

- `passed`: `true`
- stop reason: `max_point_error_tol`
- max `IsFace` point error: `0.16911033770519793 cm`
- RMS `IsFace` point error: `0.04456696389586918 cm`
- mean `IsFace` point error: `0.03671051553123811 cm`
- forward failures: `0`
- adjoint failures: `0`
- forward max relative residual: `0.0004894121081861133`
- adjoint max relative residual: `0.0004112733191146554`

## Verification

Direct VTU checks confirmed:

- `data/20-inverse-face-3152k.vtu` keeps the same rest-shape points as the
  prepared input mesh.
- The final mesh has `Displacement` point data with shape `(225052, 3)`.
- The final mesh has recovered activation cell data with shape `(1127541, 6)`.
- Recomputed max error on `IsFace` points is
  `0.16911033770519793 cm`, matching the JSON summary.
- The optimization series has `69` frames, and the final verification series
  has `1` frame.

The final max point error is below the required `0.2 cm` threshold.

## Notes

The finalization run starts from the best checkpoint saved by the long Adam
run. It is not a separate inverse search; it is a clean Cherries/Comet run that
rebuilds the model, re-solves the forward equilibrium from the checkpointed
state, and writes the accepted result artifacts.

The PyVista snapshot emitted a headless X-server warning, but the PNG was
written successfully.
