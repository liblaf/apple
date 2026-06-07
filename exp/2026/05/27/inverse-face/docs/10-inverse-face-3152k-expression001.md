# Inverse Face 3152k Expression001

## Summary

The `20-` inverse physics experiment succeeded for the 3152k human-face `Expression001` target. The final run stopped on `max_point_error_tol` with target max error below the requested 0.2 cm threshold.

- Final Comet run: `https://www.comet.com/liblaf/apple/c26fc5b892984304ae92bfa17cd15df4`
- Final command:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME="inverse-face 3152k expression001 smas100 top000025 warm099 resume" CHERRIES_TAGS="inverse-face,3152k,expression001,smas100,in-face-convex,per-tet-activation-inv,resume,top-error-narrow,warm099" uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-3152k-top0001strong-step20-warm099.npz --inverse-lr 0.00075 --top-error-weight 1.0 --top-error-fraction 0.00025`
- Cherries commit: `acad8714`

## Inputs

- Source mesh: `/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu`
- Prepared input: `data/10-inverse-face-3152k-input.vtu`
- Prepared target: `data/10-inverse-face-3152k-target.vtu`
- Extracted domain: `InFaceConvex` tetrahedra only
- Target displacement: `Expression001`
- Fixed points: union of `IsCranium` and `IsMandible`

The prepared domain has 225,052 points and 1,127,541 tetrahedra. The inverse target mask selects 17,582 `IsFace` points, and 283,391 tetrahedra have active muscle parameters.

## Materials

All materials use stable neo-Hookean with `nu = 0.49`.

- Background/fat fraction: `1 - max(SmasFraction, MuscleFraction)`, `E = 1.0`
- Muscle fraction: `MuscleFraction`, active, `E = 100.0`
- SMAS-only fraction: `max(SmasFraction - MuscleFraction, 0)`, passive, `E = 100.0`

No collision potentials were used. Forward solves used PNCG with `rtol = 5e-4`, `atol = 0`, and `max_steps = 10000`.

## Result

- Passed: `true`
- Stop reason: `max_point_error_tol`
- Best inverse step: 9
- Optimizer steps: 9
- Target mean error: 0.034545 cm
- Target RMS error: 0.058432 cm
- Target max error: 0.197071 cm
- Target displacement RMS: 0.297261 cm
- Target displacement max: 1.251746 cm
- Forward failures: 0
- Adjoint failures: 0
- Max forward steps used in final run: 1547
- Max forward relative gradient: 4.966650e-4
- Max adjoint relative residual: 4.948046e-4
- Total final-run time: 564.11 s

## Artifacts

- Final mesh: `data/20-inverse-face-3152k.vtu`
- Final summary: `data/20-inverse-face-3152k-summary.json`
- Final snapshot: `data/20-inverse-face-3152k.png`
- Final time series: `data/20-inverse-face-3152k.vtu.series`
- Final time-series frames: `data/20-inverse-face-3152k.vtu.d/`, 10 VTU frames
- Final checkpoint: `data/20-inverse-face-3152k-checkpoint.npz`

## Notes

The inverse solve used the new forward/inverse library path through `Forward`, `ModelBuilder`, and `DifferentiableForward`.

Near the final tolerance, exact displacement checkpoint resumes could start PNCG with a nearly flat relative-gradient baseline because `forward_atol = 0`. The successful final continuation therefore used `data/20-inverse-face-3152k-top0001strong-step20-warm099.npz`, which kept the step-20 activation state and used a 0.99-scaled copy of the step-20 displacement. That kept the solve in the good forward branch while avoiding the flat warm-start line-search corner.

## Validation

- `jq` summary check confirmed `passed: true`, `target/error_max = 0.19707103311134558`, `forward/failures = 0`, and `adjoint/failures = 0`.
- `find data/20-inverse-face-3152k.vtu.d -name '*.vtu' | wc -l` returned 10 frames.
- `uv run ruff check exp/2026/05/27/inverse-face/src/10-prepare-inverse-face.py exp/2026/05/27/inverse-face/src/20-inverse-face.py`
- `uv run python -m py_compile exp/2026/05/27/inverse-face/src/10-prepare-inverse-face.py exp/2026/05/27/inverse-face/src/20-inverse-face.py`
