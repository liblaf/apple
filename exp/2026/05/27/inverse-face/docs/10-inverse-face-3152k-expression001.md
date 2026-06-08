# Inverse Face 3152k Expression001

## Summary

The `20-` inverse physics experiment succeeded for the 3152k human-face `Expression001` target. After the first pass crossed the requested 0.2 cm max-error threshold, the stop rule was relaxed with a post-success patience window so the optimizer could keep improving instead of stopping at the first acceptable point.

- Final Comet run: `https://www.comet.com/liblaf/apple/7c9901d8a3724fa7a57d3a6658d96a65`
- Final command:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME="inverse-face 3152k expression001 continue past threshold" CHERRIES_TAGS="inverse-face,3152k,expression001,smas100,in-face-convex,per-tet-activation-inv,resume,post-success-patience" uv run python src/20-inverse-face.py --initial-activation-inv data/20-inverse-face-3152k-step9-warm099.npz --inverse-lr 0.0005 --top-error-weight 1.0 --top-error-fraction 0.00025 --inverse-max-steps 120`
- Cherries commit: `eecbfc73`

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
- Stop reason: `step_safety_limit`
- Best inverse step: 116
- Optimizer steps: 120
- Target mean error: 0.032697 cm
- Target RMS error: 0.055065 cm
- Target max error: 0.168029 cm
- Target displacement RMS: 0.297261 cm
- Target displacement max: 1.251746 cm
- Forward failures: 0
- Adjoint failures: 0
- Max forward steps used in final run: 4133
- Max forward relative gradient: 4.997228e-4
- Max adjoint relative residual: 4.997726e-4
- Total final-run time: 6935.55 s

The previous acceptable run stopped at step 9 with max error 0.197071 cm. The continuation improved that to 0.168029 cm, a 14.74% reduction in the max face error, while keeping both forward and adjoint failures at 0.

## Artifacts

- Final mesh: `data/20-inverse-face-3152k.vtu`
- Final summary: `data/20-inverse-face-3152k-summary.json`
- Final snapshot: `data/20-inverse-face-3152k.png`
- Final time series: `data/20-inverse-face-3152k.vtu.series`
- Final time-series frames: `data/20-inverse-face-3152k.vtu.d/`, 121 VTU frames
- Final checkpoint: `data/20-inverse-face-3152k-checkpoint.npz`
- Warm-start checkpoint: `data/20-inverse-face-3152k-step9-warm099.npz`

## Notes

The inverse solve used the new forward/inverse library path through `Forward`, `ModelBuilder`, and `DifferentiableForward`.

The first successful pass demonstrated feasibility, but stopping immediately on `max_point_error_cm = 0.2` left clear improvement available. `src/20-inverse-face.py` now keeps the threshold as the pass criterion, then requires `post_success_patience = 20` non-improving steps before the threshold stop can fire. This run reached the `--inverse-max-steps 120` cap before that patience window was exhausted after the late step-116 improvement.

Near the final tolerance, exact displacement checkpoint resumes could start PNCG with a nearly flat relative-gradient baseline because `forward_atol = 0`. The successful final continuation therefore used `data/20-inverse-face-3152k-step9-warm099.npz`, which kept the step-9 activation state and used a 0.99-scaled copy of the step-9 displacement. That kept the solve in the good forward branch while avoiding the flat warm-start line-search corner.

## Validation

- `jq` summary check confirmed `passed: true`, `stop_reason = step_safety_limit`, `best/step = 116`, `target/error_max = 0.16802884701296472`, `target/error_rms = 0.05506470541986396`, `forward/failures = 0`, and `adjoint/failures = 0`.
- `find data/20-inverse-face-3152k.vtu.d -name '*.vtu' | wc -l` returned 121 frames.
- `jq` trace check confirmed the final trace had `stagnation/post_success_patience = 20`, `stagnation/no_improve_steps = 4`, and max observed no-improve streak 11 before later improvements.
- `uv run ruff check exp/2026/05/27/inverse-face/src/20-inverse-face.py`
- `uv run python -m py_compile exp/2026/05/27/inverse-face/src/20-inverse-face.py`
