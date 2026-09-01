# 2-D pork target-direction follow-up

## Result

The matched upward and downward Stable-Neo-Hookean/L2 runs use the same `100 x 10` band-muscle model, 1,200 Adam updates, then strict fixed-seed unbounded L-BFGS refinement. Neither endpoint passes physical stationarity, so this compares recorded horizon-limited paths, not local optima or reachability.

The target is `u_x=0`, `u_y=h 4 x (1-x)`. At the centre it requests top positions `.150` (`h=+.05`) and `.050` (`h=-.05`).

| diagnostic | upward | downward |
| --- | ---: | ---: |
| frames / evaluations | 1,245 | 1,211 |
| accepted L-BFGS / strict evaluations | 43 / 504 | 9 / 168 |
| physical stationarity | no | no |
| centre displacement / requested | `+.050657 / +.050000` | `-.019244 / -.050000` |
| achieved requested magnitude | 101.3% | 38.5% |
| target RMS | .001559 | .020120 |
| high-pass RMS | .001202 | .001831 |
| curvature RMS | 9.975 | 13.714 |
| activation RMS / jump | .3372 / .2564 | 1.1062 / .5071 |
| final min `det F / det G / det Ainv` | -11.045 / -.0940 / -.7569 | .0729 / .0729 / .4176 |
| strict residual RMS | `9.995e-11` | `5.061e-11` |
| gradient inf / RMS | `3.078e-5 / 1.630e-6` | `5.836e-5 / 3.521e-6` |

The downward endpoint reaches only 38.5% of its requested centre motion and has about 12.9 times the upward target RMS. Its larger activation and roughness are consistent with a sign-asymmetric fixed-boundary nonlinear problem, but the experiment does not isolate a mechanical cause: branch selection, unbounded per-cell control, and nonconvex optimization remain confounded.

The upward endpoint remains determinant-inverted. The downward endpoint is orientation-positive at the end but transiently inverts (steps 29--105). Determinants are diagnostics, never constraints.

## Exact downward evolution

The [downward ParaView movie](../data/60-paraview-2d-target-down/2d__height-down/evolution.mp4) uses all 1,211 recorded source states at 30 FPS with no interpolation or duplication. Its [render receipt](../data/60-paraview-2d-target-down/render-receipt.json) records 1,211 source steps, PNGs, and H.264 frames (40.367 seconds).

## Reproduction

```bash
uv run python src/10-run-pork-2d.py \
  --cases 'height-down:stable:l2:100x10:-.05' \
  --output-dir data/60-pork-2d-target-down-reproduced \
  --max-steps 1200 --require-inverse-convergence false \
  --validate-derivatives true

pvpython src/40-render-pork-paraview.py \
  --input-root data/60-pork-2d-target-down-reproduced \
  --output-root data/60-paraview-2d-target-down-reproduced
```

This runs 1,200 Adam updates then strict fixed-seed, objective-normalized unbounded L-BFGS refinement. `require_inverse_convergence false` retains a horizon-limited case and full receipt rather than mislabeling a flat tail.
