# Toy Unreachable Inverse Physics

## Purpose

This experiment builds a controlled toy problem to test whether a nearly incompressible forward model (`nu = 0.49`) produces bumpy inverse behavior when the target displacement is not reachable by the available active muscle region.

The geometry is a thin box:

- full body: `box(0, 1, 0, 0.1, 0, 1)`
- SMAS layer: `box(0, 1, 0.04, 0.06, 0, 1)`
- muscle region: `box(0, 0.5, 0.04, 0.06, 0.4, 0.6)`

Boundary conditions fix the bottom surface and the four side surfaces. The target surface is the free interior of the top surface, excluding fixed side points. Two target modes were run:

- stretch: top target moves by `+0.02` in `y`
- squash: top target moves by `-0.02` in `y`

Material fractions per tetrahedron follow the requested split:

- muscle fraction: active muscle support
- aponeurosis fraction: `max(0, smas - muscle)`
- fat fraction: `1 - aponeurosis - muscle`

Material properties:

- muscle: `E = 1e2`, `nu = 0.49`, with activation
- aponeurosis: `E = 1e2`, `nu = 0.49`, no activation
- fat: `E = 1`, `nu = 0.49`

## Command

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="unreachable inverse toy convergence diagnostics" \
CHERRIES_TAGS="unreachable-inverse,toy,stretch,squash,resolution-sweep,nu049,convergence" \
uv run python src/20-toy-unreachable-inverse.py
```

Comet run: <https://www.comet.com/liblaf/apple/b099d6b601f6496c8b586cc951e7a6b7>

## Outputs

- `data/20-toy-unreachable-inverse-summary.json`
- `data/20-toy-unreachable-inverse-cases.csv`
- `data/20-toy-unreachable-inverse-table.md`
- one `input.vtu`, `target.vtu`, inverse result `.vtu`, and `.vtu.series` for each stretch/squash and resolution case
- each inverse series has `25` frames, from step `0` through step `120` with `series_stride = 5`

The final result `.vtu` for each case stores the best inverse checkpoint, not necessarily the last optimizer iterate. The JSON summary keeps both `best/*` and `final_step/*` metrics.

## Results

| case | points | tets | active tets | best step | convergence | target signed volume change | inverse signed volume change | target inverted tets | inverse inverted tets | best error RMS | best error / target RMS | top y std | top edge RMS |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20-toy-stretch-coarse` | 567 | 2304 | 96 | 111 | not converged, best in last window | 15.3125% | 5.1117% | 0.0000% | 0.9983% | 0.015658 | 78.2894% | 0.007268 | 0.005753 |
| `20-toy-squash-coarse` | 567 | 2304 | 96 | 120 | not converged, best in last window | -15.3125% | -2.5799% | 12.7604% | 0.1736% | 0.017662 | 88.3114% | 0.004710 | 0.003787 |
| `20-toy-stretch-medium` | 2475 | 11760 | 224 | 76 | drifted after best | 17.2449% | 4.0500% | 0.0000% | 0.4422% | 0.017263 | 86.3173% | 0.006826 | 0.005179 |
| `20-toy-squash-medium` | 2475 | 11760 | 224 | 106 | not converged, best in last window | -17.2449% | -2.6361% | 8.6224% | 0.3571% | 0.017936 | 89.6796% | 0.005427 | 0.003954 |
| `20-toy-stretch-fine` | 4851 | 24000 | 480 | 117 | not converged, best in last window | 18.0500% | 4.9897% | 0.0000% | 0.2500% | 0.016967 | 84.8348% | 0.007509 | 0.004936 |
| `20-toy-squash-fine` | 4851 | 24000 | 480 | 120 | not converged, best in last window | -18.0500% | -2.7051% | 9.0250% | 0.3083% | 0.017936 | 89.6805% | 0.005241 | 0.003342 |

## Convergence

The inverse solve does not cleanly converge in this rerun. Five of six cases find their best state inside the last 20-step window, and the remaining case drifts after an earlier best checkpoint. The final iterate is worse than the best checkpoint in four cases:

| case | best step | final RMS | best RMS | final loss / best loss | last-window relative improvement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `20-toy-stretch-coarse` | 111 | 0.016930 | 0.015658 | 1.1675 | 9.9083% |
| `20-toy-squash-coarse` | 120 | 0.017662 | 0.017662 | 1.0000 | 1.4880% |
| `20-toy-stretch-medium` | 76 | 0.017404 | 0.017263 | 1.0193 | 4.0142% |
| `20-toy-squash-medium` | 106 | 0.018270 | 0.017936 | 1.0382 | 0.4605% |
| `20-toy-stretch-fine` | 117 | 0.017167 | 0.016967 | 1.0237 | 2.5327% |
| `20-toy-squash-fine` | 120 | 0.017936 | 0.017936 | 1.0000 | 11.6223% |

The previous 35-step run was too short to separate unreachable physics from optimizer progress. The 120-step rerun still leaves large residuals and unstable best-step timing, so the current conclusion is narrower: the target is physically difficult or unreachable for this actuation family, and the inverse optimizer has not settled to a clean stationary solution.

## Interpretation

The toy targets deliberately prescribe large signed-volume changes on a body whose sides and bottom are fixed. Stretch demands a `+15%` to `+18%` signed-volume increase; squash demands a `-15%` to `-18%` signed-volume decrease. The squash target also creates local inversions in `8.6%` to `12.8%` of tetrahedra.

The inverse solutions do not reproduce those volume changes. Across resolutions, stretch reaches only about `+4.0%` to `+5.1%` signed volume change, and squash reaches only about `-2.6%` to `-2.7%`. The best residual remains large: `78.3%` to `89.7%` of the target displacement RMS.

The behavior persists from the coarse mesh through the fine mesh, so it is not just a coarse-discretization artifact. The recovered top surface is also spatially uneven: top `y` standard deviation is roughly `0.0047` to `0.0075`, and top-edge RMS roughness is roughly `0.0033` to `0.0058`. Those values are a direct quantitative proxy for the bumpy inverse surface.

The signed-volume rerun matters for squash. Absolute volume can hide orientation loss because inverted tetrahedra still contribute positive absolute volume. Signed volume shows that the squash target is strongly compressive and locally inverted, while the inverse solution stays much closer to the physically reachable branch.

## Conclusion

This toy setup reproduces the intended failure mode: with `nu = 0.49`, fixed sides/bottom, and a small active muscle patch, the target top motion is largely unreachable. The inverse optimizer responds with high residual error and spatially uneven displacement rather than a smooth exact match.

The convergence diagnostics also show that the inverse solve itself is not fully settled. For report purposes, the saved best checkpoint is the right artifact to inspect, and the nonconvergence should be treated as part of the result rather than hidden behind the final iteration.
