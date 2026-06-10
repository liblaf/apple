# Toy Target Magnitude Sweep

## Purpose

This sweep checks whether the inverse solution's volume change stays nearly fixed when the imposed top-surface target displacement changes in magnitude and sign. The underlying question is whether bumpy inverse results can be explained as a nearly incompressible, volume-preserving response to a forward-unreachable target.

The geometry, material split, active muscle patch, and fixed boundaries are the same as `20-toy-unreachable-inverse.md`. The sweep varies:

- target top-surface `y` displacement: `+/-0.005`, `+/-0.01`, `+/-0.02`, `+/-0.04`
- Poisson ratio: `nu = 0.49` and a softer control `nu = 0.30`
- resolution: coarse only

The lower-`nu` control is included because a target-magnitude sweep alone can show a volume-limited inverse branch, but it cannot identify high Poisson ratio as the cause.

## Command

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="toy target magnitude and poisson sweep" \
CHERRIES_TAGS="unreachable-inverse,toy,magnitude-sweep,poisson-control,nu049,nu030" \
uv run python src/30-toy-target-magnitude-sweep.py
```

Comet run: <https://www.comet.com/liblaf/apple/eaac9ece561241dba6c0400469904095>

## Outputs

- `data/30-toy-target-magnitude-sweep-summary.json`
- `data/30-toy-target-magnitude-sweep-cases.csv`
- `data/30-toy-target-magnitude-sweep-table.md`
- one `input.vtu`, `target.vtu`, inverse result `.vtu`, and `.vtu.series` per case

The final result `.vtu` stores the best inverse checkpoint found during the run. The `.vtu.series` files store optimizer snapshots every `20` steps through step `120`.

The current toy VTU writer now stores per-tetra volume diagnostics for ParaView:

- `VolumeInitial`, `VolumeTarget`, `VolumeInverse`
- `VolumeTargetRelChange`, `VolumeInverseRelChange`
- `SignedVolumeInitial`, `SignedVolumeTarget`, `SignedVolumeInverse`
- `SignedVolumeTargetRelChange`, `SignedVolumeInverseRelChange`

The completed sweep VTUs were also post-processed with those same arrays after the run. The `input.vtu` files are the only skipped VTUs because they have no displacement field.

## Results

Signed volume change is shown below. `inverse / target` is the signed inverse volume change divided by the signed target volume change, so positive values below `1` mean the inverse moved in the requested volume-change direction but with a smaller magnitude.

| nu | mode | target y | target dV | inverse dV | inverse / target | best error / target | top y std | best step | status |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.49 | stretch | 0.005 | 3.8281% | 1.1305% | 29.53% | 81.40% | 0.001743 | 108 | not converged, best in last window |
| 0.49 | squash | -0.005 | -3.8281% | -0.6565% | 17.15% | 86.80% | 0.001118 | 84 | drifted after best |
| 0.49 | stretch | 0.010 | 7.6562% | 2.5927% | 33.86% | 78.14% | 0.003637 | 90 | drifted after best |
| 0.49 | squash | -0.010 | -7.6562% | -1.2465% | 16.28% | 88.12% | 0.002443 | 119 | not converged, best in last window |
| 0.49 | stretch | 0.020 | 15.3125% | 4.7298% | 30.89% | 80.07% | 0.006933 | 87 | drifted after best |
| 0.49 | squash | -0.020 | -15.3125% | -2.5599% | 16.72% | 88.35% | 0.004919 | 109 | not converged, best in last window |
| 0.49 | stretch | 0.040 | 30.6250% | 8.1385% | 26.57% | 83.69% | 0.013211 | 119 | not converged, best in last window |
| 0.49 | squash | -0.040 | -30.6250% | -4.8250% | 15.75% | 88.64% | 0.009357 | 105 | not converged, best in last window |
| 0.30 | stretch | 0.005 | 3.8281% | 2.3082% | 60.30% | 56.00% | 0.001851 | 119 | not converged, best in last window |
| 0.30 | squash | -0.005 | -3.8281% | -2.1095% | 55.11% | 57.32% | 0.001700 | 67 | drifted after best |
| 0.30 | stretch | 0.010 | 7.6562% | 4.1334% | 53.99% | 60.72% | 0.003739 | 120 | not converged, best in last window |
| 0.30 | squash | -0.010 | -7.6562% | -3.5092% | 45.83% | 64.00% | 0.003268 | 118 | not converged, best in last window |
| 0.30 | stretch | 0.020 | 15.3125% | 6.1285% | 40.02% | 69.94% | 0.006875 | 103 | not converged, best in last window |
| 0.30 | squash | -0.020 | -15.3125% | -5.6730% | 37.05% | 71.17% | 0.006342 | 116 | not converged, best in last window |
| 0.30 | stretch | 0.040 | 30.6250% | 8.0690% | 26.35% | 79.86% | 0.011843 | 85 | plateaued |
| 0.30 | squash | -0.040 | -30.6250% | -7.9026% | 25.80% | 80.38% | 0.011972 | 120 | not converged, best in last window |

## Interpretation

For `nu = 0.49`, the inverse solution is strongly volume-limited. Stretch reaches only `26.6%` to `33.9%` of the requested signed volume increase, and squash reaches only `15.8%` to `17.1%` of the requested signed volume decrease. The squash ratios are especially stable across all four target magnitudes, while stretch is roughly stable but not constant.

The inverse response is not sign-symmetric at `nu = 0.49`. For the same displacement magnitude, stretch consistently permits more signed volume change than squash. The residual also remains large: the best RMS error is still `78%` to `89%` of the target displacement RMS.

Lowering Poisson ratio to `nu = 0.30` changes the behavior. The inverse solution captures much more of the target volume change for small and medium targets: about `60%` and `55%` at `0.005`, about `54%` and `46%` at `0.01`, and about `40%` and `37%` at `0.02`. The residual is also lower for those cases. This is evidence that high Poisson ratio is a real contributor to the volume-limited, hard-to-match inverse response.

The `0.04` cases remain difficult even at `nu = 0.30`, with inverse/target volume ratios near `26%` and best error still around `80%` of the target RMS. That points to additional limits from the fixed sides and bottom, the small active muscle patch, and the optimization landscape.

## Conclusion

The sweep supports the narrower conclusion that the inverse solution is volume-limited, especially at `nu = 0.49`, and that high Poisson ratio contributes to the bumpy unreachable inverse behavior.

It does not by itself prove that high Poisson ratio is the only cause. The inverse solve does not cleanly converge, and the low-`nu` control still struggles at large target magnitude. The safer conclusion is that high Poisson ratio, fixed-boundary volume constraints, limited actuation support, and optimizer nonconvergence all interact. The per-tetra VTUs should be used in ParaView to localize where the volume limitation and surface roughness coincide.
