# Real Mesh Area / Volume Change

## Purpose

This run compares the prescribed target displacement and the recovered inverse solution on two existing human-face experiment artifacts:

- `3152k-expression001`: the completed 3152k `Expression001` inverse-face result.
- `515k-nosmas`: the 515k no-SMAS forward target and inverse recovery.

The question is whether the bumpy inverse results are consistent with a forward-unreachable target under nearly incompressible material settings (`nu = 0.49`). The check uses signed tetrahedron volume, absolute tetrahedron volume, boundary area, target-mask area, and target-mask displacement error.

## Command

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="unreachable inverse real mesh area volume signed" \
CHERRIES_TAGS="unreachable-inverse,area-volume,real-mesh,3152k,515k,signed-volume" \
uv run python src/10-real-mesh-area-volume.py
```

Comet run: <https://www.comet.com/liblaf/apple/843d1efdf792443a9b01f91538d25454>

## Inputs

| case | input | target | inverse | target mask |
| --- | --- | --- | --- | --- |
| `3152k-expression001` | `exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-input.vtu` | `exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-target.vtu` | `exp/2026/05/27/inverse-face/data/20-inverse-face-3152k.vtu` | `TargetSurfaceMask` |
| `515k-nosmas` | `exp/2026/05/27/forward-face/data/10-forward-face-515k-nosmas-input.vtu` | `exp/2026/05/27/forward-face/data/20-forward-face-515k-nosmas.vtu` | `exp/2026/05/27/forward-face/data/30-inverse-face-515k-nosmas.vtu` | `IsFace` |

## Outputs

- `data/10-real-mesh-area-volume-summary.json`
- `data/10-real-mesh-area-volume.csv`
- `data/10-real-mesh-area-volume-table.md`

## Results

| case | state | signed volume change | abs volume change | inverted tets | boundary area change | mask area change | mask disp RMS | mask error RMS | error / target RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3152k-expression001` | target | -2.8628% | -0.1208% | 2.0379% | 0.6330% | -0.3666% | 0.297261 | 0.055065 | 18.5240% |
| `3152k-expression001` | inverse | -0.7307% | -0.7298% | 0.0114% | 0.6933% | 0.8127% | 0.293182 | 0.055065 | 18.5240% |
| `515k-nosmas` | target | -0.1199% | -0.1199% | 0.0000% | 0.0186% | 0.0474% | 0.071006 | 0.016721 | 23.5484% |
| `515k-nosmas` | inverse | -0.1257% | -0.1257% | 0.0000% | 0.0120% | 0.0556% | 0.067391 | 0.016721 | 23.5484% |

## Interpretation

The 3152k `Expression001` target is not volume-neutral in the signed sense. It asks for a total signed volume decrease of `-2.8628%`, while the absolute-volume total changes by only `-0.1208%` and `2.0379%` of tetrahedra invert. That gap means part of the target displacement is a local orientation or folding problem, not just a smooth near-incompressible deformation. The inverse solution stays much closer to a physically admissible state: signed volume change `-0.7307%`, absolute volume change `-0.7298%`, and only `0.0114%` inverted tetrahedra.

This supports the hypothesis that the 3152k bumpy inverse result is at least partly caused by a forward-unreachable target. With `nu = 0.49`, the forward solve strongly resists large local volume and orientation changes, so an optimizer can reduce point error while still being unable to reproduce the target's local signed-volume pathology.

The 515k no-SMAS case does not show the same evidence. Target and inverse signed-volume changes are nearly identical (`-0.1199%` vs `-0.1257%`), there are no inverted tetrahedra, and the mask-area changes are both small. That case still has nonzero inverse error, but this analysis does not support a global volume-unreachability explanation for it. The remaining error there is more likely tied to the inverse activation parameterization, objective, regularization, or the limited actuation family.

## Notes

The signed-volume metric is essential here. Absolute volume alone makes the 3152k target look nearly volume preserving, but signed volume reveals that the apparent preservation is mixed with local inversion.
