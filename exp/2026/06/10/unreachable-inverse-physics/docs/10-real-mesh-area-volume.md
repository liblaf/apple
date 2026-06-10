# Real Mesh Area / Volume Change

## Purpose

This run compares the prescribed target displacement and the recovered inverse solution on two existing human-face experiment artifacts:

- `3152k-expression001`: the completed 3152k `Expression001` inverse-face result.
- `515k-nosmas`: the 515k no-SMAS forward target and inverse recovery.

The important correction in this rerun is that the 3152k expression target is only valid on the `IsFace` points. The raw target displacement has nonzero values outside that area, but those values are not physical target data. For that reason, the 3152k target volume row below is a face-only diagnostic, not a physical whole-volume deformation.

The check now writes per-tetra volume arrays to VTU files for ParaView inspection.

## Command

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="unreachable inverse real mesh volume diagnostics vtu clean" \
CHERRIES_TAGS="unreachable-inverse,area-volume,real-mesh,diagnostic-vtu,isface,clean" \
uv run python src/10-real-mesh-area-volume.py
```

Comet run: <https://www.comet.com/liblaf/apple/6c98e9482a464bde984aff96ce1cc4b4>

## Inputs

| case | input | target | inverse | target mask |
| --- | --- | --- | --- | --- |
| `3152k-expression001` | `exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-input.vtu` | `exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-target.vtu` | `exp/2026/05/27/inverse-face/data/20-inverse-face-3152k.vtu` | `IsFace` |
| `515k-nosmas` | `exp/2026/05/27/forward-face/data/10-forward-face-515k-nosmas-input.vtu` | `exp/2026/05/27/forward-face/data/20-forward-face-515k-nosmas.vtu` | `exp/2026/05/27/forward-face/data/30-inverse-face-515k-nosmas.vtu` | `IsFace` |

## Outputs

- `data/10-real-mesh-area-volume-summary.json`
- `data/10-real-mesh-area-volume.csv`
- `data/10-real-mesh-area-volume-table.md`
- `data/10-real-mesh-area-volume-vtu/3152k-expression001-volume-change.vtu`
- `data/10-real-mesh-area-volume-vtu/515k-nosmas-volume-change.vtu`

The diagnostic VTUs contain point arrays for target validity and displacement comparison:

- `TargetValidPoint`
- `TargetRawDisplacement`
- `TargetFaceOnlyDisplacement`
- `InverseDisplacement`
- `InverseMinusTargetOnValidPoints`

They also contain per-tetra volume arrays for ParaView:

- `RestSignedVolume`, `RestAbsVolume`
- `TargetRawSignedVolumeRelChange`, `TargetRawAbsVolumeRelChange`, `TargetRawInverted`
- `TargetFaceOnlySignedVolumeRelChange`, `TargetFaceOnlyAbsVolumeRelChange`, `TargetFaceOnlyInverted`
- `InverseSignedVolumeRelChange`, `InverseAbsVolumeRelChange`, `InverseInverted`
- `TargetValidPointCount`, `TargetValidTetAny`, `TargetValidTetAll`, `TargetValidTetFraction`

## Results

| case | state | volume scope | physical volume? | signed volume change | abs volume change | inverted tets | mask area change | mask disp RMS | mask error RMS | error / target RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `3152k-expression001` | target-face-only-diagnostic | `IsFace-only` | no | -1.9749% | -1.0034% | 1.0034% | -0.3666% | 0.297261 | 0.055065 | 18.5240% |
| `3152k-expression001` | inverse | forward-solved full field | yes | -0.7307% | -0.7298% | 0.0114% | 0.8127% | 0.293182 | 0.055065 | 18.5240% |
| `515k-nosmas` | target | full field | yes | -0.1199% | -0.1199% | 0.0000% | 0.0474% | 0.071006 | 0.016721 | 23.5484% |
| `515k-nosmas` | inverse | forward-solved full field | yes | -0.1257% | -0.1257% | 0.0000% | 0.0556% | 0.067391 | 0.016721 | 23.5484% |

## Interpretation

The old full-volume target computation for `3152k-expression001` was not valid. The target file contains `5090` nonzero raw-displacement points outside `IsFace` with outside-face RMS `0.113695` and max magnitude `1.253522`; those values should not be used to infer whole-volume target deformation.

The corrected 3152k target row asks a narrower diagnostic question: what per-tetra volume change would result if the target displacement were applied only on the valid face points and zero elsewhere? That field is useful for ParaView localization, but it is not a physically solved target volume. It shows local face-adjacent distortion (`1.0034%` inverted tetrahedra in the diagnostic field), while the inverse solution remains on a much more physical branch (`0.0114%` inverted tetrahedra).

The direct evidence for the 3152k inverse mismatch is therefore the `IsFace` point error plus the inverse solution's physical volume field, not a whole-body target volume change. The inverse reaches a similar target-mask displacement RMS (`0.293182` vs `0.297261`) but still has `18.5240%` RMS error relative to the target displacement. The new VTU should make it possible to inspect whether the residual and the local volume changes line up with the bumpy areas in ParaView.

The 515k no-SMAS case remains a physical full-field comparison because the target is a forward-solved deformation. Target and inverse signed-volume changes are nearly identical (`-0.1199%` vs `-0.1257%`), there are no inverted tetrahedra, and the mask-area changes are both small. That case still has nonzero inverse error, but this analysis does not support a global volume-unreachability explanation for it.

## Notes

Use the per-tetra arrays in `data/10-real-mesh-area-volume-vtu/*.vtu` for visual diagnosis. For 3152k, prefer `TargetFaceOnly*` arrays when inspecting the target diagnostic and `Inverse*` arrays when inspecting the actual physical inverse result. `TargetRaw*` arrays are included only to expose why the old all-domain target-volume computation was misleading.
