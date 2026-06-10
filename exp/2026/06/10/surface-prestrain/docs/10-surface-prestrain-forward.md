# IsFace Surface Prestrain Direct Forward

## Question

The previous 515k and 3152k face inverse runs had bumpy inverse displacements.
This experiment adds a custom elastic energy on surface triangles in the
`IsFace` region and directly relaxes the previous inverse displacement with the
previous recovered muscle activation. This avoids a new full inverse solve.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/10-surface-prestrain-forward.py`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='surface IsFace prestrain direct forward' CHERRIES_TAGS='surface-prestrain,isface,515k,3152k,direct-forward,nu049' uv run python src/10-surface-prestrain-forward.py`
- Comet:
  <https://www.comet.com/liblaf/apple/ad533de825d8487487df4d93423ad130>
- Material solve:
  `E = 1`, `nu = 0.49`, previous recovered muscle activation held fixed.
- Surface term:
  IsFace boundary triangles only, selected by all three original vertices having
  `IsFace = true`.
- Prestrain convention:
  `prestrain = 0.10`, `surface_stiffness = 1.0`, preferred surface metric scale
  `(1 / 1.1)^2 = 0.826446...`. In other words, the current rest surface is
  treated as 10% longer than the preferred surface metric.

## Outputs

- `data/10-surface-prestrain-515k-nosmas.vtu`
- `data/10-surface-prestrain-515k-nosmas-surface.vtp`
- `data/10-surface-prestrain-3152k-expression001.vtu`
- `data/10-surface-prestrain-3152k-expression001-surface.vtp`
- `data/10-surface-prestrain-forward-summary.json`
- `data/10-surface-prestrain-forward-cases.csv`
- `data/10-surface-prestrain-forward-table.md`

The volume VTUs contain `PreviousInverseDisplacement`, `PrestrainDisplacement`,
`PrestrainMinusPrevious`, `PreviousErrorNorm`, `PrestrainErrorNorm`, and
`PrestrainDeltaNorm`. The surface VTPs additionally contain per-triangle
`SurfacePrestrainTriangle`, `SurfacePrestrainRestArea`,
`SurfacePrestrainDeformedArea`, `SurfacePrestrainAreaRelChange`,
`SurfacePrestrainEnergyDensity`, and `SurfacePrestrainEnergy`.

## Results

| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 515k-nosmas | 6,787 | 13,079 | 0.0167208 | 0.129417 | +0.112696 | 0.00656959 | 0.00602888 | -0.000540702 | converged, 6,287 steps |
| 3152k-expression001 | 17,582 | 34,372 | 0.0550647 | 0.0972007 | +0.042136 | 0.0125797 | 0.00697384 | -0.00560584 | max steps, 10,000 steps |

Additional roughness checks:

| case | previous surface error-edge RMS | prestrain surface error-edge RMS | delta |
| --- | ---: | ---: | ---: |
| 515k-nosmas | 0.00272571 | 0.0203924 | +0.0176667 |
| 3152k-expression001 | 0.0220337 | 0.0157856 | -0.00624811 |

## Interpretation

The prestrain membrane does reduce the displacement Laplacian roughness proxy:
about 8% on 515k and about 45% on 3152k. So the custom surface triangle energy
is active and it does apply a smoothing/tension-like effect to the IsFace
surface.

However, it does not reduce the bumpy inverse result while preserving the target
fit. The target RMS error becomes much worse on both meshes:

- 515k: `0.0167208 -> 0.129417`
- 3152k: `0.0550647 -> 0.0972007`

The 515k case is especially clear because the forward solve converged. The 3152k
case hit the 10,000-step cap, but its relative gradient norm still dropped to
`0.004699`, and the same direction is visible: smoother surface displacement,
worse target agreement.

For this parameter choice, 10% IsFace surface prestrain is best understood as a
strong direct relaxation/smoothing load, not as a fix for the previous inverse
bumps. It may visually reduce high-frequency displacement on the face surface,
but it also moves the solution away from the expression target. I would not
conclude from this run that the bumpy result is caused by missing surface
prestrain. A more useful next sweep would vary `surface_stiffness` and prestrain
magnitude, then compare target error against roughness to find whether there is
a small regularizing regime before the target fit collapses.

## Validation

- `uv run python -m py_compile src/10-surface-prestrain-forward.py`
- Read back all four saved VTU/VTP files with PyVista.
