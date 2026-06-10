# IsFace 2D Stable Neo-Hookean Surface Prestrain

## Question

The previous 515k and 3152k face inverse runs had bumpy inverse
displacements. This experiment adds a 2D Stable Neo-Hookean elastic energy on
surface triangles in the `IsFace` region, then directly relaxes the previous
inverse displacement with the previous recovered muscle activation held fixed.
This avoids a new full inverse solve.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/10-surface-prestrain-forward.py`
- Run directory:
  `exp/2026/06/10/surface-prestrain`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='surface IsFace 2D Stable Neo-Hookean prestrain' CHERRIES_TAGS='surface-prestrain,isface,515k,3152k,direct-forward,nu049,stable-neo-hookean,2d' uv run python src/10-surface-prestrain-forward.py`
- Comet:
  <https://www.comet.com/liblaf/apple/1b367bda450f42559958a53da14117d3>
- Volumetric material:
  `E = 1`, `nu = 0.49`, previous recovered muscle activation held fixed.
- Surface material:
  2D Stable Neo-Hookean with `surface_stiffness = 1.0`,
  `lambda = 16.442953020134212`, `mu = 0.33557046979865773`.
- Surface selection:
  boundary triangles where all three original vertices have `IsFace = true`.
- Prestrain convention:
  `prestrain = 0.10`, tensile mode, length scale `1 / 1.1 =
  0.9090909090909091`. The original mesh-rest surface is therefore considered
  stretched by 10%, and zero surface energy occurs when the in-plane lengths
  shrink to `1 / 1.1` of the mesh-rest metric.

## Energy

For each selected surface triangle, the implementation uses the 3D embedding of
a 2D membrane. Let `G` be the deformed first fundamental form and let
`G_p^{-1}` be the inverse prestrained reference metric:

```text
G_p = alpha^2 G_0
alpha = 1 / (1 + prestrain)
C = G_p^{-1} G
I1 = trace(C)
J = sqrt(det(C))
psi = 0.5 * mu * (I1 - 2) - mu * (J - 1) + 0.5 * lambda * (J - 1)^2
E_surface = sum(rest_area * psi)
```

This matches the polynomial Stable Neo-Hookean form used by the repo's
volumetric material, but with 2D invariants.

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
`SurfacePrestrainI1`, `SurfacePrestrainAreaStretch`,
`SurfacePrestrainEnergyDensity`, and `SurfacePrestrainEnergy`.

## Results

| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 515k-nosmas | 6,787 | 13,079 | 0.0167208 | 0.464083 | +0.447362 | 0.00656959 | 0.0150747 | +0.0085051 | max steps, 10,000 steps |
| 3152k-expression001 | 17,582 | 34,372 | 0.0550647 | 0.332058 | +0.276994 | 0.0125797 | 0.0112971 | -0.00128255 | max steps, 10,000 steps |

Additional roughness checks:

| case | previous surface error-edge RMS | prestrain surface error-edge RMS | delta |
| --- | ---: | ---: | ---: |
| 515k-nosmas | 0.00272571 | 0.0537561 | +0.0510304 |
| 3152k-expression001 | 0.0220337 | 0.0351205 | +0.0130868 |

Forward solve residuals:

| case | relative grad norm | total time |
| --- | ---: | ---: |
| 515k-nosmas | 0.000700604 | 91.8 s |
| 3152k-expression001 | 0.0134884 | 308.1 s |

## Interpretation

The 2D Stable Neo-Hookean surface prestrain is much stronger than the earlier
quadratic metric penalty. With 10% tensile prestrain and `surface_stiffness =
1.0`, both direct-forward solves hit the 10,000-step cap and both moved far away
from the target expression:

- 515k: target RMS `0.0167208 -> 0.464083`
- 3152k: target RMS `0.0550647 -> 0.332058`

It also does not consistently reduce the roughness proxy. The 515k displacement
Laplacian RMS increases by about 129%, and the 3152k Laplacian RMS decreases by
only about 10% while target error becomes much worse. Surface error-edge RMS
increases on both meshes.

For this parameter choice, a 10% 2D Stable Neo-Hookean IsFace prestrain is not a
useful direct correction for the previous bumpy inverse result. It acts as a
strong additional surface load and overwhelms the previous inverse equilibrium.
If we continue this direction, the next experiment should sweep much smaller
surface stiffness or prestrain magnitude before drawing a visual conclusion.

## Validation

- `uv run python -m py_compile src/10-surface-prestrain-forward.py`
- Read back all four saved VTU/VTP files with PyVista.
