# IsFace 1% Surface Prestrain on Expression000 Inverse Solutions

## Question

The previous 10% prestrain run used the wrong inverse artifacts. This run uses
the inverse solutions where the target displacement is `Expression000`, then
directly relaxes those inverse displacements with the recovered muscle
activation held fixed and a smaller 1% IsFace surface prestrain.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/20-surface-prestrain-expression000.py`
- Run directory:
  `exp/2026/06/10/surface-prestrain`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='surface IsFace 2D SNH 1pct prestrain Expression000' CHERRIES_TAGS='surface-prestrain,isface,Expression000,515k,3152k,direct-forward,nu049,stable-neo-hookean,2d,prestrain001' uv run python src/20-surface-prestrain-expression000.py`
- Comet:
  <https://www.comet.com/liblaf/apple/4a510195e2614ddda89a993e59d41c7d>
- 515k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-fresh-nu049-smas1-noclamp-super-loose-reg.vtu`
- 3152k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-3152k.vtu`
- Target displacement:
  `Expression000` stored as `Displacement` in the corresponding target VTUs.
- Surface selection:
  boundary triangles where all three original vertices have `IsFace = true`.
- Prestrain:
  `prestrain = 0.01`, tensile mode, prestrained length scale
  `1 / 1.01 = 0.9900990099009901`.

## Energy

The added surface term is a 2D Stable Neo-Hookean membrane energy in the 3D
embedding. For each selected triangle, let `G` be the deformed first
fundamental form and `G_p` be the prestrained reference metric:

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
volumetric material, but with 2D invariants. With `surface_stiffness = 1.0`,
the surface material has `lambda = 16.442953020134212` and
`mu = 0.33557046979865773`.

## Outputs

- `data/20-surface-prestrain-515k-expression000-smas1.vtu`
- `data/20-surface-prestrain-515k-expression000-smas1-surface.vtp`
- `data/20-surface-prestrain-3152k-expression000-smas100.vtu`
- `data/20-surface-prestrain-3152k-expression000-smas100-surface.vtp`
- `data/20-surface-prestrain-expression000-summary.json`
- `data/20-surface-prestrain-expression000-cases.csv`
- `data/20-surface-prestrain-expression000-table.md`

The volume VTUs contain `PreviousInverseDisplacement`,
`PrestrainDisplacement`, `PrestrainMinusPrevious`, `PreviousErrorNorm`,
`PrestrainErrorNorm`, and `PrestrainDeltaNorm`. The surface VTPs additionally
contain per-triangle `SurfacePrestrainTriangle`,
`SurfacePrestrainAreaRelChange`, `SurfacePrestrainAreaStretch`,
`SurfacePrestrainEnergyDensity`, and `SurfacePrestrainEnergy`.

## Results

| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 515k-expression000-smas1 | 6,787 | 13,079 | 0.0414332 | 0.189323 | +0.147889 | 0.0191753 | 0.0182607 | -0.000914652 | primary success, 8,485 steps |
| 3152k-expression000-smas100 | 17,582 | 34,372 | 0.0445670 | 0.0947698 | +0.0502028 | 0.0136435 | 0.0120072 | -0.00163629 | max steps, 10,000 steps |

Additional roughness checks:

| case | previous surface error-edge RMS | prestrain surface error-edge RMS | delta |
| --- | ---: | ---: | ---: |
| 515k-expression000-smas1 | 0.0284267 | 0.0342715 | +0.00584477 |
| 3152k-expression000-smas100 | 0.0213310 | 0.0212725 | -0.00005845 |

Forward solve residuals:

| case | relative grad norm | total time |
| --- | ---: | ---: |
| 515k-expression000-smas1 | 0.000498241 | 83.9 s |
| 3152k-expression000-smas100 | 0.000796043 | 305.4 s |

## Interpretation

The 1% surface prestrain gives a small reduction in the displacement Laplacian
RMS on both meshes:

- 515k: `0.0191753 -> 0.0182607`, about 4.8% lower
- 3152k: `0.0136435 -> 0.0120072`, about 12.0% lower

However, the relaxed displacement moves substantially away from the
`Expression000` target:

- 515k: target RMS `0.0414332 -> 0.189323`
- 3152k: target RMS `0.0445670 -> 0.0947698`

The 3152k direct forward relaxation also still failed the configured
convergence test, reaching the 10,000-step cap with relative gradient norm
`0.000796043` versus the requested `0.0005`.

For these parameters, 1% IsFace surface prestrain behaves like a surface
regularizing load: it smooths the displacement field a little, but it does not
preserve the target expression. It therefore does not provide evidence that the
bumpy inverse result can be repaired by simply applying this prestrain to the
previous inverse solution. A target-preserving variant would need either a much
weaker membrane term or an explicit target-displacement penalty/constraint
during the relaxation.

## Validation

- `uv run python -m py_compile src/20-surface-prestrain-expression000.py`
- Smoke run:
  `DEBUG=1 CHERRIES_NAME='surface prestrain Expression000 1pct 515k smoke' CHERRIES_TAGS='surface-prestrain,Expression000,stable-neo-hookean,2d,prestrain001,smoke,515k' uv run python src/20-surface-prestrain-expression000.py --cases '["515k-expression000-smas1"]' --forward-max-steps 2`
- Read back all four saved VTU/VTP files with PyVista.
