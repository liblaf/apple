# IsFace 4% Surface Prestrain on Expression000 Inverse Solutions

## Question

Repeat the corrected `Expression000` surface-prestrain experiment with a
slightly larger prestrain, `4%`, while preserving the physical 2D Stable
Neo-Hookean membrane energy.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/21-surface-prestrain-expression000-prestrain04.py`
- Run directory:
  `exp/2026/06/10/surface-prestrain`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='surface IsFace 2D SNH 4pct prestrain Expression000' CHERRIES_TAGS='surface-prestrain,isface,Expression000,515k,3152k,direct-forward,nu049,stable-neo-hookean,2d,prestrain004' uv run python src/21-surface-prestrain-expression000-prestrain04.py`
- Comet:
  <https://www.comet.com/liblaf/apple/679567a5bd0b496da696512c930716a1>
- 515k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-fresh-nu049-smas1-noclamp-super-loose-reg.vtu`
- 3152k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-3152k.vtu`
- Target displacement:
  `Expression000` stored as `Displacement` in the corresponding target VTUs.
- Surface selection:
  boundary triangles where all three original vertices have `IsFace = true`.
- Prestrain:
  `prestrain = 0.04`, tensile mode, prestrained length scale
  `1 / 1.04 = 0.9615384615384615`.

## Energy

This run keeps the same 2D Stable Neo-Hookean membrane energy as the 1% run.
For each selected triangle, let `G` be the deformed first fundamental form and
`G_p` be the prestrained reference metric:

```text
G_p = alpha^2 G_0
alpha = 1 / (1 + prestrain)
C = G_p^{-1} G
I1 = trace(C)
J = sqrt(det(C))
psi = 0.5 * mu * (I1 - 2) - mu * (J - 1) + 0.5 * lambda * (J - 1)^2
E_surface = sum(rest_area * psi)
```

With `surface_stiffness = 1.0`, the surface material has
`lambda = 16.442953020134212` and `mu = 0.33557046979865773`.

## Outputs

- `data/21-surface-prestrain-expression000-prestrain04-515k-expression000-smas1.vtu`
- `data/21-surface-prestrain-expression000-prestrain04-515k-expression000-smas1-surface.vtp`
- `data/21-surface-prestrain-expression000-prestrain04-3152k-expression000-smas100.vtu`
- `data/21-surface-prestrain-expression000-prestrain04-3152k-expression000-smas100-surface.vtp`
- `data/21-surface-prestrain-expression000-prestrain04-summary.json`
- `data/21-surface-prestrain-expression000-prestrain04-cases.csv`
- `data/21-surface-prestrain-expression000-prestrain04-table.md`

The volume VTUs contain the target, previous inverse, and prestrain-relaxed
displacements plus pointwise error/delta norms. The surface VTPs additionally
contain per-triangle Stable Neo-Hookean diagnostics including
`SurfacePrestrainAreaRelChange`, `SurfacePrestrainAreaStretch`,
`SurfacePrestrainEnergyDensity`, and `SurfacePrestrainEnergy`.

## Results

| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 515k-expression000-smas1 | 6,787 | 13,079 | 0.0414332 | 0.274238 | +0.232805 | 0.0191753 | 0.0144040 | -0.00477128 | max steps, 10,000 steps |
| 3152k-expression000-smas100 | 17,582 | 34,372 | 0.0445670 | 0.148697 | +0.104130 | 0.0136435 | 0.00807453 | -0.00556895 | max steps, 10,000 steps |

Additional roughness checks:

| case | previous surface error-edge RMS | prestrain surface error-edge RMS | delta |
| --- | ---: | ---: | ---: |
| 515k-expression000-smas1 | 0.0284267 | 0.0358544 | +0.00742769 |
| 3152k-expression000-smas100 | 0.0213310 | 0.0199538 | -0.00137720 |

Forward solve residuals:

| case | relative grad norm | total time |
| --- | ---: | ---: |
| 515k-expression000-smas1 | 0.000642601 | 100.1 s |
| 3152k-expression000-smas100 | 0.00201572 | 307.0 s |

## Comparison With 1%

| case | 1% target RMS | 4% target RMS | 1% lap RMS | 4% lap RMS |
| --- | ---: | ---: | ---: | ---: |
| 515k-expression000-smas1 | 0.189323 | 0.274238 | 0.0182607 | 0.0144040 |
| 3152k-expression000-smas100 | 0.0947698 | 0.148697 | 0.0120072 | 0.00807453 |

The 4% physical membrane load gives stronger smoothing than 1%:

- 515k Laplacian RMS is `0.00385663` lower than the 1% result.
- 3152k Laplacian RMS is `0.00393267` lower than the 1% result.

It also sacrifices more target fidelity:

- 515k target RMS is `0.0849157` higher than the 1% result.
- 3152k target RMS is `0.0539271` higher than the 1% result.

## Interpretation

The physical 2D Stable Neo-Hookean membrane behaves consistently as prestrain
increases: a larger tensile prestrain produces a smoother surface displacement,
but it pulls the relaxed state farther from the `Expression000` target. Both
4% cases reached the 10,000-step cap, so the reported states are not fully
converged under the configured tolerance.

This supports the view that SNH surface prestrain can act as a physically
meaningful smoothing load, but applying it after the inverse solve without any
target-displacement term is not target-preserving. If the goal is both physical
prestrain and target fidelity, the next variant should keep this SNH membrane
and add an explicit target penalty or constrained target boundary points during
the forward relaxation.

## Validation

- `uv run python -m py_compile src/21-surface-prestrain-expression000-prestrain04.py`
- Smoke run:
  `DEBUG=1 CHERRIES_NAME='surface prestrain Expression000 4pct 515k smoke' CHERRIES_TAGS='surface-prestrain,Expression000,stable-neo-hookean,2d,prestrain004,smoke,515k' uv run python src/21-surface-prestrain-expression000-prestrain04.py --cases '["515k-expression000-smas1"]' --forward-max-steps 2`
- Read back all four saved VTU/VTP files with PyVista.
