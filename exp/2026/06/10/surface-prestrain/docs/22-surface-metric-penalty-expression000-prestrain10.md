# IsFace 10% Metric-Penalty Prestrain on Expression000 Inverse Solutions

## Question

Try the membrane metric penalty with `10%` tensile prestrain on the corrected
`Expression000` inverse solutions. This is a diagnostic surface regularizer,
not the physical 2D Stable Neo-Hookean membrane used in the previous 1% and 4%
runs.

## Setup

- Script:
  `exp/2026/06/10/surface-prestrain/src/22-surface-metric-penalty-expression000-prestrain10.py`
- Run directory:
  `exp/2026/06/10/surface-prestrain`
- Run:
  `COMET_AUTO_LOG_GIT_PATCH=false COMET_AUTO_LOG_GIT_METADATA=false CHERRIES_NAME='surface IsFace metric penalty 10pct prestrain Expression000' CHERRIES_TAGS='surface-prestrain,isface,Expression000,515k,3152k,direct-forward,nu049,metric-penalty,prestrain010' uv run python src/22-surface-metric-penalty-expression000-prestrain10.py`
- Comet:
  <https://www.comet.com/liblaf/apple/10bb3872399341a38bca973192f37daf>
- 515k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-fresh-nu049-smas1-noclamp-super-loose-reg.vtu`
- 3152k source inverse:
  `exp/2026/05/20/inverse-face/data/20-inverse-face-3152k.vtu`
- Target displacement:
  `Expression000` stored as `Displacement` in the corresponding target VTUs.
- Surface selection:
  boundary triangles where all three original vertices have `IsFace = true`.
- Prestrain:
  `prestrain = 0.10`, tensile mode, prestrained length scale
  `1 / 1.10 = 0.9090909090909091`.

## Energy

For each selected surface triangle, let `G` be the deformed first fundamental
form and let `G_p` be the prestrained reference metric:

```text
G_p = alpha^2 G_0
alpha = 1 / (1 + prestrain)
C = G_p^{-1} G
psi = 0.5 * k * trace((C - I)^2)
E_surface = sum(rest_area * psi)
```

The implementation expands the 2D tensor penalty as:

```text
trace((C - I)^2) = (C00 - 1)^2 + 2 C01 C10 + (C11 - 1)^2
```

The run uses `k = surface_stiffness = 1.0`. This penalizes deviation from the
prestrained metric directly. It is therefore a useful surface smoothing
regularizer, but it is less physically complete than the Stable Neo-Hookean
membrane because it does not include the same nonlinear constitutive response.

## Outputs

- `data/22-surface-metric-penalty-expression000-prestrain10-515k-expression000-smas1.vtu`
- `data/22-surface-metric-penalty-expression000-prestrain10-515k-expression000-smas1-surface.vtp`
- `data/22-surface-metric-penalty-expression000-prestrain10-3152k-expression000-smas100.vtu`
- `data/22-surface-metric-penalty-expression000-prestrain10-3152k-expression000-smas100-surface.vtp`
- `data/22-surface-metric-penalty-expression000-prestrain10-summary.json`
- `data/22-surface-metric-penalty-expression000-prestrain10-cases.csv`
- `data/22-surface-metric-penalty-expression000-prestrain10-table.md`

The surface VTPs include `SurfaceMetricPenaltyErrorSq`,
`SurfacePrestrainEnergyDensity`, `SurfacePrestrainAreaRelChange`, and the same
target/previous/prestrain displacement fields used by the SNH runs.

## Results

| case | IsFace points | prestrain tris | previous RMS | prestrain RMS | RMS delta | previous lap RMS | prestrain lap RMS | lap delta | result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 515k-expression000-smas1 | 6,787 | 13,079 | 0.0414332 | 0.267693 | +0.226260 | 0.0191753 | 0.0111371 | -0.00803823 | max steps, 10,000 steps |
| 3152k-expression000-smas100 | 17,582 | 34,372 | 0.0445670 | 0.123839 | +0.0792715 | 0.0136435 | 0.00688141 | -0.00676207 | max steps, 10,000 steps |

Additional roughness checks:

| case | previous surface error-edge RMS | prestrain surface error-edge RMS | delta |
| --- | ---: | ---: | ---: |
| 515k-expression000-smas1 | 0.0284267 | 0.0362374 | +0.00781066 |
| 3152k-expression000-smas100 | 0.0213310 | 0.0173794 | -0.00395163 |

Forward solve residuals:

| case | relative grad norm | total time |
| --- | ---: | ---: |
| 515k-expression000-smas1 | 0.00480015 | 92.0 s |
| 3152k-expression000-smas100 | 0.00425128 | 303.6 s |

## Comparison With SNH Runs

| case | model | target RMS | lap RMS | surface error-edge RMS | success |
| --- | --- | ---: | ---: | ---: | --- |
| 515k-expression000-smas1 | SNH 1% | 0.189323 | 0.0182607 | 0.0342715 | true |
| 515k-expression000-smas1 | SNH 4% | 0.274238 | 0.0144040 | 0.0358544 | false |
| 515k-expression000-smas1 | metric 10% | 0.267693 | 0.0111371 | 0.0362374 | false |
| 3152k-expression000-smas100 | SNH 1% | 0.0947698 | 0.0120072 | 0.0212725 | false |
| 3152k-expression000-smas100 | SNH 4% | 0.148697 | 0.00807453 | 0.0199538 | false |
| 3152k-expression000-smas100 | metric 10% | 0.123839 | 0.00688141 | 0.0173794 | false |

## Interpretation

The 10% metric penalty produces the smoothest displacement field among these
post-inverse relaxation tests:

- 515k Laplacian RMS drops from `0.0191753` to `0.0111371`, lower than both
  corrected SNH runs.
- 3152k Laplacian RMS drops from `0.0136435` to `0.00688141`, also lower than
  both corrected SNH runs.

It is especially favorable on 3152k relative to the 4% SNH run: the metric
penalty has lower target RMS (`0.123839` vs `0.148697`) and lower Laplacian RMS
(`0.00688141` vs `0.00807453`). So yes, by these scalar proxies, the metric
penalty looks better as a smoothing regularizer.

The caveat is still important: the full relaxed states are not target
preserving and both cases hit the 10,000-step cap. The metric penalty should be
treated as a useful diagnostic or regularization term. If we want the physical
interpretation, the 2D Stable Neo-Hookean membrane is the better model; if we
want an effective bumpy-result smoother, this metric penalty is currently the
stronger knob, especially if paired with an explicit target-displacement term.

## Validation

- `uv run python -m py_compile src/22-surface-metric-penalty-expression000-prestrain10.py`
- Smoke run:
  `DEBUG=1 CHERRIES_NAME='surface metric penalty Expression000 10pct 515k smoke' CHERRIES_TAGS='surface-prestrain,metric-penalty,Expression000,prestrain010,smoke,515k' uv run python src/22-surface-metric-penalty-expression000-prestrain10.py --cases '["515k-expression000-smas1"]' --forward-max-steps 2`
- Read back all four saved VTU/VTP files with PyVista.
