# Regional baseline force-space audit

The documented 80-coefficient candidate offers only a modest improvement in
exact-rest force balance. It has **not** been activated in the neutral or final
inverse model. Force residuals at the exact reference geometry do not determine
whether the nonlinear 0.25 mm neutral surface-motion budget is feasible.

The [valid receipt](../data/spatial-baseline-audit-005/summary.json) uses 78 bulk
coordinates: four anchors each in fat and aponeurosis, five in muscle, with the
two existing skin coordinates excluded from this force fit. The basis follows
shared-face connected components. Four noncoplanar anchors lie in each dominant
component; the second muscle component owns its fifth anchor. Each remaining
small component inherits the anchor nearest its volume centroid. Nonnegative
weights sum to one. Every stress column is normalized by its tissue shear
modulus, and every anchor tensor is constrained spectrally to [-0.9, 10].

| Fit at 80.6 N/m skin resultant | Raw force residual | Dual-volume-weighted residual |
| --- | ---: | ---: |
| Constant 18 columns, weighted least squares | 98.9328% | 99.4780% |
| Regional 78 columns, weighted least squares | 98.2298% | 99.0760% |
| Regional, spectral bounds | 98.1809% | 99.1273% |
| Regional, bounds and smoothness weight 1 | 98.2109% | 99.1396% |
| Regional, bounds and smoothness weight 100 | 98.8540% | 99.4433% |
| Regional, bounds and smoothness weight 10000 | 98.9244% | 99.4745% |

Residuals are fractions of the prescribed skin-force norm, not displacement
errors. The total raw target norm is 3.797741 N. Raw- and weighted-norm minimizers
need not rank the same in the other norm. The selected smoothness weights are
force-space sensitivities, not a calibrated final inverse prior.

The bounded objective is `0.5 ||W(Aq-b)||² / ||Wb||² + 0.5 beta R`, where
`R` is the unweighted mean of the three individually volume-normalized tissue
graph energies. `W` contains inverse square roots of nodal dual volumes with a
common median normalization. `b` cancels the skin-only force increment; fixed
passive/contact forces are excluded from this target. The M matrices report
mean volume-weighted stress magnitude as a diagnostic; no magnitude penalty
enters this particular force-space objective.

Both constant and regional matrices have full column rank. Summing each
tissue's anchor columns reproduces its independently assembled constant column
with maximum relative error **4.12e-16**. Direct graph and volume quadratics agree
with the coarse G/M matrices. The bounded quadratic uses exact tensor spectral
projection and an independently evaluated unit-step KKT residual below 1e-8;
both zero and unit smoothness weights reach an active lower eigenvalue bound.

The source and arrays are archived with the receipt. The [Comet run](https://www.comet.com/liblaf/apple/b81ac235161e419c9fbbad8b738a585c)
completed normally. It assembled forces and solved small convex force-space
problems; it did not run nonlinear equilibrium or the large joint inverse.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
CHERRIES_NAME='Exact rest spatial baseline audit with spectral quadratic' \
CHERRIES_TAGS=joint-inverse,neutral,spatial-basis,force-audit,preparation \
uv run --frozen python src/23-audit-spatial-baseline.py \
  --output-dir data/spatial-baseline-audit-reproduction
```

Earlier audit `003` is **invalid evidence for this candidate**: it used different
anchors, unscaled stress columns with dimensionless bounds, and coordinate box
constraints instead of spectral constraints. Its original files remain beside a
root validation note. Audit `004` stopped at an SLSQP iteration limit and is not
accepted. Audit `005` uses a projected convex quadratic with explicit KKT
verification. Later source edits added the explicit objective description to
future receipts and adjusted lint annotations/variable naming; the numerical
algorithm is unchanged. The exact executed source remains under `005/sources/`.
