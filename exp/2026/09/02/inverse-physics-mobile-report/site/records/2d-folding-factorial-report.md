# 2-D pork folding: completed \(2^4\) study

## Scope

This is the completed 2-D Stable-Neo-Hookean/L2 factorial study. *Unreachable
target* is the name of a deliberately demanding benchmark, **not** a
certificate that the target is mathematically infeasible or unreachable by
every admissible activation. The prior
[2-D downward-target report](20-pork-2d-target-direction.md) is an orthogonal
sign-flip check, not a row or factor of this matrix.

## Matrix, target, and model

The 16 cases are the complete \(2^4\) product of domain, muscle extent,
activation sharing, and Poisson ratio. The physical thickness and target centre
height are both 0.1 and 0.05 respectively.

| Domain | Physical domain | Resolution | Triangles | Free top observations |
| --- | --- | --- | ---: | ---: |
| long | \([0,1]\times[0,0.1]\) | \(100\times10\) rectangles | 2,000 | 99 vertices / 198 components |
| short | \([0,0.1]\times[0,0.1]\) | \(10\times10\) rectangles | 200 | 9 vertices / 18 components |

| Factor | Levels |
| --- | --- |
| Domain | long, short |
| Muscle extent | middle band \(0.04\le y\le0.06\), full |
| Activation sharing | independent three-vector per active triangle, one shared three-vector |
| Poisson ratio | \(\nu=0.35,\ 0.49\) |

The target is the boundary-compatible normalized parabola

\[
u_x=0,\qquad u_y=h\,4(x/L)(1-x/L),\qquad h=0.05.
\]

Bottom and side displacement components are fixed; top and interior vertices
are free. The L2 loss observes both displacement components at free top
vertices only. The common 0.01 nominal cell edge does **not** make this a
resolution-convergence study: short and long also differ in physical length,
element/observation count, target wavelength, slopes \(\propto h/L\), and
curvature \(\propto h/L^2\). Thus the short target is physically steeper and
more curved; a domain contrast is not a pure length effect.

The material has only passive fat and active muscle—no skin energy. Fat uses
\(E=0.003\) MPa and muscle \(E=0.030\) MPa; both use the case Poisson ratio.
Band cases have 400 (long) or 40 (short) active triangles; full cases have
2,000 or 200. Per-cell controls therefore have 1,200, 6,000, 120, or 600 raw
DoFs respectively; shared cases have three. There are no activation bounds,
activation regularizer, contact term, determinant constraint/barrier, or
inversion repair. The observed folds are diagnostics of this deliberately
unconstrained model, not physical-material predictions.

## Optimisation and interpretation contract

Every case runs 1,200 Adam exploration updates (initial learning rate 0.03,
multiplied by 0.99 each update), then unbounded objective-normalized L-BFGS
refinement. It has no loss-based early stop. Forward solves in refinement use
residual tolerance \(10^{-10}\); the optimizer internal gradient tolerance is
\(10^{-12}\). A final practical-stationarity gate requires valid accepted
evaluations, no failed refinement trial, nonincreasing accepted refinement
objective, strict final equilibrium, \(\lVert g\rVert_\infty\le2\times10^{-8}\),
and gradient RMS \(\le10^{-8}\).

- **Forward converged** means the equilibrium solve for one evaluation met its
  residual tolerance. It does not imply a unique equilibrium, positive
  determinants, inverse stationarity, or a smooth selected objective.
- **Practically stationary** means the final accepted iterate passes the stated
  local gate. It is not a global-optimum or reachability certificate.
- **Horizon-limited** means the declared optimizer budget did not establish
  that gate. Its retained result is diagnostic, not inverse-converged.

The initial state, each accepted iteration, and the strict L-BFGS seed
re-evaluation are exact recorded VTU states; the series orders them for 30-FPS
playback. Rejected L-BFGS trial points are recorded separately in
`refinement-evaluations.csv` and are never inserted into the frame sequence.
Traces include strict determinant diagnostics:
inverted cells have \(\det F<0\), inverted-rest fraction is rest-area weighted,
and `negative_det_f_mean` is the rest-area-weighted mean of
\(\max(-\det F,0)\).

## Completion receipt

The root [matrix summary](../data/60-pork-folding-2d/summary.json),
[analysis receipt](../data/80-folding-analysis/receipt.json),
[case table](../data/80-folding-analysis/cases.csv), and
[final-shape receipt](../data/90-folding-final-shapes/receipt.json) agree on
the following execution record.

| Item | Result |
| --- | ---: |
| Completed cases | 16 / 16 |
| Exact recorded optimization states, including strict seed re-evaluations | 21,875 |
| Practically stationary cases | 13 / 16 |
| Accepted forward / inverse / adjoint failures | 0 / 0 / 0 |
| Failed refinement-trial forward solves | 0 |
| Non-finite failures | 0 |

The three retained horizon-limited cases are
`l010-band-per_cell-nu35` (\(\|g\|_\infty=1.771\times10^{-6}\)),
`l010-band-per_cell-nu49` (\(4.484\times10^{-5}\)), and
`l100-band-per_cell-nu49` (\(8.347\times10^{-5}\)). Their final gradient RMS
values are \(1.904\times10^{-7}\), \(6.953\times10^{-6}\), and
\(4.849\times10^{-6}\), respectively. They are shown rather than relabelled
as converged; all 16 nevertheless have zero recorded solver failures.

## Final outcomes

`stationary` is the practical gate. `inv.` gives final / peak inverted
rest-area fraction. `Trajectory min det F` is the minimum over every recorded
inverse state, not the determinant at the final state. The three midline
diagnostics are arc-length ratio, turning density, and x-reversal fraction. The
full receipt, including inversion timing, gradient RMS, target RMS, activation
RMS, and determinant sign classes, is in
[results.md](../data/80-folding-analysis/results.md).

| Case | Stationary | Target RMS | Midline arc / turn / reversal | Inv. final / peak | Trajectory min \(\det F\) |
| --- | --- | ---: | --- | --- | ---: |
| l010-band-per_cell-nu35 | no | 9.572e-4 | 2.919 / 76.23 / .10 | .115 / .130 | -75.20 |
| l010-band-per_cell-nu49 | no | .002994 | 1.446 / 44.73 / 0 | .005 / .005 | -11.75 |
| l010-band-shared-nu35 | yes | .01642 | 1.019 / 5.636 / 0 | 0 / 0 | .2906 |
| l010-band-shared-nu49 | yes | .005532 | 1.173 / 20.58 / 0 | 0 / 0 | .4343 |
| l010-full-per_cell-nu35 | yes | 1.490e-10 | 1.147 / 14.33 / 0 | 0 / 0 | .5849 |
| l010-full-per_cell-nu49 | yes | 8.093e-11 | 1.191 / 43.39 / 0 | 0 / 0 | .8440 |
| l010-full-shared-nu35 | yes | .007264 | 1.000 / .0455 / 0 | 0 / .005 | -.5100 |
| l010-full-shared-nu49 | yes | .002968 | 1.127 / 12.26 / 0 | 0 / 0 | .8737 |
| l100-band-per_cell-nu35 | yes | 6.020e-5 | 1.439 / 57.88 / .05 | .024 / .0255 | -9.814 |
| l100-band-per_cell-nu49 | no | .001575 | 1.516 / 78.98 / .04 | .007 / .007 | -26.31 |
| l100-band-shared-nu35 | yes | .004441 | 1.000 / .0436 / 0 | 0 / 0 | .08254 |
| l100-band-shared-nu49 | yes | .003976 | 1.000 / .0497 / 0 | 0 / 0 | .2644 |
| l100-full-per_cell-nu35 | yes | 2.653e-9 | 1.005 / 4.648 / 0 | .001 / .001 | -2.034 |
| l100-full-per_cell-nu49 | yes | 7.967e-7 | 1.029 / 10.14 / 0 | 0 / 0 | .1380 |
| l100-full-shared-nu35 | yes | .0008996 | 1.000 / 5.862e-7 / 0 | 0 / .0005 | -.2238 |
| l100-full-shared-nu49 | yes | .0008986 | 1.000 / 5.974e-7 / 0 | 0 / 0 | .9740 |

Importantly, stationarity is not an orientation certificate. Two stationary
endpoints remain inverted: `l100-band-per_cell-nu35` (final / peak .024 /.0255)
and `l100-full-per_cell-nu35` (.001 /.001). Two stationary shared full cases
have transient inversions that recover by the final frame. The three
horizon-limited band/per-cell cases also remain inverted and must not be mixed
into a stationary-artifact count.

## Descriptive matched comparisons

These are finite-matrix matched comparisons, not causal or general effects.
The machine-readable paired differences and difference-in-differences are in
[factor effects](../data/80-folding-analysis/factor-effects.csv) and
[factorial coefficients](../data/80-folding-analysis/factorial-coefficients.csv).

- **Activation sharing:** per-cell cases have mean target RMS 6.983e-4 versus
  .005300 for shared (7.59-fold higher for shared), while their mean midline
  arc ratio / turning density are 1.462 / 41.29 versus 1.040 / 4.826. The four
  band/per-cell settings include the three failed gates and the most severe
  curl, so this is a fit/roughness association, not a proof that sharing cures
  folds.
- **Muscle extent:** full cases have lower mean target RMS (.001504 versus
  .004494), lower mean arc ratio (1.063 versus 1.439), and lower mean turning
  density (10.60 versus 35.52) than band cases. All eight full cases pass the
  stationarity gate; band has five passes. This contrast is nevertheless not
  an independent proof about material extent, because it changes the active
  region and controls together.
- **Domain:** short cases have mean arc ratio 1.378 and turning density 27.15;
  long cases have 1.124 and 18.97. The individual matched domain differences
  vary, and short also changes physical curvature, wavelength, observations,
  and element count. It therefore does not isolate or solve folding.
- **Poisson ratio:** \(\nu=.49\) has lower mean final inverted rest fraction
  (.0015 versus .0175) but higher mean turning density (26.27 versus 19.85).
  The long band/per-cell pair also becomes more negative in trajectory minimum
  determinant at \(.49\) (-26.31 versus -9.814). Incidence/area and local
  severity should not be collapsed into one monotone elasticity conclusion.

## Determinant signs and curl mechanism

For an active triangle, \(\det G=\det F\,\det A^{-1}\). The paired-sign
class \(F^-A^-G^+\) occupies final rest-area fractions .065,
.005, .0165, .0065, and .001 in the five finally inverted cases, respectively:
`l010-band-per_cell-nu35`, `l010-band-per_cell-nu49`,
`l100-band-per_cell-nu35`, `l100-band-per_cell-nu49`, and
`l100-full-per_cell-nu35`. The alternative \(F^-A^+G^-\) class is .05, 0,
.0075, .0005, and 0. This is the observed local sign mechanism by which a
displayed deformation may invert while the active elastic determinant remains
positive. Stable Neo-Hookean here supplies no determinant barrier.

The top-only observation leaves inner geometry unconstrained; per-cell
piecewise controls have no neighbour regularizer, skin, contact, activation
bound, or determinant barrier. These ingredients make local incompatible
controls and determinant pairing available while top residual is reduced. They
do not prove a unique cause of every visible curl. The raw-control-count-only
story is also not supported: full per-cell has five times the active controls
of band per-cell at a fixed geometry, yet the full matched outcomes are much
less curled in most comparisons. The [midline arc ratio](../data/80-folding-analysis/final_midline_arc_length_ratio.png),
[turning density](../data/80-folding-analysis/final_midline_turning_density.png),
[x reversal](../data/80-folding-analysis/final_midline_x_reversal_fraction.png),
and [midline y-range](../data/80-folding-analysis/final_midline_y_range.png)
plot those measures directly.

## Branch-selection audit

The [branch-selection receipt](../data/70-branch-selection-audit/receipt.json)
compares strict \(10^{-10}\) solves seeded from stored Adam equilibrium with
local continuation from the endpoint. At the horizon-limited
`l010-band-per_cell-nu49` endpoint, the base gradient norm is
\(7.6170\times10^{-5}\). For the minus-gradient direction at
\(\epsilon=10^{-5}\), fixed-seed central difference is .210431 rather than
the adjoint \(-7.6170\times10^{-5}\) (relative error 1.00036), with a
plus-branch objective gap \(4.21014\times10^{-6}\). Local continuation gives
\(-7.61094\times10^{-5}\), relative error .000797.

The well-behaved shared endpoint `l010-band-shared-nu49` makes the same
selection issue visible at a smaller scale: for the minus-gradient direction
at \(10^{-5}\), fixed seed has 97.60% relative error, local continuation has
2.92%; at \(10^{-7}\), local-continuation error is .0238%. This is evidence
that a solver seed can select different strict equilibria under perturbation;
it does not show a forward-solve failure or certify global smoothness. The
reported gradients and stationarity receipts therefore apply to the recorded
branch and accepted optimization sequence.

## Figures and reproduction

The [16-case contact sheet](../data/90-folding-final-shapes/contact-sheets/all-16-final-shapes.png)
uses shared geometry axes and no vertical exaggeration. Matched sheets are
available for [domain](../data/90-folding-final-shapes/factor-pairs/geometry-pairs.png),
[muscle extent](../data/90-folding-final-shapes/factor-pairs/muscle_extent-pairs.png),
[activation sharing](../data/90-folding-final-shapes/factor-pairs/activation_sharing-pairs.png),
and [Poisson ratio](../data/90-folding-final-shapes/factor-pairs/poisson-pairs.png).

To reproduce the matrix, use empty output roots. The command deliberately
retains every result, including the three expected practical-gate failures.

```bash
folding_root=data/60-pork-folding-2d-reproduced
analysis_root=data/80-folding-analysis-reproduced
finals_root=data/90-folding-final-shapes-reproduced
render_root=data/100-paraview-folding-2d-reproduced

uv run python src/60-run-pork-folding-2d.py \
  --cases all --max-steps 1200 --output-dir "$folding_root" \
  --require-inverse-convergence false --validate-derivatives true

uv run python src/80-analyze-pork-folding.py \
  --dimensions 2d --input-2d-roots "$folding_root" --output-dir "$analysis_root"

uv run python src/90-visualize-pork-folding-finals.py \
  --input-roots "$folding_root" --output-dir "$finals_root"

pvpython src/40-render-pork-paraview.py \
  --input-root "$folding_root" --output-root "$render_root"
```

The final [2-D ParaView render receipt](../data/100-paraview-folding-2d/render-receipt.json)
records 16 successful videos. Across the matrix, the source series, PNG
sequence, and H.264 stream each contain exactly 21,875 frames; every case has
matching counts, with no interpolation or duplication. They contain the exact
recorded states, including the strict L-BFGS seed re-evaluation, rather than
only initial-plus-accepted-iteration states. The videos are
30 FPS, 1,800 by 1,006 H.264/yuv420p streams (validated by `ffprobe`), with
duration agreement within one half-frame. Rejected trials remain exclusively
in each case's `refinement-evaluations.csv`. Example videos are the horizon-limited
[short band/per-cell \(\nu=.49\) evolution](../data/100-paraview-folding-2d/2d__l010-band-per_cell-nu49/evolution.mp4)
and the stationary-but-inverted
[long band/per-cell \(\nu=.35\) evolution](../data/100-paraview-folding-2d/2d__l100-band-per_cell-nu35/evolution.mp4).
