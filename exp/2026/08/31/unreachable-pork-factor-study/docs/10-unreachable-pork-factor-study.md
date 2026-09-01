# Unreachable-target streaky pork: 2-D OFAT study

## Scope

This is the current unified 2-D one-factor-at-a-time (OFAT) study. It varies
target height, top-loss type, mesh resolution, and elastic energy around the
same 2-D pork model.

*Unreachable target* is a deliberately demanding benchmark label, **not** an
infeasibility or reachability certificate. These experiments do not prove that
no admissible activation can match a target, nor that the raw-control objective
has a finite global minimizer. The separate
[2-D folding factorial](30-pork-folding-factorial.md) studies matched
interactions; the [downward-target study](20-pork-2d-target-direction.md) is
an orthogonal target-sign check.

## Common 2-D model and protocol

The specimen is \(1\times0.1\), with the middle band
\(0.04\le y\le0.06\) active muscle and the rest fat. Bottom and both lateral
boundaries are fixed; the top and interior are free. The target is the
boundary-compatible parabola

\[
u_{\mathrm{target}}(x)=(0,h\,4x(1-x)).
\]

Muscle is ten times stiffer than fat. There is no skin term, contact treatment,
activation bound, activation regularizer, determinant constraint/barrier, or
inversion repair. Every active triangle has an independent unbounded symmetric
three-DoF activation in \(A^{-1}-I\); active energy uses \(G=FA^{-1}\).
Artifacts are therefore retained as diagnostics rather than suppressed.

Each case uses 1,200 Adam exploration steps followed by unbounded L-BFGS
refinement, with strict forward residual tolerance \(10^{-10}\). The practical
stationarity criterion is \(\lVert g\rVert_\infty\le2\times10^{-8}\) and
gradient RMS \(\le10^{-8}\), alongside valid solves and trial diagnostics.
There is no loss-based early stop. Every recorded state is preserved in
`history.vtu.series`; a final summary records the trace, strict equilibrium,
loss/roughness diagnostics, and determinant minima.

Terminology matters:

- **Forward converged** means one equilibrium solve met its residual tolerance;
  it does not imply inverse stationarity, a unique branch, or positive
  determinants.
- **Practically stationary** means the final iterate passes the stated local
  gradient receipt; it is not a global-optimum or reachability certificate.
- **Horizon-limited/nonstationary** means the finite optimization budget did
  not establish that receipt, even when all forward solves converged.

## Completed current receipts

All eight unified final-case receipts are complete. Every recorded forward
solve converged, and every case reports zero forward, inverse-evaluation, and
refinement-trial failures. `Frames` is the exact recorded-state count
(`evaluations` in the case summary); `Trajectory min DetF` is the minimum over
the whole trace, while `Final min DetF` is the endpoint value. The linked
receipt is the source for each row.

| OFAT change | Stationary | Frames | Final target RMS | High-pass RMS | Top-curvature RMS | Final min DetF | Trajectory min DetF | Failures | Receipt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| baseline: Stable-NH/L2, \(h=.05\), 100x10 | no | 1,245 | .001559 | .001202 | 9.975 | -11.04 | -26.31 | 0 / 0 | [summary](../data/10-pork-2d/baseline/summary.json) |
| height low: \(h=.025\) | no | 1,253 | .0002799 | .0002560 | 1.662 | .4787 | .4277 | 0 / 0 | [summary](../data/10-pork-2d/height-low/summary.json) |
| height high: \(h=.10\) | no | 1,237 | .004714 | .003256 | 23.27 | -45.33 | -131.54 | 0 / 0 | [summary](../data/10-pork-2d/height-high/summary.json) |
| L1 loss | no | 1,406 | .0008708 | .0006915 | 4.622 | -13.81 | -14.93 | 0 / 0 | [summary](../data/10-pork-2d/loss-l1/summary.json) |
| L∞ loss | no | 1,491 | .002763 | .001625 | 12.91 | -12.01 | -18.35 | 0 / 0 | [summary](../data/10-pork-2d/loss-linf/summary.json) |
| 50x5 mesh | yes | 1,282 | .0001468 | .0003284 | .3860 | .6399 | -11.50 | 0 / 0 | [summary](../data/10-pork-2d/mesh-medium/summary.json) |
| 200x20 mesh | no | 1,259 | .000783285 | .000586854 | 12.3409 | -21.4544 | -46.4457 | 0 / 0 | [summary](../data/10-pork-2d/mesh-dense/summary.json) |
| Linear elasticity | yes | 1,297 | .0001471 | .0003458 | .3948 | .009044 | -.07157 | 0 / 0 | [summary](../data/10-pork-2d/energy-linear/summary.json) |

Failures are listed as forward / inverse-evaluation; refinement-trial failures
are also zero for all eight cases. No case was silently discarded for
inversion. Only 2/8 cases pass the physical-stationarity receipt: the 50x5
mesh and linear-elastic cases. This is not a ranking of the interventions: the
other six are retained precisely because their strict forward solves succeeded
but the prescribed inverse run did not establish stationarity.

The dense run is explicitly nonstationary, not early-cut or mislabelled. It
completed all 1,200 Adam updates, then made 749 strict L-BFGS function
evaluations and exhausted five zero-progress restarts
(`stalled_restart_limit`). Its final gradient RMS is \(2.26\times10^{-8}\),
above the \(10^{-8}\) physical-stationarity threshold, despite zero solver
failures. Across all eight cases the canonical root contains 10,470 exact
recorded states.

The canonical visual and analysis receipts are the
[all-eight final-shape sheet](../data/50-2d-final-shapes/all-2d-final-shapes.png),
[final-shape receipt](../data/50-2d-final-shapes/receipt.json),
[completeness receipt](../data/30-analysis/completeness.json),
[convergence receipt](../data/30-analysis/convergence.json),
[case table](../data/30-analysis/cases.csv), and
[factor-effects table](../data/30-analysis/factor-effects.csv). The
[canonical eight-case 30-FPS video receipt](../data/40-paraview-2d/render-receipt.json)
references materialized videos for all eight canonical histories.

## Cautious OFAT readings

- **Target height.** Lowering \(h\) from .05 to .025 reduces final target RMS
  from .001559 to .0002799, high-pass RMS from .001202 to .0002560, and gives
  positive final determinant (.4787). Raising it to .10 instead gives target
  RMS .004714, high-pass RMS .003256, curvature 23.27, and final / trajectory
  minima -45.33 / -131.54. These are matched height observations in this model,
  not a general nonlinear scaling law; none of the three final points passes
  the practical-stationarity gate.
- **Loss.** Relative to L2 baseline, L1 finishes with lower top RMS (.0008708)
  and lower roughness (.0006915 high-pass; 4.622 curvature), while L∞ has
  higher RMS (.002763) and roughness (.001625; 12.91). Both have negative final
  determinants and are nonstationary under the smooth-gradient gate; L1/L∞
  also change loss geometry, so their difference is not evidence that one norm
  generally prevents or causes folds.
- **Mesh.** The 50x5 case is practically stationary, has positive final
  minimum determinant (.6399), and low final roughness (.0003284 high-pass;
  .386 curvature), despite a negative trajectory minimum (-11.50). It has one
  quarter as many triangles and controls as 100x10, so it is not a mesh
  convergence result. The 200x20 case has a lower final target RMS (.000783285)
  than baseline but is nonstationary after its full Adam-plus-refinement
  protocol, has substantially higher curvature (12.3409), and ends inverted
  (final / trajectory DetF -21.4544 / -46.4457). Four times as many cells and
  controls also change optimization conditioning; these two resolutions do not
  establish a monotone resolution trend.
- **Elasticity.** Linear elasticity is practically stationary with final
  minimum DetF .009044 and low target/high-pass RMS (.0001471/.0003458), versus
  the nonstationary Stable-NH baseline with final DetF -11.04 and RMS
  .001559/.001202. This is one matched energy comparison, not an assurance
  that linear elasticity is physical or intrinsically artifact-free: its
  trajectory minimum DetF is still negative (-.07157), and the activation map
  also reaches negative determinant.

## Why bumps can persist despite successful forward solves

The objective observes free top displacement but leaves interior geometry free.
Independent piecewise-constant activations can therefore become locally
incompatible while improving the observed top residual. For active cells,

\[
\det G=\det F\,\det A^{-1}.
\]

Thus a displayed inversion \(\det F<0\) can pair with an inverted activation
map \(\det A^{-1}<0\) and leave \(\det G>0\). Stable Neo-Hookean energy in this
unbounded setup has no determinant barrier. This is a recorded mechanical
route to an artifact, not proof that it is the only cause of visible bumpiness.

There is also a solver-selection limitation. The strict
[branch-selection audit](../data/70-branch-selection-audit/receipt.json) shows
that fixed stored-Adam seeding can select a different strict equilibrium after
an arbitrarily small control perturbation. At the audited per-cell endpoint,
the fixed-seed central difference at \(\epsilon=10^{-5}\) is .210431 versus
adjoint \(-7.6170\times10^{-5}\), while local continuation gives
\(-7.61094\times10^{-5}\) (relative error .000797). Strict forward convergence
therefore does not make the fixed-seed selected objective globally smooth; the
reported gradients and stationarity criteria apply to the recorded local
branch. This qualifies numerical interpretation without relabelling a
branch-selection jump as a forward-solver failure.

## Reproduction and related reports

The final-case roots contain the summary, exact trace, target, final/best VTUs,
and `history.vtu.series`. Re-run from this experiment directory with an empty
output root, retaining horizon-limited outcomes rather than terminating the
matrix on a failed practical gate:

```bash
uv run python src/10-run-pork-2d.py \
  --cases baseline:stable:l2:100x10:.05 \
  --output-dir data/10-pork-2d-final-case-baseline-reproduced \
  --max-steps 1200 --forward-max-iterations 3000 \
  --require-inverse-convergence false --validate-derivatives true
```

Use [the folding factorial](30-pork-folding-factorial.md) for the full matched
2-D interaction matrix and final videos, and
[the downward-target report](20-pork-2d-target-direction.md) only as its
separate sign-flip cross-check.
