# Feasible descent and stationary termination

The user requested a fix for review finding 1. Coordinatewise Adam followed by
spectral PSD projection need not be a descent direction. A fresh Adam restart
can repeat the same uphill proposal, causing the old strict-slope assertion to
abort even at a feasible, nonstationary state. An exactly constrained-stationary
state can also produce zero slope and hit that assertion.

## Change

Descending projected Adam proposals keep their existing behavior. A non-descent
proposal instead uses the same metric as the declared stationarity check:

`dq = project_PSD_box(q - gq / mass) - q`

`dtheta = clamp(theta - gtheta, 0, 4) - theta`

Each tetrahedron has one positive scalar mass for all six tensor coordinates,
so spectral projection is also projection in this mass metric. Its optimality
condition gives `g dot d <= -sum(mass * dq^2) - sum(dtheta^2)`. A common positive
factor caps the largest coordinate change at the configured learning rate,
preserving descent and feasibility. The existing roughness, Armijo, primal-error
and physical/contact checks still decide whether to accept the trial.

Accepted corrected proposals reset Adam moments and record their direction
method and scale. Rejected proposals preserve the accepted checkpoint, Adam
state and warm adjoint. Nonfinite values fail visibly. A nonzero projected
direction without numerically resolved descent records `descent_unresolved`.

An exactly zero projected direction is an explicit constrained-stationarity
certificate. It terminates as converged only with the existing projected-gradient
and primal-accuracy checks satisfied. This is a separately recorded exception to
the ordinary five-consecutive stable-window requirement; it creates no fictitious
accepted steps. Unresolved primal accuracy instead records
`stationary_primal_unresolved`. Both unresolved statuses are explicit failures,
not convergence claims.

## Validation

`data/projected-descent-check-001/summary.json` binds the tested runner and test
source hashes. Its tiny CPU fixtures exercise the production `Fitter.fit_step`:

- The reported two-tetrahedron counterexample gives old Adam slope
  `+0.0005798988714`; the corrected feasible direction gives
  `-0.001009722879`, with maximum coordinate step `0.003`.
- Accepted correction records the method and resets Adam moments.
- Three rejected proposals preserve the entire checkpoint byte hash, optimizer
  state and prior warm adjoint.
- Resolved and unresolved stationary jaw-bound cases terminate without evaluating
  a new trial or incrementing accepted steps.
- NaN and infinity gradients fail before trial evaluation.
- Sequential scheduling keeps working on A until its declared iteration budget.

`data/expression-continuation-check-004/summary.json` also passes the existing
scheduling, import preservation and incompatible-parent rejection checks.
Ruff and Python compilation pass for all changed scripts.

Commands, from this experiment directory:

```bash
DEBUG=1 CHERRIES_NAME='Projected descent CPU regression' \
CHERRIES_TAGS='expression-fit,projected-descent,cpu-check' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/103-check-projected-descent.py

DEBUG=1 CHERRIES_NAME='Sequential continuation after projected descent fix' \
CHERRIES_TAGS='expression-fit,sequential,continuation,cpu-check' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/102-check-expression-continuation.py \
  --output-dir data/expression-continuation-check-004
```

## Continued fitting

Run 004 was stopped after saving MouthOpen update 7. Run 005 continues that state
and the saved BrowDownLeft update with their original Adam moments. All 21 imported
files are hash-recorded in `data/expression-fitting-005/continuation.json`.
The objective, materials, strong smoothness, hinge, collision and gradient
methods are unchanged. The new descent policy is recorded in the run protocol.

```bash
CHERRIES_NAME='Sequential expressions with feasible projected descent' \
CHERRIES_TAGS='expression-fit,sequential,fixed-material,rigid-eyes,pncg,mandible-hinge,projected-descent' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/93-fit-expressions.py \
  --continue-from data/expression-fitting-004
```

The runtime-only fitting service and review publisher both follow run 005.
Startup verified MouthOpen remains current with 7 accepted updates, RMS
`7.507920 mm`, and hinge angle `0.179908 degrees`. Those values were inherited
from run 004, not improvements attributable to this correction. The CPU checks
establish the defect correction; resumed GPU fitting remains unconverged.

[Comet run](https://www.comet.com/liblaf/apple/4999e9d224d748618c051418f0613ef7) ·
Live review (private preview omitted).

Review findings about the Hessian diagnostic flag and end-to-end hinge-gradient
validation are outside this requested fix and remain open.
