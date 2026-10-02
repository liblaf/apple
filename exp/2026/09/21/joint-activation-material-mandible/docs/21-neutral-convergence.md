# Neutral shared-field convergence

## Purpose

`src/21-neutral-converge.py` prepares the full neutral face before the large joint inverse run. Its default constant basis optimizes the 18 signed bulk baseline-stress coordinates and the shared skin-stiffness coordinate. Its receipt-gated spatial basis optimizes 78 bulk anchor coordinates and the same stiffness coordinate. The isotropic skin baseline stress stays prescribed during each continuation stage, so a zero-stress state cannot satisfy the preparation by construction.

The current continuation targets are 10%, 25%, 50%, and 100% of the 80.6 N/m skin-resultant proxy. A higher stage may start only from a checkpoint that completed the preceding stage. The proxy is a continuation device rather than a measured subject-specific prestress.

## Convergence contract

The implementation retains spectral projected gradient as a baseline and uses
full-memory metric-projected BFGS for the report-worthy continuation. The BFGS
direction minimizes its 19- or 79-dimensional quadratic model over the exact
bulk spectral and skin-stiffness bounds. Its projected-gradient KKT residual must be
at most `1e-10`; a feasible unconstrained step is used directly. Monotone Armijo
backtracking remains the outer acceptance gate. Every rejected proposal restores
the accepted shared coordinates, displacement, warm adjoint, material state,
fixed values, and numerical receipts. Projection preserves the prescribed skin
stress exactly.

Numerical convergence requires all of the following:

- projected-gradient infinity norm at most `1e-3`;
- relative objective range at most `1e-5` over five accepted evaluations;
- both tests for three consecutive accepted evaluations after at least five updates.

The neutral shape budget is evaluated independently: no inverted tetrahedra, `det(F)` in `[0.25, 2]`, minimum skin area ratio at least `0.25`, surface RMS motion at most `0.25 mm`, and muscle-centroid RMS motion at most `0.5 mm`. A stationary point outside these budgets is recorded as a basis/model feasibility result rather than an optimizer failure.

Final preparation additionally requires explicit soft-tissue–bone contact and a numerically valid contact receipt. A no-contact run can test optimizer mechanics but cannot produce `preparation_complete=true` or a successful preparation checkpoint. The neutral oral audit is diagnostic and does not assert anatomical validation.

## Optimizer smoke

The successful bounded smoke was run from the experiment directory with:

```bash
DEBUG=1 \
CHERRIES_NAME="Neutral convergence optimizer smoke 004" \
CHERRIES_TAGS="joint-inverse,neutral,convergence,smoke,no-contact" \
uv run --frozen python src/21-neutral-converge.py \
  --mode smoke \
  --output-dir data/neutral-convergence-010-smoke-004 \
  --max-accepted-steps 1 \
  --max-forward-evaluations 10 \
  --wall-budget-seconds 600
```

The one projected Armijo proposal was accepted. The objective decreased from `0.6135842607` to `0.6081172144`, and the projected-gradient infinity norm decreased from `0.7152945` to `0.4821385`. The run used two forward and two adjoint evaluations in `26.43 s`; the accumulated solver times were `12.65 s` forward and `4.44 s` adjoint. The terminal state had surface RMS motion `0.18757 mm`, muscle-centroid RMS motion `0.10624 mm`, minimum `det(F)=0.83553`, zero inversions, and `neutral_invariants_ok=true`.

The smoke receipt correctly says `success=false` and `status=smoke_completed`. It establishes that the projected step, strict forward/adjoint solves, rollback-ready checkpoints, corrected neutral audit, and JSON trace work on the full mesh. It is not convergence evidence and does not include contact.

Two earlier fresh smoke directories are preserved as failure evidence. `neutral-convergence-010-smoke-002` exposed non-finite JSON serialization for the incomplete stabilization window; `neutral-convergence-010-smoke-003` then exposed logging of an intentionally absent range metric. Both defects were fixed before smoke 004.

## Contact validation and smoke

`src/12-validate-contact.py` validates the declared physical IPC barrier before it can enter a preparation run. Its synthetic four-tetrahedron fixture contains one active point-triangle pair and uses the same `dhat=0.1 mm`, `0.01 MPa` area-weighted physical barrier as `data/contact/config.json`. The machine receipt at `data/contact-validation/summary.json` passed:

- energy/force directional finite differences: maximum relative error `6.07e-7`;
- exact contact-Hessian vector product: maximum relative error `6.33e-7`;
- implicit material plus all six jaw-coordinate derivatives: maximum vector relative error `4.85e-5` and maximum component error `2.96e-4`;
- direct crossing CCD fraction `0.3984`, with the Equilibrium boundary guard independently rejecting a crossing proposal at fraction `0.3281`;
- queued two-expression snapshot errors exactly zero in this fixture.

This certifies the numerical barrier path and the contact contribution to the implicit Hessian. It does not measure a soft-tissue–bone interface law. The associated normal Comet run is [ff06e64ab0d04e77a560c9061e98d74a](https://www.comet.com/liblaf/apple/ff06e64ab0d04e77a560c9061e98d74a); numerical output completed, while shutdown metadata was interrupted after Comet and the default Git plugin spent more than two minutes scanning the 164,000-file dirty checkout. No numerical receipt depends on that metadata upload.

The contact-enabled full-face smoke at `data/neutral-convergence-010-contact-smoke-001` then accepted one projected Armijo step. Objective decreased from `0.6135873` to `0.6081167`, and projected-gradient infinity norm decreased from `0.7152868` to `0.4821511`. Both states had valid contact receipts; active pairs changed from 143 to 141 and minimum active distance from `49.69` to `49.62 um`. Surface and muscle budgets and corrected neutral invariants passed. The contended run took `96.11 s` for two forward and two adjoint evaluations and correctly records `success=false`, `status=smoke_completed`.

The first report-mode 10% segment reached three accepted updates (`objective=0.5894795`, projected-gradient infinity norm `0.7063149`) before its process ended without a summary, failure receipt, Python traceback, retained exit code, or kernel OOM record. Its cause is unknown, so it is retained only as an interrupted trajectory at `data/neutral-convergence-010-contact`. The checkpoint is loadable, but it is not convergence evidence. A briefly launched second segment was intentionally interrupted before its first solve to wait for the concurrent full-face contact derivative gate. Future runs disable Comet git metadata and patch discovery through the documented experiment-local SDK environment keys; archived source, input hashes, and protocol provenance remain enabled.

## Evidence layout

Each run writes:

- `protocol.json`: input/checkpoint hashes, fixed target, optimizer and convergence contract, and contact configuration;
- `trace.json`: accepted evaluations with objective terms, projected and raw gradients, shape metrics, contact and oral diagnostics, material spectra/bounds, solve receipts, and cumulative timing;
- `trials.json`: every accepted or rejected Armijo proposal;
- `terminal.pt`, periodic checkpoints, and `best-admissible.pt`;
- `summary.json`: distinct optimizer, shape-budget, contact, and final-preparation booleans.

`success=true` is reserved for a contact-enabled state with optimizer convergence, neutral-budget compliance, and a valid contact receipt. Finite step, evaluation, or wall-time exhaustion remains `not_converged`.

## Remaining full run

The contact-enabled 10% stage has converged. The constant-basis 25% stage
reached stationarity outside the surface-motion budget. The Spatial80 25%
segment entered the budget but exhausted 200 accepted updates before
stationarity, so the sequence remains stopped before 50% or 100%. The
continuation is run by
`src/22-run-neutral-continuation.py`. Every stage
uses `forward_rtol=1e-6`, `forward_atol=1e-12`, and `adjoint_rtol=1e-7`. A stage
stops the continuation if it fails to reach stationarity, violates hard
geometry, or exhausts its declared budgets. Convergence figures are generated
from each accepted trace by `src/42-render-convergence.py`. The converged 10%
state is an intermediate continuation checkpoint. Full neutral preparation was
not achieved and still requires successful 25%, 50%, and 100% stages.

## Opt-in inexact Newton-CG forward pilot

The default equilibrium method remains PNCG. An opt-in `newton_cg` method now uses exact FEM plus IPC Hessian-vector products with `linear_rtol=1e-3`, at most 12 Newton steps, verified linear residual and descent, CCD-capped strict Armijo, fresh owned contact states for every trial, and the unchanged final force criterion. It has no MINRES or PNCG fallback. Its protocol and every forward receipt identify the selected method and numerical settings.

The normal synthetic validation at `data/contact-validation-newton/summary.json` passed the direct barrier checks, implicit material plus all-six-jaw finite differences, and queued-expression isolation. The maximum implicit vector relative error was `1.59e-6`, maximum component error was `3.37e-5`, and queued errors were zero. The run is [2ab96518a4fd43a387f2782c02367d85](https://www.comet.com/liblaf/apple/2ab96518a4fd43a387f2782c02367d85).

The bounded full-face smoke at `data/neutral-convergence-010-contact-newton-smoke-001` started from the atomically saved PNCG update-7 checkpoint and accepted two outer updates. It used three forward and three adjoint solves in `53.8 s`. Each outer proposal passed on its first trial. The inner solves took one or two Newton steps and `4.0-7.8 s`, with linear residuals between `9.69e-4` and `9.99e-4`, `CCD=1`, full Newton steps, no inner backtracking, and terminal force norms between `1.2e-15` and `2.36e-13`, below the unchanged `1e-12` threshold. Contact and neutral budgets passed throughout.

For the two matched outer states, Newton and the still-running PNCG trajectory agreed in objective within `9.1e-7` and in projected-gradient infinity norm within `3.6e-7`. The matched first PNCG forward solve took `45.5 s` and 635 PNCG steps; the corresponding Newton forward solve took `7.8 s`. This validates the bounded inner solver pilot. Outer shared-parameter stationarity remains a separate unresolved problem: the PNCG-backed spectral projected-gradient trajectory continued to reduce objective while its projected gradient oscillated and sometimes increased.

## Metric-projected BFGS and converged 10% state

`src/16-validate-outer.py` checks the constrained quadratic solver on analytic
active boxes, a coupled SPD metric, rotated bulk spectral clipping, the skin log
bound, zero directions at interior and active-bound stationary points, direct
feasible steps, and nonsymmetric or indefinite metric rejection. The final CPU
receipt is `data/outer-validation-v2/summary.json`; its maximum KKT residual is
`2.16e-11`, below the fixed `1e-10` tolerance. The normal run is
[102e15542e7b40bf982378aed0ef373f](https://www.comet.com/liblaf/apple/102e15542e7b40bf982378aed0ef373f).

The current-source active-bound pilot at
`data/neutral-convergence-010-contact-metric-bfgs-smoke-002` moved the skin
stiffness multiplier from `2.4963` to its declared upper bound `3.0`. Two full
steps reduced objective from `0.3506954` to `0.2945044` and projected-gradient
infinity norm from `1.7634` to `0.4871`; both metric subproblems met the KKT
tolerance and every shape/contact gate passed. The preceding smoke 001 is kept
as pre-fix evidence with its own archived sources and is not attributed to the
adopted implementation.

The report-worthy 10% run is
`data/neutral-convergence-010-contact-metric-bfgs-segment-003`. It converged in
82 accepted updates, 91 forward evaluations, and 83 adjoints over `1490.05 s`.
The terminal objective is `0.2336667975`, projected-gradient infinity norm is
`7.66e-5`, and the five-state relative objective range is `8.04e-7`; both tests
passed for three consecutive accepted states. Surface RMS motion is `0.11374
mm`, muscle-centroid RMS motion is `0.07967 mm`, minimum `det(F)` is `0.6351`,
there are no inverted tetrahedra, and contact remains numerically valid with 138
active pairs. The checkpoint records `preparation_complete=true` for the 10%
stage and fixes the skin stiffness multiplier at its upper bound. The normal run
is [85cc4ac98c224d6584ee402f72aaebf0](https://www.comet.com/liblaf/apple/85cc4ac98c224d6584ee402f72aaebf0).

## Stationary 25% constant-basis limitation

The guarded continuation is preserved at
`data/neutral-continuation-metric-bfgs-002`. Its 25% child reached numerical
stationarity in 87 accepted updates, 89 forward evaluations, and 88 adjoints
over `1277.17 s`. The terminal projected-gradient infinity norm is `1.07e-5`
and the five-state relative objective range is `1.60e-6`, with three consecutive
qualifying states. Contact is numerically valid, there are no inverted
tetrahedra, minimum `det(F)` is `0.6585`, and muscle-centroid RMS motion is
`0.19656 mm`.

The surface RMS motion is `0.27598 mm`, above the fixed `0.25 mm` budget.
Accordingly, the child receipt says `optimizer_converged=true`,
`neutral_budget_met=false`, and `status=stationary_outside_neutral_budget`.
The wrapper stopped and did not launch 50% or 100%. The terminal checkpoint is
`data/neutral-continuation-metric-bfgs-002/neutral-025/terminal.pt` with SHA-256
`cf4ac850733163c4d308eed594f0413dbf0771d0cf44fbe5f88c356d4d3c063a`.
This is evidence that the optimized constant shared-stress basis misses the
declared 25% shape budget. It is not a general material-capacity impossibility
claim; a separately validated spatial-basis nonlinear probe is the next bounded
test.

## Spatial80 25% nonlinear probe

The spatial extension is explicit rather than a default change. It binds the
immutable 4/4/5-anchor basis and audit receipts, embeds a constant20 checkpoint
exactly into 80 coordinates, fixes skin baseline index 78, and optimizes the
remaining 79 coordinates. The objective adds exactly
`0.5 * 100 * bulk_spatial_roughness`; the strong weight is separate from
`prior_total`. Checkpoints store the exact basis receipt, field layout, CPU and
full-face derivative gates, objective fingerprint, and one-way embedding
receipt. BFGS and SPG history may be reused only when the complete target,
basis, objective, contact, input, solver, and coordinate fingerprint matches.

The CPU/CUDA field receipt at
`data/spatial-fields-validation-cpu-v8/summary.json` passed 19 checks, including
constant embedding, differentiable reconstruction and G/M forms, spectral
bounds, fixed skin target, and consistent implicit CUDA device placement. The
contact-enabled full-face gate at
`data/spatial-face-gradient-validation-002/summary.json` passed 16 directional
checks over 33 forward solves. Its maximum total-objective relative error is
`0.002494` and maximum mechanical relative error is `0.002498`, below the fixed
2% gate.

The first report-mode nonlinear segment is preserved at
`data/neutral-convergence-025-contact-spatial80-metric-bfgs-001`. It reached the
declared 200-update cap in `2754.29 s` without numerical or physical rejection:
all 200 Armijo steps used fraction 1, all 200 curvature updates were accepted,
and every metric subproblem met the `1e-10` KKT tolerance. The objective fell
from `1.3748984` to `0.6532545`. Surface RMS motion fell from `0.27598 mm` to
`0.18302 mm`, muscle-centroid RMS motion is `0.15214 mm`, minimum `det(F)` is
`0.57383`, there are no inverted tetrahedra, and contact remains valid with 141
active pairs and minimum active gap `50.29 um`. This demonstrates that the
Spatial80 basis can enter the declared 25% neutral shape budget under the frozen
strong smoothness prior.

The segment is not converged: terminal projected-gradient infinity norm is
`0.02429` and the five-state objective range is `7.18e-4`, above `1e-3` and
`1e-5`. The summary therefore says `success=false`, `status=not_converged`, and
`stop_reason=accepted-step budget`. The inverse-Hessian condition number grew
to about `1.09e4`; the final metric subproblem took 1839 small CPU projection
iterations but remained strict and finite. Over the last 100 updates, a log
linear fit gives a projected-gradient half-life of about 48.5 updates and an
indicative 223 further updates to reach `1e-3`; objective stabilization may
require more. A same-target continuation may reuse the exact stored H only
after its objective fingerprint matches. No 50% or 100% stage may start before
actual 25% convergence and matching visual receipts. The normal run is
[a05b3d7df06740b3810435a06c224e5f](https://www.comet.com/liblaf/apple/a05b3d7df06740b3810435a06c224e5f).
The terminal checkpoint SHA-256 is
`fb98ed5d398625c2900adf3241ab3a01d74633730f3c05a06170bdddb16f8c65`;
the summary SHA-256 is
`af06dcde90ea0982314f92bb7905a0bc08989ef710f627bf1b8ccd2d0401a6b7`.
