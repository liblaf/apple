# MouthOpen projection feasibility after run009

Run009 accepted 21 updates from the audited run008 checkpoint, reducing weighted
skin RMS from 2.0953739184741136 to 1.9995158395305934 mm. Both Adam counters
are 254. It then failed at attempted update 22: SLSQP status 8, positive
directional derivative for line search. Process and Cherries shutdown finished
with exit 1. This is a failed optimizer proposal, not inverse convergence.

The saved checkpoint, endpoint and summary arrays, hashes and counters agree.
A fresh independent 146 audit passed: force 0.0009903444384470864 N,
100 retained inversions, inverted rest-volume fraction 1.2304029393884475e-5,
and feasible collision geometry. The full original tet boundary, bones/eyes and
complete branch fit/force curves were rendered with 147. Fifteen tailnet assets
were byte-verified and the front target/fit image was inspected.

Audited run009 preview (private preview omitted).

## What the failed projection shows

Attempt 22's QP added two near-boundary determinant constraints, original cells
48684 and 612399, bringing the total to 12. SLSQP stopped after 26 iterations
and 198 objective evaluations, with large multipliers. The full input matrix
was not saved because serialization happened after successful QP solution;
the traceback truncates matrix columns. This evidence cannot distinguish
numerical failure from infeasibility of the requested decrease target.

All 12 printed geometry lower bounds are negative, so zero pose motion is
feasible in the recorded linear model. The strain contribution is already
descending, -0.00029135952509470853. The requested joint target is
-0.0004974138751, which also demands pose descent. Therefore QP failure does
not establish absence of a feasible descent direction, even in this approximate
model. Prior successful matrices also do not certify endpoint stationarity.

## Next diagnostic

The additive 161 diagnostic will reconstruct the exact saved state with its
fresh shifted gradient, unchanged Adam history and analytic determinant probes,
then save the complete matrix, bounds, gradients and slopes before optimization.
CPU HiGHS will test feasibility of the original joint decrease target.

If that target is infeasible and zero pose is a valid descending witness,
the diagnostic will explicitly test a target equal to half of its strain
slope. If the original target is feasible but SLSQP fails, the diagnostic will
test a solve initialized at the LP's feasible point and verify its constraints
and QP optimality residual. Each decision will be recorded; physical limits
are not changed. One new quarter-step candidate will undergo CCD, force
equilibrium, original inversion gates and objective decrease checks. Diagnostic
states and optimizer proposals are not automatically adopted.

All source runs are preserved. Internal force target remains 1e-9; original
force acceptance remains 1e-8. Collision, exact saved skin pre-strain,
IsFixed-only DOFs, all-four-fixed-tet exclusion, count 100 and inverted
rest-volume fraction 1e-4 remain fixed. No commits or pushes.

## Completed diagnostic

The diagnostic finished with exit 0 and completed Cherries shutdown. HiGHS
classified the reconstructed original target as infeasible. The exact saved
matrix contains 12 determinant constraints. Zero pose is a feasible geometry
witness with negative strain slope -0.0002913595286850904, so the explicit
replacement target is half that slope, -0.0001456797643425452. The QP solved
in two iterations from zero pose. Minimum slack was -1.04e-16, KKT stationarity
residual 6.53e-17 and complementarity residual 4.82e-18. These residuals certify
the small proposal QP, not the full inverse problem's stationarity.

An independent CPU minimization of the pose slope over the saved geometry
constraints gave a best possible joint linear slope of -0.00046598461257776435,
which cannot meet the requested -0.0004974138858415288. Its primal minimum
slack was -2.08e-17, dual stationarity residual 2.07e-17 and duality gap
2.71e-20. This supplies a separate numerical check of target infeasibility;
see `independent-linear-descent-bound.json`.

The alpha 0.25 candidate passed CCD, nonlinear correction, Armijo and all
physical gates. RMS decreased from 1.9995158395305934 to 1.998917492982653 mm.
Force was 0.0009408461912344195 N; retained inversion count stayed 100, with
no newly inverted cells and unchanged inverted rest-volume fraction
1.2304029393884475e-5. The corrector used PNCG and 8 Newton steps. This is
evidence that the failed decrease requirement blocked a useful direction;
it does not establish inverse convergence.

Source checkpoint controls, moments, both counters 254 and input hashes were
preserved. The diagnostic candidate is not adopted. The next additive inverse
run will restart the exact independently audited run009 checkpoint and apply
the explicit feasibility test when choosing its proposal target.

Reproduction from the experiment group:

```bash
TMPDIR="$PWD/tmp/qp-feasibility-001-runtime" \
CHERRIES_NAME='MouthOpen QP feasibility witness diagnostic' \
CHERRIES_TAGS='mouthopen,qp-feasibility,diagnostic' OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/161-diagnose-mouthopen-qp-feasibility.py \
> tmp/161-mouthopen-qp-feasibility-001.log 2>&1
```

Evidence: `data/mouthopen-qp-feasibility-001/{protocol.json,qp-inputs.npz,
qp-result.json,summary.json,candidate.npz,completion-verification.json}`.
The exact process identity is in `job.json`; the live diagnostic receipt and
HTML were byte-verified over tailnet. [Comet diagnostic](https://www.comet.com/liblaf/apple/1c7ad352cec24cfa859eb948591fcb8a).
