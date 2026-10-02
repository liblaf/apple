# MouthOpen continuation with a feasible decrease target

Diagnostic161 showed that run009 stopped because its requested linear decrease
could not be attained under the current determinant constraints. A smaller
target supported by the feasible zero-pose direction passed actual nonlinear
correction, Armijo, collision and the original force/inversion limits. See
[the diagnostic evidence](161-mouthopen-qp-feasibility.md).

The next additive continuation starts from the exact independently audited
run009 checkpoint with both Adam counters 254 and its unchanged moments. The
diagnostic candidate is not adopted. The proposal algorithm first classifies
the original 0.1 joint-decrease halfspace. If infeasible, a zero-pose geometry
witness with a negative strain slope supports an explicit target of half that
slope. Every choice is recorded, and unresolved feasibility or QP failures
remain visible. This rule changes the numerical proposal only; actual CCD,
nonlinear equilibrium, Armijo and all physical gates decide acceptance.

Relative adjoint/predictor shift remains 1e-4, analytic determinant probes use
epsilon 5e-5, initial alpha is 0.25, q learning rate is 0.002, pose rate is 0.1,
and no extra pose increment caps are applied. Internal force target is 1e-9;
the original physical acceptance threshold remains 1e-8. Collision, exact
saved skin pre-strain, IsFixed-only constraints, exclusion only of all-four-fixed
tets, at most 100 retained inversions and inverted rest-volume fraction at most
1e-4 remain unchanged.

This is continued optimization, not convergence evidence. A final stationarity
claim still requires independent physical audit and reduced-objective checks
using a lower-shift or unshifted adjoint and feasible re-equilibrated probes.
The published audited surface remains run009 until a later endpoint is audited.
All prior runs and unrelated files are preserved; no commits or pushes.

## Run010 launch and verification

`src/162-continue-mouthopen-feasible-witness.py` writes
`data/inverse-mouthopen-coupled-010`. It enables `projection_feasible_witness`
while older wrappers leave it disabled. The QP starts directly at its certified
LP or zero-pose witness; unresolved cases stop visibly. Successful matrix
archives record both original and chosen targets, and pre-solve archives
preserve inputs even when the solver fails.

The focused CPU witness tests verify feasible-target preservation, the
certified infeasible-target branch, visible failure for unresolved/invalid
certificates and unchanged default behavior. Existing determinant tests and
the exact saved161 matrix replay passed. Eleven relevant pytest cases passed,
along with Ruff, formatting, compilation and CPU-only configuration checks.

```bash
TMPDIR="$PWD/tmp/run010-runtime" \
CHERRIES_NAME='MouthOpen feasible witness continuation' \
CHERRIES_TAGS='mouthopen,inverse,feasible-witness' OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/162-continue-mouthopen-feasible-witness.py \
> tmp/162-continue-mouthopen-coupled-010.log 2>&1
```

Launch PID 770610, start ticks 3687572, boot ID
`fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 92488. Exact command,
working directory and project TMPDIR are recorded in run010 `job.json`.
The state pointer and heartbeat now target run010. Startup source hashes,
initial RMS/pose/activation statistics, both counters 254, numerical settings
and original acceptance policy were checked in
`feasible-witness-resume-verification.json`. Four live page assets were
byte-verified over tailnet.

Its first accepted alpha 0.25 step used the certified half-strain target and
actual PNCG/Newton correction. RMS reached 1.9989172384064546 mm, force
0.0009403417825273527 N and 100 retained inversions; both counters are 255.
These are live progress values, not an independently audited final endpoint
or inverse convergence. The run remains active beyond this startup check.

Update 2 then accepted alpha 0.5 using the original feasible target, with
140 PNCG iterations and 10 Newton steps. RMS reached 1.9928957873566548 mm,
force 0.00032596485213935505 N, 100 inversions and counters 256. Thus both
target-selection branches have passed actual nonlinear correction in the new
continuation. The first update used 160 PNCG iterations and 8 Newton steps;
small differences from diagnostic161 are within iterative solver tolerances.
The updated startup verification records each branch and QP residual.

At the 23:20 UTC check, update 3 had accepted alpha 0.25 with 120 PNCG
iterations and 7 Newton steps. RMS was 1.9914515183673682 mm, force
0.0009277460627128872 N, retained inversion count 100 and counters 257.
The process remained running; newer values are in the live receipts.

[Comet run010](https://www.comet.com/liblaf/apple/1ab2f58961ec42f7b9e2a1ceb6ab46a4)
and live progress (private preview omitted).

## Stopped at a nonzero-pose witness requirement

Run010 subsequently stopped at attempted update 4 with
`pose_projection_failed`: the original descent target was infeasible and the
zero-pose geometry witness was unavailable. Numerical process and Cherries
shutdown finished with exit 1. No candidate from attempt 4 was accepted. The
saved endpoint at update 3 and both counters 257 passed independent audit:
RMS 1.9914515183673682 mm, force 0.0009277460627128851 N, 100 retained
inversions and inverted rest-volume fraction 1.2304029393884475e-5.

Only original cell 688235 invalidated the zero-pose witness: its old J was
2.837156513e-5, while the strain-only prediction was -2.348915146e-5. The
pose direction must supply at least 2.448915146e-5 of determinant change to
meet the existing 1e-6 prediction margin. This does not establish a physical
block: a geometry-constrained CPU LP still certifies a negative joint slope.
Read `163-mouthopen-attainable-descent.md` for the generalized target rule.
