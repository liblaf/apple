# Certified rejection outside the joint correction limit

Run015 finished after 14 accepted updates, at RMS 1.9085361133435759 mm and
Adam counters 304/304. Its next alpha 0.5 proposal stopped at an unresolved
dual certificate. The saved endpoint passed fresh physical audit: force
0.0009821411795372357 N, 100 inversions, inverted rest-volume fraction
1.284627435785114e-5 and valid collision. Its full original tet boundary,
bones, eyes and complete fit/force curves are published, with 15 assets verified.
It is not inverse-converged.

## Frozen proposal evidence

Unlike the prior objective-subtraction issue, the failed QP has large dual
multipliers and nearly all strain increments on their component bounds. The
extended-precision dual lower bound is 1.9478017397312184, while the unchanged
correction trust limit permits at most 0.009893093863233922 in the QP objective.
Even a conservative floating-point allowance of about 1.35e-4 leaves a wide
separation. A nonnegative feasible dual point bounds the best possible primal
objective from below; this proposal cannot meet the declared trust limit.
Merely polishing its KKT residual cannot make it admissible.

The same frozen cache at smaller alphas fully certifies under the original
tests. At alpha 0.25, correction norm 0.0413618 is below limit 0.0703317;
at alpha 0.125, norm 0.0206619 is below limit 0.0351658. Full evidence is in
`tmp/run015-attempt15-trust-diagnosis/summary.json` and its bound inputs.

## Continuation policy

The bounded dual helper receives the existing correction-objective cap. A
validated dual lower bound strictly above that cap produces a distinct
`JointTrustRegionInfeasibleError` result and receipt. The runner records the rejected
trial and halves alpha in its normal line search. Generic unresolved numerical
certificates still stop visibly. Successful proposals still pass every original
KKT, box, descent and trust check before CCD and actual nonlinear equilibrium.
No physical allowance or numerical certificate threshold is relaxed.

For the bound, a rounded box-feasible point is allowed: strong convexity gives
the Lagrangian value minus half the weighted squared stationarity residual as
a lower bound when valid complementary bound multipliers are included. A
conservative floating-point envelope is also subtracted. This distinguishes a
proven violation of the existing correction limit from a failed numerical solve.

Run016 starts from the exact independently audited015 checkpoint, all moments
and counters 304/304. Initial alpha is 0.5 so the new classification is tested
at the same proposal scale. The finer startup derivative stencil, joint active
strain and pose policy, physical force threshold 1e-8, internal target 1e-9,
count cap 100 and volume-fraction cap 1e-4 are retained. No diagnostic candidate
is adopted.

## Command

```bash
mkdir -p tmp/run016-runtime
TMPDIR="$PWD/tmp/run016-runtime" \
CHERRIES_NAME='MouthOpen continuation with certified trust backtracking' \
CHERRIES_TAGS='mouthopen,inverse,continuation,bounded-joint,trust-backtracking' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/176-continue-mouthopen-trust-backtracking.py \
> tmp/176-continue-mouthopen-coupled-016.log 2>&1
```

Output: `data/inverse-mouthopen-coupled-016`. Launch requires CPU certificate
and integration review; the state pointer records whether it is active. Actual
physical steps and a finished reduced-objective stationarity audit remain
necessary before claiming convergence.

## Verification and launch

Eleven final-code CPU checks passed, including frozen alpha 0.5 rejection,
alpha 0.25/0.125 acceptance, analytic box lower bounds, cap values below/equal
to/above an analytic optimum, negative dual rejection and preservation of
generic unresolved failures. The final exception is
`JointTrustRegionInfeasibleError`; the runner catches only that class.
Source-bound evidence: `tmp/run015-certified-trust-replay-v2/tests.json`.

Root independently evaluated the separable Lagrangian minimum in extended
precision and verified the smaller-alpha primal constraints and trust norms.
Its dual bound agrees with the helper within 3.1e-15, far inside the explicit
1.35e-4 floating envelope. Independent reconstruction of smaller-alpha controls
differs only by BLAS roundoff below 2.1e-17; the helper's original exact
reconstruction assertion remains unchanged and passed. Receipt:
`tmp/176-root-independent-trust-proof.json`.

Every original solver assertion is retained; only cap validity guards were
added. An independent review found no mathematical or runner blocker. Ruff
and compilation passed. Run016 has launched, with exact process identity in
its `job.json` and the state pointer. Actual startup and physical corrected
updates still need verification.

## Verified startup and nonlinear progress

At 2026-09-30 02:39 UTC, the exact run016 process was active: PID 1029817,
start ticks 4865670, boot `fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session
46034. The checkpoint and endpoint hashes match the audited run015 source.
Initial RMS, pose, activation statistics and counters 304/304 match exactly;
force agrees within roundoff. An independent CPU audit verified source q,
pose and displacement, all four reconstructed Adam moment updates and the
step-305 proposal. The loaded source snapshots match the reviewed files.

The first alpha 0.5 trial was explicitly rejected before physical evaluation:
its conservative dual lower bound 1.9468219514355891 exceeds the unchanged
correction-objective limit 0.00989309379838268. Alpha 0.25 then passed the full
QP certificate, CCD and 67 PNCG correction iterations. The next five accepted
updates also performed actual correction: 80/1, 180/9, 30/0, 22/0 and 54/0
PNCG/Newton iterations. Their alphas were 0.25, 0.25, 0.0625, 0.125 and 0.25.
Other larger proposals were rejected by the original CCD and inversion gates.

At local update 6, counters are 310/310, RMS is 1.8709183867732457 mm, force
is 0.0009965625125781309 N, retained inversion count is 100 and inverted
rest-volume fraction is 1.284627435785114e-5. Rotation norm is
8.695188350294547 degrees and translation norm is 4.445719325897327 mm.
Every accepted update meets the original physical gates and the stricter
internal force target. This is live progress, not an independently audited
finished endpoint or inverse convergence.

Receipts in run016: `trust-backtracking-resume-verification.json`,
`independent-source-audit.json`, `latest-monitor.json` and
`live-tailnet-verification.json`. All four live tailnet assets were byte-verified.
The live page points to run016; the audited full-surface preview remains run015.
