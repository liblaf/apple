# Determinant prediction at the run008 inversion boundary

Run008 resumed from refined run006 and accepted 19 joint updates, reaching RMS
2.0953739184741136 mm. Both Adam counters are 233. It was stopped by a
verified-process SIGINT when another force-cutoff branch appeared at the
100-inversion boundary. Numerical process and Cherries shutdown completed with
exit 130. Endpoint, checkpoint and summary arrays, counters and hashes agree.
This is not inverse convergence.

A fresh independent 146 audit passed: force 0.0009906641494569983 N,
100 retained inversions, inverted rest-volume fraction 1.1758180236951449e-5,
no collision intersections, minimum active gap 56.1198 micrometers. The force
is below both the internal 0.001 N target and original 0.01 N acceptance gate.

At update 19, alpha 0.0625, 0.03125 and 0.015625 all triggered nonlinear
correction and finished below internal force tolerance with 101 inversions.
Alpha 0.0078125 was accepted with zero corrector iterations, force
0.0009906641494570022 N and 100 inversions. The compact trial evidence is
`data/inverse-mouthopen-coupled-008/determinant-predictor-diagnostic.json`.
The tighter internal tolerance reduced this effect but did not remove it.

Projection 19 itself solved its 11 determinant constraints plus descent
halfspace accurately: maximum violation 9.77e-17 and projected joint slope
-5.5576587870227125e-5. The four active determinant constraints were original
cells 14438, 513946, 580952 and 667287. Cell 513946 has three fixed vertices and
one free vertex; it is not eligible for all-fixed exclusion. Its positive
determinant was about 1.01717e-7, below the full-step model's 1e-6 margin.
Existing rejected-trial receipts contain aggregate inversion counts, so they
do not establish which cell became the 101st inversion.

The shifted strain tangent solves its requested shifted system accurately but
still has about 11.5% residual against the unshifted equations. Pose basis
residuals are about 0.10–0.15%. Finite geometry approximation also contributes:
for accepted update 18, the predicted determinant of cell 513946 was
1.044660385e-7, compared with actual 1.017170166e-7. These observations motivate
a frozen-state sensitivity calibration rather than another force tightening.

The additive diagnostic `src/159-diagnose-mouthopen-determinant-tangent.py`
will bind the audited run008 checkpoint, hold its controls and optimizer state
fixed, and compare shifts 0.001 and 0.0001. It will measure native residuals,
analytic determinant directional derivatives, and finite-difference error.
The lower-shift case will test a newly projected direction through CCD,
nonlinear force equilibrium and the unchanged inversion gates, saving per-cell
determinants even for rejected corrected states. Diagnostic states are not
automatically adopted. Neither this comparison nor a successful forward step
certifies reduced-objective stationarity.

All original runs and rejected candidates are preserved. Collision, exact skin
pre-strain, IsFixed-only DOFs, all-fixed-cell exclusion, 100 retained inversions,
1e-4 inverted rest-volume fraction and original force acceptance remain fixed.
The internal forward target remains 1e-9. No commits or pushes.

The stopped endpoint was rendered with the full original tet boundary, bones,
eyes and the complete branch's fit/force history. The fixed-control refinement
appears at the same iteration with an explicit marker. Fifteen tailnet assets
were byte-verified; the front target/fit view and fit/force curves were visually
inspected. The mouth remains less open than the target.

Audited run008 preview (private preview omitted).

The diagnostic is running in `data/mouthopen-determinant-calibration-001`.
Its exact PID, boot ID and start ticks are recorded in `job.json` and the active
state pointer. The live continuation page links its status receipt while
retaining the stopped run008 metrics. Both HTTP paths were verified.

```bash
TMPDIR=exp/2026/09/23/new-neutral/tmp/determinant-calibration-001-runtime \
CHERRIES_NAME='MouthOpen frozen determinant tangent calibration' \
CHERRIES_TAGS='mouthopen,isfixed,collision,skin-prestrain,determinant-calibration,diagnostic' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/159-diagnose-mouthopen-determinant-tangent.py \
  > tmp/159-mouthopen-determinant-calibration-001.log 2>&1
```

Ruff and CPU compilation passed. Analytic determinant derivatives were checked
on positive, near-singular, exactly singular and inverted cells. The source
checkpoint's normalized pose reproduces its saved physical pose exactly.
Dependency snapshots and source hashes are preserved in the diagnostic output.

[Run008 independent audit](https://www.comet.com/liblaf/apple/518d4a4571b24bd39c39bfba1a7f5f9e)
and [full-surface review](https://www.comet.com/liblaf/apple/15e0c82a78b7421fac04aa3a35823d3f).

## Completed calibration

Both cases and Cherries finished normally, exit 0. Baseline calibration took
61.8659 seconds; the lower-shift case including its candidate took 47.6164
seconds. [Diagnostic Comet](https://www.comet.com/liblaf/apple/c806dcb2690847228305f813bd731e31).

Reducing relative shift from 0.001 to 0.0001 reduced the full strain tangent's
native unshifted residual from 0.11511786 to 0.02515374. Both requested shifted
systems met their 1e-7 relative residual target. These remain approximate
tangents; the smaller native residual is not an unshifted-solve certificate.

The lower-shift projected candidate at alpha 0.25 passed collision, internal
force and original inversion gates, with no newly inverted cells. Its loss
decreased from 0.07551258571351217 to 0.07546597925752704, corresponding to
RMS 2.094727185054225 mm. Force was 0.0009737401965896845 N and the inversion
count remained 100. It needed zero corrector iterations, so the diagnostic
does not establish that larger corrected steps will pass.

For limiting cell 513946, the actual determinant rose from approximately
1.08239e-7 to 1.94053e-5; the analytical model predicted 1.93951e-5. With the
same candidate direction, the change from lowering shift was about 1.01834e-5,
whereas finite-geometry versus analytical prediction differed by 7.64e-9.
This supports reducing numerical damping and using the calibrated determinant
derivatives. It does not support changing the physical inversion allowance.

The candidate is preserved as a diagnostic and is not adopted. Additive run009
will resume the exact independently audited run008 checkpoint and its own
moments/counters 233/233. It uses fresh gradients at relative shift 1e-4,
analytic cofactor determinant derivatives of small q/pose tangent probes at
epsilon 5e-5, and initial alpha 0.25. The existing 0.1 joint-descent halfspace,
learning rates, internal force target and original physical gates remain.

## Continuation launched (run009)

`src/160-continue-mouthopen-calibrated-tangent.py` now runs in additive output
`data/inverse-mouthopen-coupled-009`. It starts from run008 itself, not from the
diagnostic candidate. Gradients are freshly evaluated at shift 1e-4; the saved
Adam moments still represent the source run's gradient history. Its newly
computed proposal can therefore differ from the diagnostic's fixed proposal.

The analytic determinant path is optional and defaults off for earlier
wrappers. It differentiates det(F) through cofactors, including singular cells,
along small q and pose tangent probes. No deformed-tet matrix inverse is used.
Receipts distinguish the linear q intercept from the actual small-q seed.
Focused derivative checks passed, including exact linearity with three fixed
vertices and zero response to rigid translation. All 11 existing projection,
block-Adam, convergence and tet-policy tests passed; Ruff and CPU compilation
passed. No loaded sources are changed during run009.

```bash
TMPDIR=exp/2026/09/23/new-neutral/tmp/run009-runtime \
CHERRIES_NAME='MouthOpen continuation with calibrated determinant tangents' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,collision,calibrated-tangent,lower-shift,continuation' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/160-continue-mouthopen-calibrated-tangent.py \
  > tmp/160-continue-mouthopen-coupled-009.log 2>&1
```

The authoritative state pointer carries the exact PID, start ticks, boot ID,
entrypoint, log and project TMPDIR. The live progress publisher binds run009;
the full-surface preview remains the independently audited run008 endpoint.

### First corrected steps accepted

Startup reproduced audited run008's RMS and physical pose exactly, with both
initial optimizer counters 233 and matching source checkpoint/endpoint hashes.
The first three updates were accepted at alpha 0.25. They performed 20, 59 and
80 PNCG iterations respectively; the third also performed six Newton steps.
Thus these accepted states include actual nonlinear force correction.

At update 3, RMS is 2.0909908149125704 mm, force 0.0008677240312383849 N,
and the retained inversion count remains 100. Both Adam counters are 236.
The current source hashes, exact process identity, configuration and live
summary/progress assets were verified. The receipt is
`data/inverse-mouthopen-coupled-009/calibrated-tangent-resume-verification.json`.
This restores corrected forward progress; reduced-objective stationarity has
not been certified. The run and continuation heartbeat remain active.
