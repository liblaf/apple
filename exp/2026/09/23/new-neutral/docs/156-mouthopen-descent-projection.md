# MouthOpen continuation with joint descent in the pose projection

Run006 ended normally at `projected_direction_not_descent` after 87 accepted
updates. Both the numerical process and Cherries shutdown finished, exit 0.
This is an optimizer proposal failure, not inverse convergence.

## Saved endpoint and audit

- Weighted skin RMS: 2.145552398622598 mm, down from 3.0464148747443907 mm.
- Jaw rotation magnitude: 8.535237861665214 degrees; translation: 4.553094278684047 mm.
- Physical free force: 0.008432883425139663 N, below 0.01 N.
- Retained inversions: 97; inverted rest-volume fraction: 1.2549996711732493e-5.
- Minimum determinant: -10.482397572787061. This is not inversion-free physics.
- Both Adam counters: 214. The rejected proposal did not advance either counter.

A fresh rebuild with `146-audit-mouthopen-coupled.py` independently verified the
endpoint, IsFixed DOFs, free lip vertices, exact saved skin material arrays,
collision feasibility without intersections, and the original inversion gates.
The full original tet boundary, bones/eyes, target comparison and complete parent
fit/force history were rendered with 147. Fifteen published HTML/JSON/image/VTP
assets were byte-verified over HTTP; the front target/fit image was inspected.
The mouth remains less open than the target.

- Audited full-surface preview (private preview omitted)
- [Run006 Comet](https://www.comet.com/liblaf/apple/2faa02e00bd24b1fb96855829c345dd1)
- [Independent audit Comet](https://www.comet.com/liblaf/apple/3826fe6aff1441128b73c8d8cbd8fc87)
- [Review Comet](https://www.comet.com/liblaf/apple/bbd74978359f486494d5edf6db973dfb)

## Why the proposal stopped

Attempted update 88 solved its eight determinant constraints accurately, with
two active constraints, maximum violation 3.16e-17 and QP stationarity residual
1.46e-17. However, the QP found the nearest geometrically feasible Adam pose
increment without constraining the objective derivative. The joint directional
derivative changed from -4.3635775495903155e-4 to +1.9209276648279336e-5.
The runner correctly rejected this direction before any nonlinear trial.

The CPU diagnostic reconstructs the pose gradient from saved Adam moments and
the saved unprojected proposal; it infers the strain slope from the recorded
joint slope. Those are explicitly labelled algebraic diagnostics, not a fresh
adjoint or reduced-objective stationarity certificate. Saved evidence:
`data/inverse-mouthopen-coupled-006/descent-direction-diagnostic.json` and
`tmp/156-cpu-descent-diagnostic.py`.

## Minimal numerical change

Run007 retains the same six-dimensional nearest-Adam pose QP and fixed strain
proposal. It adds one halfspace:

`g_q dot dq + g_pose dot dp <= 0.1 * original_joint_directional_derivative`.

The halfspace is normalized by the pose-gradient norm. The actual returned pose
increment is checked again. Extra pose clipping remains disabled so it cannot
change the projected direction. Infeasibility is an explicit failure.

For the saved failed proposal, the augmented CPU QP gives a joint slope of
-4.3635775495903054e-5 and maximum constraint violation 1.94e-17. This only
certifies the local model based on the damped adjoint. Actual candidates still
require CCD, nonlinear force equilibrium, Armijo decrease, at most 100 retained
inversions and at most 1e-4 inverted rest-volume fraction. IsFixed semantics,
all-fixed-cell exclusion, collision, exact skin pre-strain and force atol 1e-8
are unchanged. Relative adjoint/predictor shift remains 0.001.

Five focused CPU tests passed: feasible descent after a geometry-only ascent,
unchanged already-feasible descent, visible failure of incompatible constraints,
joint Adam equivalence, and frozen block state. Ruff passed on changed sources
and tests. Independent code review found no blocker. No commits or pushes.

## Continuation

Run007 starts from the exact audited run006 checkpoint with both moment pairs
and counters 214, learning rates q=0.002 and pose=0.1, and initial trial alpha 1.
All proposal sources are snapshotted by the runner before optimization.
Use `data/mouthopen-continuation-state.json` for the exact current process.

```bash
TMPDIR=exp/2026/09/23/new-neutral/tmp/run007-runtime \
CHERRIES_NAME='MouthOpen continuation with joint descent pose projection' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,collision,coupled-predictor,descent-projection,continuation' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/156-continue-mouthopen-descent-projected.py \
  > tmp/156-continue-mouthopen-coupled-007.log 2>&1
```

Run006 logged a Comet replay database error when `/tmp` filled. Local checkpoint,
progress and audit files are intact on the project filesystem. Run007 uses a
new TMPDIR there; other jobs' temporary files were preserved. This affects log
transport reliability, not the recorded numerical acceptance policy.

[Run007 Comet](https://www.comet.com/liblaf/apple/237927bec7eb4e868070989c5a814777).
The live page (private preview omitted)
points to run007. No inverse convergence is certified. A future stationarity
candidate still requires an independent physical audit, an unshifted or clearly
labelled lower-shift adjoint, and feasible re-equilibrated directional probes.

### First actual forward acceptance

Run007 rebuilt the saved start at exactly 2.145552398622598 mm RMS with both
initial counters 214 and a verified source checkpoint hash. Its first two
updates were accepted at alpha 1. After update 2, RMS is 2.1423050164215875 mm,
force is 0.008372816094222371 N, and 98 retained tetrahedra are inverted. Both
steps passed collision, force, geometry and actual loss-decrease checks.
The live page and its receipts were verified against run007. This clears the
specific projected-ascent stall; it does not establish inverse convergence.
See `data/inverse-mouthopen-coupled-007/descent-projection-resume-verification.json`.

### Force-threshold branch under the inversion cap

Run007 was cleanly interrupted after update 23, both Adam counters 237, to
diagnose repeated vanishing progress. Cherries shutdown completed with exit 130.
The saved summary, endpoint and checkpoint arrays/counters were checked for
consistency before changing its status to `stopped_for_force_threshold_diagnosis`.
RMS is 2.106035576781888 mm; force is 0.009999352723935217 N; retained inversions
are 100. This is not inverse convergence.

At local update 22, alpha 0.001953125 started at raw force 1.000292e-8, triggered
80 PNCG and two Newton steps, and ended with 101 inversions. Alpha 0.0009765625
started at 9.998197e-9 and was accepted with no correction. Update 23 shrank again
to alpha 0.000244140625. The tangent displacement scales with alpha, while
nonlinear correction of the old residual does not disappear when crossing the
stopping threshold. Saved compact evidence is
`data/inverse-mouthopen-coupled-007/force-threshold-diagnostic.json`.

The next diagnostic fixes all controls and tightens only the internal forward
stopping target to 1e-9. It separately reports admission under the unchanged
original 1e-8 force, collision and inversion gates. Rejected diagnostic states
cannot replace the inverse checkpoint. If run007 refinement fails those gates,
the same test will be applied to audited run006 with its own controls and Adam
state. No residual term is being inserted into every tangent seed, which would
break its zero-step consistency.
