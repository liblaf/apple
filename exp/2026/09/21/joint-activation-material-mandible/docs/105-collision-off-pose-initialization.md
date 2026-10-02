# Collision-off pose initialization

The user proposed disabling collision while fitting the mandible. Run 007 tests
that change with zero muscle activation and the existing one-angle hinge. The
full source skull, mandible, and fixed eyes remain available for the contact-on
joint stage. This is an initialization experiment, not a contact-validated fit.

## Why this experiment

In run 006, every accepted pose increment was 0.03125 degrees. The initial
1-degree proposal and all later 0.0625-degree proposals failed boundary CCD
before PNCG could relax free tissue. The first eight accepted steps reached
0.25 degrees with RMS 7.630657 to 7.465654 mm. Forward solves consumed about
93% of recorded elapsed time; each took thousands of PNCG iterations.
Disabling IPC removes both this initialization-path restriction and its energy,
gradient, Hessian and collision-query cost. The resulting fit may penetrate
rigid anatomy, so the measured angle is only a candidate initializer.

## Model and handoff

The new `joint_collision_off_pose.py` uses the same accepted-state force PNCG,
strict line search, damping, restart interval, step-norm limit, and implicit
adjoint machinery. Materials, prescribed skin stress, target shapes, hinge
geometry, and zero muscle activation are preserved. No contact term is present
in the pose-stage objective. The original adopted neutral is still the target
origin: a collision-off relaxation does not rebase the expression targets.

The run verifies the adopted neutral with collision on once, then starts each
expression with a collision-off zero-pose equilibrium and pose fitting. Saved Initial RMS is the
actual collision-off zero-pose error, which may differ from the original target
motion RMS. Pose steps use the existing bounded scalar/secant optimizer.

At pose stationarity, `pose-only-final.pt` and `pose-stage.json` preserve the
collision-off result. Before joint activation fitting:

1. Extend the saved FEM deformation with the full source skull, posed mandible,
   and fixed eye nodes. Check soft-rigid triangle intersections and active-pair
   clearance using the original contact object.
2. If geometry is feasible, restore IPC and solve a fresh contact-on equilibrium
   at the fitted jaw angle, then recompute all gradients. Collision-off gradients
   and warm adjoints cannot carry over to the changed mechanical problem.
3. Start joint activation and pose fitting only after that solve passes the
   usual force and contact gates. Joint Adam moments and convergence history
   start fresh. The physical model switch does not count as an accepted fit step.

Intersections, inadequate clearance, or a failed contact-on equilibrium produce
`requires_contact_recovery`, retain the accepted pose checkpoint, and stop the
run. The implementation does not silently run joint fitting without collision
or repair penetrations by changing the anatomical reference.

## Reproduction

From this experiment directory:

```bash
CHERRIES_NAME='Collision-off pose initialization then contact-on joint fitting' \
CHERRIES_TAGS='expression-fit,pose-first,collision-off-initialization,fixed-material,pncg' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/93-fit-expressions.py \
  --output-dir data/expression-fitting-007 --pose-first true \
  --pose-collision false --calibration-source data/expression-fitting-001
```

The existing calibration supplies the unchanged strong smoothness weight for
the later contact-on joint problem; its value is irrelevant to the exactly-zero
activation field in pose-only fitting. The new off/on branch is tested by
`105-check-pose-collision-switch.py`. These CPU tests validate state and runtime
selection; actual anatomical performance and convergence are measured by the
new run rather than inferred from those tests.

## Results

Validation passed in `data/pose-collision-switch-check-002/summary.json`,
`data/pose-first-check-003/summary.json`, and
`data/projected-descent-check-004/summary.json`. The first receipt exercises the
actual runtime switch and contact handoff with mocked equilibrium/installer
construction. It verifies contact object/runtime consistency, cleared adjoints,
full geometry extension including skull and eyes, intersection and sub-buffer
rejection, unchanged checkpoint bytes on rejection or failed contact solve,
and fresh contact-on deformation and gradients on success.

All three receipts bind runner SHA256
`6d94b0d9c0fa5f0f53fc94e12d854bc54e4db6445fcfa05e78b02cd438a42b7c`.
The new collision-off runtime SHA256 is
`27dcc80f0b950653193e6a1aac8ae66e29a2d8b44292a83608c90264a2232170`.
Ruff and compilation checks passed for all changed runner and review files.

Run 006 was intentionally stopped after MouthOpen update 11 at 0.34375 degrees
and RMS 7.403999884 mm. Its accepted checkpoints are preserved and hash-recorded
in `data/expression-fitting-006/external-interruption.json`.
Run 007 launched in the transient `apple-expression-fit.service` using the
command above. Starting the run does not establish convergence or speedup.
The live review identifies collision-off initialization explicitly and keeps its
pose convergence separate from contact-on joint convergence.

[Comet run](https://www.comet.com/liblaf/apple/cf7d8ccde0d84cd29ff6061f9d055c6f) · Live review (private preview omitted).

Startup verification confirms the actual pose runtime has collision disabled, muscle activation is identically zero, and all implementation hashes match the tested protocol. The collision-off zero-pose equilibrium took 1,202 PNCG iterations / 32.94 seconds and reached force 1.51513e-10, below 1.51920e-10. Its initial RMS is 7.629727 mm. See `data/expression-fitting-007/startup-verification.json`. The first 1-degree proposal is being solved; this warmup timing alone is not an angular-progress speedup measurement.

The first full 1-degree proposal was accepted without backtracking. RMS fell
from 7.629727 to 6.975478 mm
(8.57%). Its forward equilibrium took
5,214 PNCG steps / 143.06 seconds,
followed by 14.92 seconds for the adjoint. Total startup,
zero-pose relaxation and first update elapsed 213.05 seconds.
Muscle activation remains exactly zero. The result is collision-off and the
pose stage has not converged. See `data/expression-fitting-007/first-pose-update.json`.
This establishes that the old 0.03125-degree boundary-CCD restriction is absent
in this trial; it is not a contact-validity claim or an equal-load speed benchmark.

## Completed pose stage and blocked contact handoff

The run stopped as `requires_contact_recovery` after 10 accepted
pose updates and 104.04 minutes. MouthOpen reached
9.447061 degrees and RMS 3.262435 mm,
a 57.24% reduction from its
collision-off initial RMS. Muscle activation remains exactly zero.

`pose-stage.json` records first-order jaw stationarity:
projected mapping 7.74320975e-05 versus tolerance
0.001, with resolved primal objective correction
3.96222883e-09. This is not full joint
stationarity or a global-optimum claim.

`contact-handoff.json` reports soft-rigid intersections. Consequently no
contact-on joint solve was started; all other 35 expressions remain queued.
The current state also has 193 inverted tetrahedra
and minimum det(F) -4.711457. Inversions were diagnostic-only,
as requested; intersections were the reason the handoff stopped. The existing
check does not localize whether intersections involve skull, mandible or eyes.

The 10 accepted pose proposals used 219,611
PNCG steps and 100.32 forward minutes;
adjoints took 2.67 minutes. The 6-degree
solve alone required 92,156 steps / 42.10 minutes, and the final solve required
37,114 steps / 16.94 minutes. Removing collision reduced per-iteration cost,
but convergence at some larger deformations remained expensive. These traces
do not establish the precise conditioning or deformation-instability cause.

Both the fitting and review-publisher services have exited. The terminal review
is served over tailnet and shows the final 9.447-degree / 3.262-mm state with
collision-off and recovery-required labels. Any continuation to joint fitting
must first produce a contact-valid equilibrium without reusing collision-off
gradients. The current frozen neutral and accepted checkpoints are preserved.
