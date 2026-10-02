# MouthOpen continuation with bounded joint increments

Run013 continues the exact independently audited run012 checkpoint, including
both Adam counters 268 and all first/second moments. Diagnostic172 showed that
a bounded joint strain-and-pose proposal could pass actual collision-on
equilibrium and the original acceptance policy. Its state is not adopted.

## Method and acceptance

The new opt-in `projection_bounded_joint` path in script130 computes reduced
determinant gradients for positive retained cells below determinant 1e-4 and
the previously limiting positive cells 18514 and 155249. The model builder
checks these gradients against a coupled directional tangent, preserves the
source state and records all coefficients. A separate damped residual response
supplies the affine determinant intercept; it is not inserted into the seed.

For each line-search alpha, the QP finds an actual joint increment near the
scaled Adam direction in its diagonal metric. The strain increment bound
`abs(delta_q) <= 0.01` is inside the QP. Pose is unrestricted by extra increment
caps. Selected determinant predictions must exceed the numerical margin 1e-6,
and predicted objective decrease must reach 0.1 of the original scaled Adam
decrease. The correction has a declared metric trust limit of twice the Adam
increment norm. Primal, dual, stationarity, complementarity and gap checks must
pass. Unresolved numerical certificates stop visibly with saved inputs.

These are proposal rules. Acceptance still requires coupled-seed CCD, actual
collision-on nonlinear equilibrium, Armijo decrease, at most 100 retained
inverted tetrahedra and at most 1e-4 inverted rest-volume fraction. The original
policy neither fixes the identity of inverted cells nor imposes a minimum
determinant floor. The extra old-positive-cell screen that stopped diagnostic171
is not added to physical acceptance.

Exact saved skin pre-strain, IsFixed-only constraints and exclusion only of
all-four-fixed tetrahedra are preserved. Internal force target is 1e-9
(0.001 N); original physical acceptance remains 1e-8 (0.01 N). Relative adjoint
and predictor shifts are 1e-5. Q/pose learning rates are 0.002/0.1. The first
trial alpha is 0.125. The total active-strain parameters remain unrestricted.

Actual increments are applied once, with Armijo slope recomputed after adding
them to the controls. Proposed Adam moments are committed only after an
accepted step. Older wrappers default to the unchanged legacy path.

## Evidence before launch

- Independent audit012: RMS 1.9802796612356741 mm, force
  0.0008540888196111258 N, 100 retained inversions.
- Diagnostic172: 140 PNCG plus 5 Newton iterations; RMS
  1.9783457028037603 mm, force 0.0008842308565484633 N, 100 inversions,
  inverted rest-volume fraction 1.2527892839856141e-5. Collision and Armijo
  passed. See `172-mouthopen-bounded-physical-gates.md`.
- CPU helper replay reproduces diagnostic171 increments within 7e-18, including
  810 active component bounds. Immutability, alpha scaling, unscaled residual
  intercept, trust-limit failure and changed-cache rejection checks passed.
  Receipts: `tmp/joint-projection-cpu-001/`.
- Ruff and compilation passed for script130, script173 and both new helpers.

## Command

Run from this experiment directory:

```bash
mkdir -p tmp/run013-runtime
TMPDIR="$PWD/tmp/run013-runtime" \
CHERRIES_NAME='MouthOpen bounded joint continuation' \
CHERRIES_TAGS='mouthopen,inverse,continuation,bounded-joint' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/173-continue-mouthopen-bounded-joint.py \
> tmp/173-continue-mouthopen-coupled-013.log 2>&1
```

Output: `data/inverse-mouthopen-coupled-013`. Exact job identity and source
snapshot are recorded at launch. This run is not inverse-converged; a finished
stationarity candidate still needs independent physical and reduced-objective
stationarity audits. The full-surface preview remains the audited run012 until
a new endpoint passes audit.

## Launch

Launched 2026-09-30 at about 01:32 UTC. PID 934029, process start ticks 4496593,
boot `fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 45386. The exact
receipt is `data/inverse-mouthopen-coupled-013/job.json`. Project filesystem
had 1.4 TB free; no other numerical GPU process was active before launch.
An independent review found no integration blocker. All sources are copied
into the run's `sources/` and hash-recorded in `protocol.json`.

Comet: <https://www.comet.com/liblaf/apple/9ae63d1e7f7a444e8b46d06cfa1a3f20>

The process exited with code 1 before the first inverse iteration. The
fixed-state Lagrangian derivative check for normalized translation x gave
relative errors 0.474048, 0.0407565 and 0.00195874 at steps 1e-4, 1e-5 and
1e-6. The last value exceeds the unchanged 1e-3 agreement threshold. Other
coordinates and strain passed. There is no run013 optimizer checkpoint, and
no joint QP was run. Process and Cherries have exited; the failure receipt is
`completion-verification.json`.

The strongly shrinking error warrants finer finite differences before
concluding that the derivative is wrong. Additive run014 retains source012
and its original optimizer history while extending only the check stencil
with 3e-7 and 1e-7; the agreement threshold and physical policy are unchanged.
See `174-mouthopen-gradient-check-resolution.md`.
