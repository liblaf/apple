# MouthOpen continuation toward convergence

The user requested continuation until convergence on 2026-09-30 (Asia/Shanghai).
Run `inverse-mouthopen-coupled-003` continues the independently audited run002
endpoint. This report is a live operations record, not a convergence certificate.

## Starting evidence and fixed physical policy

- Initial RMS: 6.304117916095423 mm; normalized loss 0.6835100895190149.
- Starting jaw magnitude: 1.1250182962031285 degrees; translation 1.5215500715542742 mm.
- Starting free force: 0.008877356288276796 N, below 0.01 N.
- Original `IsFixed` defines constraints; all lip vertices remain free.
- Skin pre-strain, thickness and modulus are preserved exactly from the corrected neutral.
- Bulk excludes 2,249 all-fixed tetrahedra; 1,144,268 retained cells participate.
- At initialization 6 retained cells are inverted, minimum J = -1.1970693369899073.
- The existing allowance is at most 100 inverted retained cells and at most 0.0001 of total retained rest volume. This is an approximation, not inversion-free physics.
- Collision and CCD remain enabled with the existing selected soft/rigid collision mesh; lip-to-lip collision is outside this mesh policy.
- Physical free-force tolerance remains 1e-8 in solver units = 0.01 N.

See `145-mouthopen-coupled-fit.md` for prior runs and independent endpoint audit.

## Continuation and convergence monitoring

The four Adam moment tensors are copied from run002. Its saved legacy iteration
is 8, so the next update uses Adam bias-correction step 9. This is correct despite
20 accepted updates across runs001/002, because run002 intentionally reset moments.
The new run records both local iteration and optimizer step. No physical state is reset.

The first line search starts at alpha 0.125, twice the last accepted alpha 0.0625.
Subsequent searches start at `min(1, 2 * last_accepted_alpha)`. CCD, inversion gates,
nonlinear forward convergence and Armijo sufficient decrease still accept each update.

The monitor records the unrestricted gradient measure
`D = 0.02 * sum(abs(g_raw6)) + 0.1 * sum(abs(g_normalized_pose))`, independently of
CCD step clipping. At initialization D = 0.10955923180373633, dominated by the jaw
contribution 0.10873322972614155. It requests an independent audit only when:

1. Ten accepted updates have accumulated, each with a valid physical forward under the declared inversion allowance.
2. Loss change across those ten updates is at most 1e-6 relative to the window loss.
3. D is at most `1e-8 + 1e-3 * D_initial`.

Passing these conditions sets `stationarity_candidate_requires_audit`, never
`inverse_converged=true`. The shifted adjoint remains an approximation. A fresh
unshifted or explicitly labelled lower-shift adjoint and feasible re-equilibrated
objective probes must support any reduced-objective convergence claim. The existing
fixed-state Lagrangian pullback check only checks derivative implementation.

There is no wall-clock budget. The 10,000-update ceiling is a safety limit; reaching
it is not convergence. A line-search stall also requires diagnosis, not relabelling.

## Reproduction and live process

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='MouthOpen coupled continuation toward convergence' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,collision,coupled-predictor,continuation,convergence' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/148-continue-mouthopen-coupled.py \
  > tmp/148-continue-mouthopen-coupled-003.log 2>&1
```

The normal Cherries profile is used with commits disabled. Comet:
<https://www.comet.com/liblaf/apple/36842c3de6464a28b260c71a9ebc61e3>

Launched Python PID 179968 (verify command and process start before any signal).
Tool session 75445 contains its process completion. Progress and checkpoint files:
`data/inverse-mouthopen-coupled-003/`. Inspect `summary.json`, `progress.jsonl`,
`trials.jsonl`, `protocol.json` and per-predictor receipts. Sources are snapshotted
inside the run directory. Do not edit sources loaded by this running process.

Heartbeat `continue-mouthopen-until-convergence` monitors this chat every ten minutes.
Do not launch duplicate GPU work. If a numerical blocker occurs, preserve the saved
endpoint and diagnose before continuing in a new additive run. Do not loosen physical
acceptance policies to obtain a convergence label.

## Validation and delivery

Before launch, Ruff passed and nine focused tests passed: optimizer continuation,
convergence monitoring, `IsFixed` boundary policy and all-fixed tetrahedron filtering.
An independent source review found no serious continuation or monitor issue. The actual
run revalidated the starting free force, contact, tet policy, source bindings, fixed-state
gradient pullback and exact zero-update predictor.

Live progress is being published at
`PRIVATE_PREVIEW_URL`.
The preceding independently audited full-surface view remains at
`PRIVATE_PREVIEW_URL`.

After the numerical process and Cherries shutdown finish, audit the saved endpoint
using `src/146-audit-mouthopen-coupled.py --run-dir data/<finished-run>`.
Render the full original tet boundary with bones and eyes using 147 after the audit.
Its current `parent_run_dir` supports one parent: preserve truthful lineage when
combining the entire run001/002/003 history. Verify published assets over tailnet.
Only pause the heartbeat when convergence is independently supported, or when a
concrete physical/model decision needs user input.

### First live verification

After four new accepted updates (24 total across runs001/002/003), RMS fell to
6.010769759675241 mm, free force was 0.008347285896860504 N, jaw magnitude was
1.3777061307151985 degrees / 1.8679679854089615 mm, and 13 retained cells were
inverted. These are current-run accepted checkpoints, not independently audited
finished results. The live page and both JSON receipts returned HTTP 200 after
publication. Browser visual inspection timed out, so only source and HTTP validation
are confirmed. The earlier audited full-surface page remains linked.

## Inversion boundary and continued search (2026-09-30 02:30 CST)

Read `data/mouthopen-continuation-state.json` first for the current run; it supersedes
the original run003 process details above.

Run003 reached the 100 retained-cell inversion count cap at RMS 4.138547042502408 mm.
The other cap, rest-volume fraction, was still below its limit: 3.905030215572624e-5
versus 1e-4. The next cell to invert was original tet 1126965, with two fixed
Mandible vertices and two free vertices; its J was approximately 9e-13 during
diagnosis. It is not one of the excluded all-fixed tetrahedra.

The joint line search shrank to floating-point-sized updates while the gradient
measure remained about 0.055435, over 500 times the convergence-monitor threshold.
The process was interrupted with a verified-PID SIGINT after saving local iteration
58. Cherries shutdown completed, exit code 130; `stop-request.json` records the reason.
The terminal status is `stopped_at_inversion_constraint`, never convergence.

A fresh independent run of 146 audited this endpoint successfully: force
0.009619241586533078 N; configured contact feasible; original IsFixed and exact skin
material fields preserved; 100 retained inversions, minimum J -5.1796659860734575.
The full-surface view, bones/eyes and full run001→002→003 curve history are now in
`data/review-mouthopen-coupled-003-complete`. The renderer verifies checkpoint and
endpoint hashes and RMS continuity for every parent. Its 78 accepted updates include
the tiny terminal updates and should not be interpreted as 78 meaningful movements.
All 15 published page/JSON/image/VTP files were byte-verified over the tailnet. The
front target-versus-fit image was visually inspected; the mouth is still materially
less open than the target.

Run004 tested an explicit strain-only phase with the jaw exactly fixed. Separate
Adam counters preserve the frozen pose optimizer; the strain learning rate was
0.002. Entry point: `src/150-continue-mouthopen-blocks.py`; log:
`tmp/150-continue-mouthopen-coupled-004.log`. All 17 initial line-search trials hit
the predictor inversion cap; zero new updates were accepted. It exited normally
with `line_search_stalled`, not convergence.

The next experiment projects the six-dimensional jaw proposal using coupled-tangent
determinant sensitivities. It retains every currently positive cell with J<=0.05
in a linearized feasibility constraint, with target J>=1e-6. Six small normalized
pose probes and one strain probe use the same damped physical Hessian. These are
sensitivity probes, not accepted equilibria. Every actual candidate still passes
CCD, the full nonlinear forward solve, and the unchanged count/volume acceptance
gates. An infeasible or non-descent projection is reported explicitly. No inversion
allowance, skin material, collision policy, or physical force tolerance is relaxed.

CPU checks cover exact joint Adam equivalence, frozen block/counter behavior, an
active linear projection, an unchanged already-feasible direction, and explicit QP
infeasibility. The determinant helper uses the same matrix orientation and source
cell mapping as the physical acceptance check. Exact saved normalized pose values
are restored to avoid division/multiplication roundoff at a nearly singular cell.

### Projected continuation admitted

Run005 is active through `src/151-continue-mouthopen-projected.py`, continuing the
exact audited run003 checkpoint. It uses q learning rate 0.002, original pose
learning rate 0.1, no explicit rotation/translation increment clipping, and initial
line-search alpha 0.125. Its first projection constrained 23 previously positive
retained cells. The strain-only full tangent put the limiting J at -8.04e-8;
the projected combined linear model put the minimum at 1e-6. These are proposal
metrics only, not equilibrium claims.

The first accepted full forward reduced RMS from 4.138547042502408 to
4.102186473137868 mm and the inversion count from 100 to 97. Force was
0.009869848823638233 N. Jaw magnitude changed to 3.278834358064124 degrees and
4.39997973699208 mm. This restores useful feasible progress without changing the
physical gates. The current run is not independently audited or inverse-converged.
The live page points to run005 while the full-surface page shows audited run003.

Run005 log: `tmp/151-continue-mouthopen-coupled-005.log`; process identity is in its
`job.json`, and the active-run pointer is `data/mouthopen-continuation-state.json`.
The heartbeat reads that pointer. Run004 stopped normally with zero accepted steps.
Projection source review found no safety blocker. General use with a clipped pose
would need re-projection after clipping; the current run explicitly has both pose
increment caps set to null and therefore applies no such clipping.

### User-authorized initializer comparison

Run005 was cleanly interrupted at accepted update 61 for a serialized comparison
of the coupled tangent and a collision-off tissue estimate followed by push-out.
Its optimizer counters are 127 for both blocks; the exact checkpoint is retained.
A fresh 146 audit passed (RMS 3.0464148747443907 mm, force
0.00947097518116147 N, 91 retained inversions). The temporary benchmark entrypoint
is `152-test-mouthopen-collision-off-seed.py`. Read the `initializer_test` field
of `data/mouthopen-continuation-state.json` for its live process. Do not launch
competing GPU continuation while this process is active. The original optimizer
will resume in additive run006 through `153-continue-mouthopen-after-seed-test.py`
after the comparison. Diagnostic endpoints do not replace the live fit.
See `152-collision-off-initializer-comparison.md` for the test protocol.

The completed initializer comparison admitted neither tested pose. The corrected
ray-containment variant left triangle intersections and 346 / 1,756 retained
inversions after push-out (cap 100). No collision-on corrector was reached.
The rejected states are published at `/isfixed-neutral/initializer-test/`.

Run006 now runs `src/153-continue-mouthopen-after-seed-test.py`, continuing the
independently audited run005 checkpoint with both Adam counters 127 and all
moments preserved. Its initial trial alpha 0.5 continues the adaptive rule from
run005's last accepted alpha 0.25. No physical or optimizer learning-rate policy
changed. The first accepted update used alpha 0.125, reduced RMS to
3.036463255343473 mm, and had force 0.009740429699486668 N with 91 retained
inversions. This is progress, not inverse convergence.

The state pointer records the exact process; log:
`tmp/153-continue-mouthopen-coupled-006.log`. Live page binds run006, and the
audited full-surface page now shows run005, including its entire parent history.
The heartbeat remains active and reads the pointer.

### Objective-aware projected continuation (run007)

Run006 finished normally after 87 accepted updates at RMS 2.145552398622598 mm,
with `projected_direction_not_descent` on attempted update 88. Its saved physical
endpoint passed independent audit: force 0.008432883425139663 N, 97 retained
inversions, inverted rest-volume fraction 1.2549996711732493e-5. The full-surface
preview now shows this audited endpoint. This was not inverse convergence.

The nearest feasible pose projection turned the joint predicted derivative
positive. An augmented QP adds a normalized joint-descent halfspace while keeping
the strain proposal and existing determinant constraints. A CPU reconstruction
of the failed proposal confirmed a feasible descent direction. Run007 resumes
the exact run006 checkpoint and both Adam counters 214 through
`src/156-continue-mouthopen-descent-projected.py`. Physical acceptance, moments,
learning rates and adjoint shift are preserved. The next trial starts at alpha 1.

Read `docs/156-mouthopen-descent-projection.md` for evidence, limitations and
reproduction; read the state pointer for the exact active process. Run007 has its
own TMPDIR on the project filesystem because the system `/tmp` filled during
run006 and disrupted Comet replay. The live page and heartbeat now track run007.

### Fixed-control refinement diagnostic

Run007 was stopped after 23 updates (Adam237/237) when accepted steps shrank to
zero-correction seeds just below force tolerance, while corrected trials exceeded
the count cap with 101 inversions. The stopped endpoint passed independent audit
and is now the published full-surface preview. It is not inverse-converged.
`src/157-polish-mouthopen-equilibrium.py` now tests fixed-control refinement to
internal force 1e-9, with original admission gates unchanged. Read the active
pointer and `157-mouthopen-equilibrium-polish.md`; diagnostic states are not
automatically adopted. Run007 is tested first and run006 only if needed.

### Continuation from refined run006 (run008)

The fixed-control diagnostic finished normally. Run007's tighter equilibrium
had 101 retained inversions and was rejected. Run006's refinement reached force
0.0009667553070054522 N with the same 97 inverted cells, so it supplies the
initial displacement for additive run008. Original run006 controls, moments and
Adam counters 214/214 are preserved. The refined initial RMS is
2.141789908845402 mm; this correction is not an inverse update.

Run008 uses `src/158-continue-mouthopen-refined-equilibrium.py`, internal forward
atol 1e-9, and the original physical acceptance contract 1e-8, 100 inverted
retained cells and 1e-4 inverted rest-volume fraction. Collision, exact saved
skin pre-strain, IsFixed-only constraints and all-fixed exclusion are preserved.
The first four updates were accepted, reaching RMS 2.1329402242604956 mm, force
0.0009898191307541105 N and 98 inversions; updates 3 and 4 used full alpha 1.
This is resumed progress, not verified convergence.

Read `157-mouthopen-equilibrium-polish.md` for diagnostic evidence and run008
reproduction. The state pointer, live page and active heartbeat now track run008;
the published independently audited full-surface endpoint remains run007.

### Run008 stopped for determinant sensitivity calibration

Run008 accepted 19 updates (Adam counters 233/233), reaching RMS
2.0953739184741136 mm. At 100 retained inversions, corrected trials again
produced 101 inversions while tiny accepted trials did no correction. The run
was stopped with verified SIGINT; process and Cherries finished with exit 130.
Its saved endpoint passed fresh independent audit: force
0.0009906641494569983 N and inverted rest-volume fraction
1.1758180236951449e-5. This is not inverse convergence.

The next diagnostic compares determinant sensitivities at two numerical shifts
on this frozen checkpoint, then tests a changed projected direction. It does
not change the physical gates or tighten the force threshold again. Read
`159-mouthopen-determinant-calibration.md` and the active state pointer.

The calibration completed: relative shift 1e-4 reduced the strain tangent's
unshifted residual from 11.5% to 2.52%. A new projected quarter-step passed
the same gates with 100 inversions and lower loss. Its limiting-cell determinant
prediction closely matched the candidate, but no nonlinear correction was
needed; this is not a stationarity certificate. The candidate is not adopted.

Run009 now continues from the exact audited run008 checkpoint and both Adam
counters 233 through `160-continue-mouthopen-calibrated-tangent.py`. It uses
relative adjoint/predictor shift 1e-4, analytic determinant projection with
probe epsilon 5e-5, and initial alpha 0.25. The original physical policy,
internal force target 1e-9 and optimizer learning rates are preserved.

### Run009 projection failure

Run009 finished with `pose_projection_failed` at attempted update 22, after
21 accepted updates. Both counters are 254, and RMS is 1.9995158395305934 mm.
The numerical process and Cherries finished with exit 1. Its saved endpoint
passed fresh independent physical audit and is now the full-surface preview.
This is not inverse convergence.

The recorded linear model has a geometrically feasible zero-pose direction
with a negative strain slope, but the failed QP demanded a larger joint
decrease. Full failed matrices were not saved, so a new diagnostic will
reconstruct them and distinguish infeasibility of that target from numerical
SLSQP failure. Read `161-mouthopen-qp-feasibility.md` and the active pointer.

### Attainable-descent continuation (run011)

Diagnostic161 confirmed the run009 target was infeasible and tested a smaller
target based on a descending zero-pose witness. Run010 accepted three corrected
updates, then stopped because a positive determinant constraint required
nonzero pose motion. Its saved endpoint was independently audited and published:
RMS 1.9914515183673682 mm, force 0.0009277460627128851 N, 100 retained
inversions and both counters 257. This is not inverse convergence.

The complete failed matrix showed a certified nonzero-pose descent direction.
Run011 resumes exact audited010 controls, displacement and optimizer history,
using `163-continue-mouthopen-attainable-descent.py`. If the original joint
target is infeasible, a geometry-only LP certifies the best available joint
slope; half its negative value defines an attainable target. Actual CCD,
nonlinear force equilibrium, Armijo and original physical gates remain decisive.
Read `163-mouthopen-attainable-descent.md` and the state pointer for current
process/receipts. No physical allowance was loosened.

## Run012: affine residual-aware trial projection

Run011 was independently audited at1.9818275442754796 mm RMS after its late
line-search stall; full preview published. Diagnostic164 captured original
cell18514 crossing zero during correction despite a valid seed. Diagnostic165
output003 at relative shift1e-5 passed actual correction using a trial-specific
pose proposal that guards both seed and affine residual determinant predictions.
Its diagnostic candidate was not adopted.

Run012 uses src/167-continue-mouthopen-affine-residual.py, exact audited011
checkpoint/moments/counters267, relative adjoint/predictor shift1e-5 and initial
alpha0.125. Each alpha receives its own actual pose increment; the residual
intercept stays unscaled and out of derivative probes/seeds. Physical policy
is unchanged. First accepted update passed60 PNCG+5 Newton, RMS1.9802796612 mm,
force0.00085408882 N and100 inversions; this is progress, not convergence.
See docs/164-mouthopen-corrector-determinants.md,
docs/165-mouthopen-affine-corrector.md and
docs/167-mouthopen-affine-continuation.md for evidence and reproducibility.
The active pointer/job identity is authoritative for monitoring.

## Bounded joint continuation (run014)

Run012 ultimately stalled and passed independent audit at RMS
1.9802796612356741 mm, force 0.0008540888196111258 N and 100 inversions,
with both Adam counters 268. Diagnostics168/169 showed that repairing a
pose-only determinant forecast did not provide a useful corrected step.
Diagnostics170/171 developed full reduced strain-and-pose determinant gradients
and a bounded joint proposal. Diagnostic172 passed nonlinear equilibrium and
original physical gates at RMS 1.9783457028037603 mm; its state was not adopted.

Run013 attempted this method but stopped before any inverse update on a
fixed-state finite-difference check. Run014 keeps the same audited012 source
and optimizer history, adding smaller derivative-check stencils while retaining
the original agreement threshold. Startup passed and the first two actual
nonlinear updates reached RMS 1.9720125239661843 mm. This is progress only.
The current active pointer identifies run014 and its exact process.

Read reports170-174 for evidence. The bounded joint proposal uses numerical
component increment limits and certified QP/dual residuals. The original
physical policy remains count at most 100 and rest-volume fraction at most
1e-4; it does not fix inverted-cell identity or impose a minimum-J floor.
Internal force target is 1e-9, original physical acceptance 1e-8. The audited
full-surface preview remains run012 until a newer finished endpoint is audited.

## Run015: stable dual decrease

Run014 finished after 22 accepted updates when the dual line search lost a
tiny valid objective decrease to scalar subtraction. Its saved endpoint passed
independent audit at RMS 1.9621245226690815 mm, force
0.000996781585126526 N, 100 inversions and counters 290/290. The audited
full-surface preview has been updated to run014, with all 15 assets verified.

Run015 resumes that exact checkpoint and optimizer history. It evaluates the
same clipped-quadratic dual objective difference directly, retaining Armijo
and every original certificate assertion. Frozen failure replay, regression
tests and independent review passed. At the 02:14 UTC monitor, run015 had four
accepted corrected updates, RMS 1.9484776261612966 mm and counters 294/294.
It remains running and has not established inverse convergence. See report175
and the state pointer for method, process identity and current receipts.
