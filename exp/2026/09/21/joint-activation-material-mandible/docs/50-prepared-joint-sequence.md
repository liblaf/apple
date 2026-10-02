# One-shot prepared joint sequence

**Current admission status:** neutral preparation is incomplete. The constant
25% stage became stationary outside the shape budget; the Spatial80 branch was
stopped at segment 002 update 111 when the user required complete source skull
collision. The partial-FEM collider is superseded. A launch guard now rejects
the old surface selection; full-skull initialization and validation must finish.
No full-target neutral, converged control,
or final joint trajectory has been produced. The sequence must remain gated.

`src/50-run-prepared-joint-sequence.py` closes the manual gap after neutral
preparation. It runs one foreground sequence:

1. full-target contact-enabled jaw preflight;
2. strong-smoothness calibration;
3. fixed-shared `control_converge`;
4. control exports with `31-render-final.py` and `46-render-optimization-state.py`;
5. control review-site refresh with `43-build-review.py`;
6. the up-to-100-accepted-update/12-hour `joint_trend`;
7. final exports and a review-site refresh containing the actual control and joint runs.

There is no scheduler, background process, retry loop, or polling service. A
nonzero subprocess exit, invalid receipt, exhausted unconverged control budget,
or lineage mismatch stops the sequence immediately. The final trend begins only
after the control summary says `success`, `preparation_complete`, and
`inverse_converged`, and its terminal checkpoint hash matches that summary.

The sequence defaults to the validated Newton-CG contract
(`linear_rtol=0.001`, 12 Newton steps, no fallback), the fixed `0.05`
dimensionless activation-neighbor RMS budget, up to 1,000 control updates/24
hours, and up to 100 accepted final updates/12 hours. The wall budget may stop
the final trend sooner after at least one numerically valid accepted joint
update. It preflights the converged
100%-proxy neutral checkpoint and both Newton validation receipts before
starting calibration. It does not loosen any gate in `30-joint-pilot.py`.

Use a fresh `--sequence-dir`; previous sequences cannot be overwritten. Required
arguments are the actual `--neutral-checkpoint` and the matching
`--neutral-lineage-visual-dir`. The exact launch command will be recorded only
when a completed 100% checkpoint exists; earlier commands naming nonexistent
50%/100% constant-continuation outputs are not valid launch instructions.

The jaw preflight runs `28-run-jaw-preflight.py` with
`--admission-mode final_launch_ready`. Its receipt must bind the exact full-target
neutral, inputs, contact specification, rigid-bone CCD validation, shared basis,
production tolerances, and current sources. Its v2 receipt must contain one zero test,
12 uniquely labeled extreme-axis diagnostics, the retained 1° diagnostic,
and the predeclared +0.01° world-x rotation test,
all starting from identical neutral-seed copies. Zero and nonzero cases require
successful forward/contact/FEM checks; zero must reproduce its seed within
`1e-6 m` maximum Euclidean nodal difference. This 1 µm geometry QA budget is
independent of the unchanged force tolerances; surface and volume node RMS
differences are also reported. The old v1 displacement check incorrectly reused
the force absolute tolerance and is not admitted. Extreme-axis and 1° failures
remain diagnostics. A diagnostic-only receipt
can never authorize calibration. The separate rigid-bone evidence defaults to
`data/rigid-bone-ccd-validation-003/summary.json` and is passed explicitly to
preflight and all inverse stages.

If no `--neutral-run-dir` is supplied, the neutral checkpoint's parent
directory is used. Neutral shape and contact galleries default to `shape-visuals`
and `contact-visuals` beside that checkpoint, as produced by the neutral
continuation runner. Their receipts must match the admitted neutral checkpoint
hash; explicit `--neutral-visual-dir` and `--neutral-contact-visual-dir` paths may
select matching galleries elsewhere. The required lineage gallery must also
prove the same terminal checkpoint hash. Generate it with `48-render-neutral-lineage.py`
after full neutral convergence. Existing reference contact, preparation,
and source-bone galleries remain supplied by `43-build-review.py` defaults. The generated
control/final `31` and `46` directories are passed explicitly as optimization
galleries. The review site is rebuilt in `data/review-site`, which the existing
server reads dynamically at `PRIVATE_PREVIEW_URL`; no server
restart is performed.

Every subprocess gets a distinct `CHERRIES_NAME` and `CHERRIES_TAGS`. The fresh
sequence directory contains `commands.json` with exact argv arrays, display
commands, names, tags, start/end times, and exit codes, plus a separate combined
stdout/stderr log for each command. `manifest.json`, `summary.json`, or
`failure.json` records the neutral, calibration, control, and final hashes and
the terminal status.

`source-hashes.json` freezes the experiment, Apple, and tensor-reference Python
sources. Every subprocess checks those hashes before and after execution; code
changes stop the sequence. The review receives the exact admitted contact and
Newton derivative receipts. Its deliverable flag stays false until full neutral
and control preparation and a successful joint trajectory with at least one
accepted update exist.

For a spatial neutral checkpoint, admission also verifies the exact basis and
audit hashes, its recorded CPU and full-face derivative receipts, the
`0.5 * 100` roughness convention, all 16 finite-difference checks, and both
mechanical and total-objective errors below 2%. The derivative implementation
hashes must still match. These receipt paths come from the neutral protocol;
the sequence does not silently select the newest run directory.
