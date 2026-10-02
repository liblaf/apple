# MouthOpen continuation with accurate shifted linear solves

Run018 resumes the exact independently audited run017 checkpoint, controls,
displacement, four Adam moment tensors and both counters 373. The diagnostic
state is not adopted. Collision, skin pre-strain, IsFixed constraints, retained
tetrahedra, objective and acceptance gates are unchanged.

## Evidence for the numerical change

Run017 finished at attempted update 61 because the selected cell 18514 failed
the combined determinant derivative comparison. Its accepted endpoint remains
physically valid: positional RMS 1.8455912433462538 mm, force
0.0007533984389691991 N, 100 retained inversions and inverted rest-volume
fraction 1.2676763228688024e-5. The full objective is 0.1440810838419122.
See run017 `independent-audit.json`, `completion-verification.json` and
`independent-lineage-audit.json`.

Frozen diagnostic200 tested the same controls, displacement and proposed Adam
direction. All six signed probes at linear relative tolerance 1e-7 failed the
unchanged comparison `1e-7 + 0.005 * max(abs(adjoint), abs(tangent))`.
At tolerance 1e-9 all six passed. Probe magnitudes were 5e-5, 1e-5 and 1e-6.
The independently tightened row18514 adjoint differs from its archived value
by only 4.48e-9; its shifted residual is 8.35e-10. Root also compared this strict
adjoint against every strict tangent, and all passed. This supports improving
linear accuracy while retaining the original finite probe epsilon 5e-5.

Diagnostic200 finished normally, including Cherries shutdown, before run018
launched. Its report is `200-mouthopen-joint-probe.md`; root's additional
comparison is `tmp/200-root-strict-comparison.json`.

## Configuration and deadline

`src/201-continue-mouthopen-accurate-tangents.py` sets both adjoint and predictor
linear tolerances to 1e-9. Relative shifts remain 1e-5. Fresh gradients use the
tighter solves; all source moments are retained. Initial trial alpha 0.015625
is twice the source endpoint's accepted 0.0078125, matching the adaptive next
trial rule. It is a trial proposal, not a change to saved controls.

The objective remains normalized position L2 plus target normal matching plus
same-muscle activation smoothness. Normal and smooth weights remain
9.039348924363765 and 6.5083312255418886e-6. Raw roughness need not decrease
independently when the weighted total decreases; all components are reported.

Original physical force acceptance remains 1e-8 internal units (0.01 N),
forward target 1e-9 (0.001 N), retained inversion count at most 100 and inverted
rest-volume fraction at most 1e-4. The joint QP, numerical determinant margin,
strain-increment box, trust bound, derivative comparison and all certificate
thresholds remain unchanged. Actual CCD, nonlinear equilibrium and Armijo
decide acceptance.

Runner130 now computes the earliest relative or absolute deadline before
setup and rejects evaluation after setup exhausts it. Run018's absolute cutoff
is 2026-09-30 05:55 UTC / 13:55 Asia/Shanghai. Protocol records the deadline.
This implements the user's computation deadline, not inverse convergence.
Both collision-on expressions will be rendered after computation ends.

Runner130 also records objective continuity when the source objective and all
terms exactly match. Only a true change produces an objective-transition
marker; run017 retains its original L2-to-mixed transition in the lineage.
Old L2 and fresh-neutral initialization paths do not gain a parent-protocol
dependency.

Focused CPU checks covered deadline selection and expiration, timezone
requirements, exact objective continuity and genuine objective transitions.
Root independently checked configuration and source audit hashes. Ruff and
compilation passed. Receipts are `tmp/201-metadata-deadline-cpu-tests.json`
and `tmp/201-root-continuation-preflight.json`.

## Command and lifecycle

Run from `exp/2026/09/23/new-neutral`:

```bash
mkdir -p tmp/run018-runtime
TMPDIR="$PWD/tmp/run018-runtime" \
CHERRIES_NAME='MouthOpen accurate tangents before deadline' \
CHERRIES_TAGS='mouthopen,inverse,normal,smoothness,accurate-tangent,deadline' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
.venv/bin/python -u \
src/201-continue-mouthopen-accurate-tangents.py \
> tmp/201-continue-mouthopen-coupled-018.log 2>&1
```

Launched 04:04:55 UTC with PID 1229595, start ticks 5415491,
boot `fe94509a-c363-40c5-9ffa-8a0cc2ad928e`, tool session 6455.
`data/inverse-mouthopen-coupled-018/job.json` and the authoritative state pointer
record the process. Startup and actual accepted updates must be checked;
neither CPU tests nor the frozen derivative diagnostic certify inverse
convergence. No commits or pushes were made.

## Verified startup and resumed progress

The initial RMS, full loss, loss components, pose and activation statistics
match audited017 exactly; force differs only at roundoff. The independent
CPU first-cache audit reproduces all four step-374 Adam moment tensors exactly.
Protocol records objective continuity and the absolute cutoff. All 440 source
snapshots and nine current loaded files were hash-verified. Four live tailnet
assets match byte for byte. See run018 `independent-source-audit.json`,
`accurate-tangent-resume-verification.json` and `live-tailnet-verification.json`.

The formerly failing combined determinant comparison passes at the original
5e-5 epsilon and original comparison thresholds. Four accepted updates now
reach RMS 1.8455508693680702 mm and full loss 0.14401470819735934, with force 0.0007667989823972464 N, 100 inversions and
unchanged inverted-volume fraction. Their alpha values double from 0.00390625
to 0.03125. These first seeds already met the internal force target and required
zero corrector iterations; this is accepted progress, not evidence of inverse
convergence or a new independent physical endpoint audit. Future receipts must
verify sustained correction as steps grow.

Comet: <https://www.comet.com/liblaf/apple/8a1947ddf7bf4440a62248030286c4c3>
