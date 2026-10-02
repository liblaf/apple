# Collision-on pair finalization preparation

The user requested computation before 14:00 Asia/Shanghai on September 30,
followed by visualization of MouthOpen and Smile. MouthOpen runs locally until
05:55 UTC; Smile runs on paratera-4090 until 05:50 UTC. The deadline does not
establish inverse convergence. This report records preparation while computation
continues; no final experimental images have been rendered in this phase.

## Complete objective curves

Renderers 147 and 181 now use `src/coupled_review_curves.py` to add
`objective-components-curves.png`. Separate panels show recorded total loss,
normalized position L2, weighted normal loss, weighted activation smoothness,
and raw activation roughness. Historical L2-only rows provide their position
loss, while unavailable normal and smoothness terms remain missing rather than
being fabricated as measured zeros. Positional RMS remains derived from the
position term alone.

The lineage join preserves an initial row at the same iteration when it records
an equilibrium refinement or an actual objective transition. Exact objective
continuity is checked against the parent protocol and adds no transition marker.
The shared plotting helper and both renderer call sites passed five focused CPU
tests, including a synthetic temporary PNG render. Ruff and compilation passed.
The synthetic plotting test is not an experimental endpoint visualization.

An independent CPU replay of the full persisted MouthOpen 001-to-018 lineage
passed. The fixed-control refinement occurs at cumulative iteration 226, and the
L2-to-mixed objective transition occurs at 325. Run018 continuation begins at
385 with the same objective and adds no transition. These cumulative curve
indices include early historical updates and are distinct from current Adam
counters. The receipt is
`data/inverse-mouthopen-coupled-018/full-lineage-stitch-audit.json`.

## Smile storage recovery

The original remote 30 GB filesystem filled during run003. Checkpoint, endpoint
and summary agree at update 32; raw progress includes a numerically accepted
update 33 whose endpoint/checkpoint save failed. That row is preserved as failure
evidence and must not enter the adopted continuation trajectory.

The remote owner copied and hash-verified the required files to a separate
50 GB volume. The first relocated audit encountered a source-path binding
assertion before physical evaluation. The corrected audit passed against the
original source binding and saved its output on the new volume. Run005 then
encountered nested configuration argument parsing before CUDA and made no
updates. Additive run006 corrected that wrapper, passed the CPU preflight and
derivative check, and resumed exact durable step32 with both original Adam
counters32. Its first saved update reduced positional RMS to 4.949189878433682
mm, with force 0.0009457356328868062 N and 98 retained inversions. The objective
and physical policy were unchanged. The independent parent audit hash is
`069548b0e48555ac9d1d0a021f3641e56bf4f0cbfb4b74fe7a82a23965763182`.

The remote owner's authoritative shared receipt is
`exp/2026/09/30/collision-off-expressions/data/session-pair-coordination.json`.
The historical directory name does not describe the current collision-on scope.
Rendering recovered Smile requires the full verified mirror, a hash-bound
recovery receipt for the durable parent trajectory, and exact objective
continuity evidence. The frozen remote runner's historical objective-change
label must not create a false transition when the complete objective is equal.

The completed local recovery integration uses
`src/smile_recovery_lineage.py` and renderer181's explicit
`--recovery-lineage-receipt`, `--recovery-preflight` and
`--recovery-parent-audit` inputs. It re-hashes every 3090 metadata input,
independently checks exact parent/child objective definitions and terms, and
requires raw parent rows 0 through 33 to agree with the recovery evidence.
Only rows 0 through 32 enter the adopted trajectory. Run006's initial state is
retained as a separate zero-update boundary, with a square marker and a break
in plotted lines, because no independent initial displacement archive survived.

CPU integration passed against the frozen initial 003-to-006 receipt and fails
against a changed child receipt path. The combined finalization suites passed
17 tests. This validates the plotting and provenance code, not inverse
convergence. After final006 is mirrored, regenerate 3090 metadata against that
completed bundle; the startup receipt is intentionally insufficient for final
rendering. Exact final commands are recorded in
`tmp/coupled-finalization-readiness-002.json`.

At 05:45 UTC, the completed run006 bundle and regenerated final 3090 metadata
also passed CPU integration. The original terminal reason is
`joint_projection_failed`; its numerical worker and Cherries exited with code 1
at 05:35:28.757735 UTC, then audit180 exited successfully at 05:36:21.751650 UTC.
No deadline signal was sent for this failure. The exact last numerical operation
time is not inferred from process completion. The audited endpoint is update12,
Adam counters44: positional RMS 4.944966442803761 mm, force
0.0009780286001987168 N, 99 retained inversions, inverted rest-volume fraction
2.9434288519495092e-5. These facts do not establish inverse convergence.

The final binding resolver explicitly understands the collector's data-relative
directory layout and uses `--local-data-root data` for omitted immutable local
inputs, accepting only exact recorded hashes. The interruption verifier binds
the immutable collector manifest, exact worker identity, process/GPU completion
evidence, final audit and saved file hashes; it deserializes the CPU checkpoint
and checks exact controls, displacement and counters against endpoint/progress.
All 23 combined tests passed in the root verification. Updated final commands
and source hashes are in `tmp/coupled-finalization-readiness-003.json`.

## Deadline interruption receipts

Runner130 normally writes a terminal deadline status. An external SIGINT can
leave the immutable summary marked `running`. After verifying the numerical
process and Cherries have both exited, `src/deadline_interruption.py` can record
that interruption without rewriting the summary. The receipt binds the original
job identity, all six saved input hashes, exact endpoint/checkpoint controls,
displacement and counters, and the observed stop time. It also records whether
the user deadline was met. A late observation remains auditable and never
permits optimization to resume.

Renderer147 accepts that verified receipt and labels the result
`interrupted_for_computation_cutoff`. Five focused CPU tests cover valid and late
receipts, source mutation, mismatched state, and unrelated PID reuse. No real
interruption receipt was created while run018 was active. The finalizer command,
only after actual shutdown, is:

```bash
.venv/bin/python src/deadline_interruption.py \
  --run-dir data/inverse-mouthopen-coupled-018 \
  --observed-stop-utc '<actual UTC stop observation>' \
  --cherries-shutdown-verified
```

## Delivery gates

After the exact numerical workers and Cherries have exited, independently audit
the last consistent accepted endpoints. Render full original tet boundaries,
bones and eyes, together with complete branch position/force/objective curves.
Verify published tailnet asset bytes and inspect the final images. Report actual
stop times, positional RMS, each weighted objective term, raw roughness, jaw
pose, force, retained inversion count and volume fraction. Unless separately
certified, label the pair deadline-limited and not inverse-converged.

## Final delivery

Both computations are finished and both audited previews are published. Final commands and source hashes are recorded in `tmp/coupled-finalization-completion-004.json`. Readiness003 used same-byte provenance copies at different paths from the final metadata; renderer181 correctly rejected that invocation. The completed command binds the exact full006 preflight and parent-audit paths. See `docs/203-collision-on-expression-pair-final.md` for final results and publication evidence.
