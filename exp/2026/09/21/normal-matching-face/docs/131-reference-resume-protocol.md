# Recovery of the selected reference-length face fit

The original fit starts each independent branch from neutral, using the protocol in
[115-reference-fit-protocol.md](115-reference-fit-protocol.md). The original
smoothness-off process stopped unexpectedly after accepted update 102. Its log has
no recorded solver failure or graceful shutdown; the cause is unknown. Its final
summary still says `running`, with `failure: null`. Preserve all numerical artifacts
in `data/130-reference-fit` and its separate `interruption.json`; do not reinterpret
the interrupted run as completed.

Recover smoothness-off from q102, u102, and the saved Adam moments/counter, then run
smoothness-on from neutral. Both receive the original 200-update budget, objective,
fixtures, physics, and tolerances. The new output is `data/132-reference-continuation`.
The dimensionless objective remains `P / l_ref_mm**2 + N + eta * R`, with
`l_ref_mm = 13.236093032531715`, `eta = 0` or `1.8346203690062914e-05`, Adam learning
rate 0.3, betas (0.9, 0.999), and epsilon `5.707952862381702e-05`.

## Preflight and replay

The independent CPU audit `data/131-reference-resume-checks/checks.json` must pass.
It checks the 99 original source records, three fixture hashes, neutral origin,
103 accepted trace/solver records, exact last-state/checkpoint agreement, saved Adam
moments/counter, and synthetic next-update algebra. Recheck every receipt before
initializing recovery. The recovery source archive must contain every original
source path/hash unchanged. Freeze this supplement before running.

Copy the original off-branch history through 102 without replacing the original.
Save restored state as `continuation-start.pt`. Re-evaluate its forward/adjoint state
using saved u102 as the forward seed and a zero adjoint seed. The old u101 and
adjoint warm starts were not checkpointed: this is an explicitly resumed trajectory,
not a claim of bitwise continuity. Require the following absolute replay differences:

- Objective and normalized position contribution: at most 1e-4 times their saved values.
- Position RMS: 0.001 mm; normal angle RMS: 0.01 degree.
- Normal loss: 1e-4 times its saved value.
- Activation smoothness: 1e-12 times max(1, saved value).
- Minimum physical det(F): 0.001; inversion count: exact agreement.
- Maximum displacement difference: 1e-6 m.
- Physical-gradient RMS relative difference: 0.02.

Verify q, m, v, and counter are unchanged by replay. Save replay metrics, solver
receipts, errors/limits, and the recomputed gradient. Gradient direction at 102
cannot be compared because the original gradient vector was not checkpointed.
Use the restored Adam state and replayed gradient for update 103; save step103 for
an independent algebra check. Continue through 200. On any failed gate, preserve
failure evidence and do not advance that branch. The independent smooth-on branch
still starts from neutral. A branch failure makes the overall run unsuccessful.

## Execution and lifecycle

Working directory: `exp/2026/09/21/normal-matching-face`.
Launch in a transient user service with no persistent unit file. This preserves the
process across tool/chat transitions. Keep the service after completion until its
exit status and Cherries shutdown log are inspected, then stop the transient unit.

```bash
systemd-run --user --unit=apple-reference-fit-132 --collect \
  --property=RemainAfterExit=yes \
  --property=WorkingDirectory=exp/2026/09/21/normal-matching-face \
  --property=StandardOutput=append:exp/2026/09/21/normal-matching-face/logs/132-service.log \
  --property=StandardError=inherit \
  /usr/bin/env CHERRIES_NAME='Face reference loss: recovered off and fresh on fits' \
  CHERRIES_TAGS='face,raw6,normal-matching,reference-length,neutral,smoothness,recovery' \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python src/132-resume-reference.py
```

After both branches finish and the service exits successfully, extend the independent
CPU verifier to verify the parent source subset, exact trace/receipt prefix,
restoration/replay, update103 Adam algebra, final metrics, and every solver receipt.
Use the recovered output for analysis and rendering. Report this interruption and
finite-tolerance replay alongside any inversion or convergence limitations. Keep
all comparisons at the same update count, with an additional common inversion-free
checkpoint comparison. Verification success does not establish physical validity
or inverse-optimization convergence.
