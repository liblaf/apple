# Sequential expression fitting

The user requested one expression at a time. Run 004 continues the accepted
states from run 003 and starts with MouthOpen. It stays on the current expression
until inverse convergence or an explicit line-search failure. Exhausting the
per-expression accepted-step budget or overall wall budget stops the run; it
does not declare convergence or move to the next expression.

The original target order, fixed materials, six active-stress coordinates per
muscle tetrahedron, strong smoothness, one mandible hinge angle, PNCG equilibrium,
implicit adjoints and contact checks are unchanged. While one expression remains
active its adjoint is retained for the next solve. The physical and optimizer-step
methods are checked against the archived parent by parsed syntax trees.

## Continuation evidence

Run 003 was stopped after saving one accepted update each for MouthOpen and
BrowDownLeft. The new run imports their checkpoint bytes, Adam moments, gradients,
accepted/trial histories and exact smoothness calibration. It records parent
protocol and source hashes and hashes every imported file in `continuation.json`.
The parent remains unchanged. NeckCompression had only an empty directory after
interruption and remains queued with zero accepted updates.

The initial launch failed before any physics or checkpoint import because the
in-memory target order was a tuple and the saved order was a JSON list. The order
is now explicitly materialized as a list, including for later resume comparisons.
The failed startup is retained in `data/expression-fitting-004-startup-rejected`.

## Checks and launch

`data/expression-continuation-check-003/summary.json` records passing CPU checks:
two consecutive updates of A before B; advancement after a recorded line-search
failure; budget stopping before B; imported accepted steps counting toward the
budget; byte-preserved activation, Adam state and histories; rejection of a
running parent, an inconsistent empty directory, and a changed physical protocol.

```bash
DEBUG=1 CHERRIES_NAME='Sequential continuation final CPU check' \
CHERRIES_TAGS='expression-fit,sequential,validation,cpu' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/102-check-expression-continuation.py \
  --output-dir data/expression-continuation-check-003

CHERRIES_NAME='Sequential expression fitting with one mandible hinge angle' \
CHERRIES_TAGS='expression-fit,all36,sequential,fixed-material,rigid-eyes,pncg,mandible-hinge' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/93-fit-expressions.py \
  --continue-from data/expression-fitting-003
```

The runtime-only `apple-expression-fit.service` runs the latter command. Logs
are in `logs/93-expression-fitting-004-terminal.log`. The runtime-only review
publisher now follows run 004; the page shows accepted counts, RMS reduction and
hinge angle, with the currently active expression named explicitly.

Live review (private preview omitted).
This is continued fixed-material fitting, not a converged expression result or
the final joint material optimization.

Live startup verification confirmed `schedule=sequential`,
`current_expression=MouthOpen`, and one retained update each for MouthOpen and
BrowDownLeft. All 21 imported files and both optimizer checkpoints matched their
recorded hashes. The active [Comet run](https://www.comet.com/liblaf/apple/99e0c809ceb74b6a9a4d38f9739a3690)
records the continued optimization.
