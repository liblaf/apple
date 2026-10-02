# Preflight validation

The CPU normal and Raw6 regularizer gate passed. Normal loss is zero at target,
unchanged by a common translation, oriented by the supplied triangle winding,
and weighted by reference rather than deformed areas. A constructed 16:1 area
ratio is reproduced, and collapsed triangles fail visibly. Centered finite
differences agree with normal autograd to relative error 7.191e-11 and with the
Raw6 Frobenius regularizer derivative to 9.64e-11. Off-diagonal terms count twice.

The full 3D implicit derivative gate passed at a small, recorded nonzero Raw6
control field. It checks the normal-only path separately from the actual
combined positional/normal/smoothness objective, in gradient-sign and
muscle-coherent directions, using epsilon .002 and .001. The maximum relative
error is 1.8877% for normal-only in the muscle-coherent direction at epsilon
.001; the corresponding epsilon .002 error is .4987%. Perturbation-size
disagreement is 1.3890%, below the predeclared 2% limit. All other errors are
at most .01093%, including every combined-objective check. These are finite
accuracy checks, not exact derivatives or inverse convergence certificates.

Validation uses tighter forward tolerances (relative 1e-6, absolute 1e-12,
maximum 10,000 steps) than training (relative 5e-4, absolute 1e-10, maximum 5,000
steps). The adjoint retains the established 5e-4 relative tolerance. A fixed
random seed 20260921 generates validation controls and directions only; all
fitting starts are exactly neutral and deterministic.

Working directory:
`exp/2026/09/21/normal-matching-face`.

```bash
env DEBUG=1 CHERRIES_NAME='Face normal loss: CPU algebra and Raw6 regularizer validation' CHERRIES_TAGS='face,normal-loss,regularizer,validation,cpu' .venv/bin/python src/05-verify-normal.py
env DEBUG=1 CHERRIES_NAME='3D face normal matching: implicit derivative validation' CHERRIES_TAGS='face,raw6,normal-matching,validation' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py --output 06-validation --validation-only true
```

Both completed with exit code 0 after Cherries shutdown. DEBUG validation records
local evidence without a remote Comet run. Receipts are in
`data/05-normal-verification/checks.json` and
`data/06-validation/gradient-validation.json`. The latter directory also saves
the mesh, normalizations, fixture hashes, configuration and source snapshots.
The main runner requires matching numerical hashes, normalizations, fixture and
loss/optimizer settings before it can start.

The first CPU normal-only receipt is preserved in
`data/05-normal-verification-initial`. A preflight source-archiving error is
preserved in `data/06-validation-archive-error`; PyTorch exposes a virtual
relative `_ops.py` filename, which the original broad source collector mistook
for a local experiment source. The collector now accepts absolute source paths
only. That attempt failed before derivative evaluations or optimization.
An earlier CLI invocation omitted the required `true` argument for the boolean
validation flag and exited during argument parsing. Neither attempt supplies
run results. An inherited PyTorch non-leaf `.grad` warning appears in the log;
the required control gradients and both solver receipts passed explicit checks.

The initial full-solver gate and two-update smoke audit passed before an
additional start-isolation review identified the inherited cross-branch adjoint
solution cache. The runner now clears that cache before each branch. The complete
implicit gate was rerun against the final source hashes and passed with the
numbers reported above. Earlier artifacts remain under
`data/06-validation-before-adjoint-reset`,
`data/09-smoke-before-adjoint-reset` and
`data/09-verification-before-adjoint-reset`. The final smoke uses one update per
branch to verify the isolated initial update before all main runs restart.

The final one-update smoke is saved in `data/09-smoke`, with its independent
CPU receipt in `data/09-verification/checks.json`. All four starts have exactly
zero controls, displacements and Adam moments, and an explicit zero initial
adjoint guess. For a given data objective, smoothing off/on initial physical
activation updates agree within 6.60e-7 relative RMS difference. Adding normal
matching changes the initial physical update RMS by +2.964%; the 5% neutral
objective weighting is not an assertion about gradient magnitude.

A later audit-only strengthening of `src/20-verify.py` enforces the exact
configured update count, one forward and adjoint receipt per evaluated state,
completed status and null failure for every branch. The fresh CPU smoke audit
`data/09-verification-completion-contract/checks.json` passed: its configured
one update requires two evaluated states and two receipts per branch. The main
100-update run will require 101 of each. This change does not modify the frozen
numerical runner or fitting source snapshots.
