# Pose-first expression fitting

Run 006 starts each expression from the approved eye-inclusive neutral with
zero muscle activation. It first optimizes only the existing 1-DoF mandible
hinge. Once that scalar problem passes its recorded stationarity and primal
accuracy gate, it unlocks the six symmetric active-stress coordinates per muscle
tetrahedron and continues optimizing activation and jaw together. Expressions
remain sequential, starting with MouthOpen. Run 005 is preserved for comparison.

## Model and optimizer

Materials, skin loading, neutral/target transfer, 0–40 degree hinge bounds,
full source cranium/mandible and fixed eyes, PNCG equilibrium, implicit adjoint,
strong smoothness and the normalized objective are unchanged. Pose-only means
zero *muscle activation*; prescribed skin stress remains active.

The scalar stage uses a separate maximum proposal of 1 degree. Positive secant
curvature supplies an inverse-Hessian estimate when available; otherwise the
bounded scalar gradient supplies the direction. Proposals undergo the same
contact, raw and residual-corrected Armijo, and estimated primal-error checks as
the joint stage. After backtracking, the next proposal cap is at most twice the
last accepted angular displacement. This avoids tying jaw progress to the
activation Adam learning rate of 0.003 (0.03 degrees for the jaw).

Pose convergence requires
`abs(j - clamp(j - gradient_j, 0, 4)) <= 1e-3`, where `j` is in 10-degree units,
and absolute first-order primal objective correction no larger than
`1e-6 * max(1, abs(objective))`. This is a constrained first-order stationarity
criterion for the scalar problem, not a global-minimum or joint-fit certificate.
The mapping uses a unit gradient step; its tolerance is not a bound on angle
error to an unknown optimum. The equilibrium/contact checks must already have
passed at the saved state. No stable-objective history is required for this
separate scalar gate: a secant step can reach stationarity before five additional
numerically resolvable decreases exist.

The transition writes `pose-only-final.pt`, `pose-stage.json`, and
`joint-initial.pt`. It preserves the accepted jaw, deformation, zero activation
and full joint gradient already evaluated at the same state and objective.
Adam moments and joint convergence history start fresh. The transition does not
increment accepted steps or declare inverse convergence. Stage labels and pose/
joint step counts are recorded separately. Rejected pose proposals preserve the
accepted checkpoint and do not unlock activation. Budget exhaustion stops the
run without a stage transition or advancement to another expression.

## Commands and validation

From this experiment directory:

```bash
DEBUG=1 CHERRIES_NAME='Pose staging projected descent regression' \
CHERRIES_TAGS='expression-fit,pose-first,validation' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/103-check-projected-descent.py \
  --output-dir data/projected-descent-check-003

CHERRIES_NAME='Pose first then joint sequential expressions' \
CHERRIES_TAGS='expression-fit,sequential,pose-first,fixed-material,rigid-eyes,pncg,mandible-hinge' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/93-fit-expressions.py \
  --output-dir data/expression-fitting-006 --pose-first true \
  --calibration-source data/expression-fitting-001
```

The existing projected-descent regression passed under the staged runner.
The scalar stage and transition CPU checks in
`data/pose-first-check-001/summary.json` also passed. Actual production methods
on a small analytic objective produced angles 0, 1, and 1.2 degrees, with
objectives 1.0288, 1.0008, and 1.0; positive secant inverse curvature was 0.25.
They verify exact zero activation, both stationary bounds, unresolved primal
rejection, checkpoint and warm-adjoint rollback, full-gradient preservation,
fresh joint optimizer/history, activation unlocking and sequential total budgets.
Both check receipts bind runner SHA256
`59a8b1e8fff0e119e704028eaf3289e860a8dbb249e87728dea945dcc0bda302`.

```bash
DEBUG=1 CHERRIES_NAME='Pose-first CPU regression' \
CHERRIES_TAGS='expression,pose-first,cpu-check' \
uv run --frozen python src/104-check-pose-first.py
```

These check optimizer/state-machine behavior using a small analytic objective;
they do not constitute a new full anatomical finite-difference derivative test.
The prior exact-Hessian and residual-corrected implicit-derivative validation
scope is retained, including its limitations at contact transitions.

## Live evidence

Run 005 was intentionally stopped with MouthOpen at 13 accepted updates,
RMS 7.388115 mm and opening angle 0.359365 degrees. Its accepted checkpoints
are preserved and hash-recorded in `expression-fitting-005/external-interruption.json`.

Run 006 launched in the transient `apple-expression-fit.service` with the
command above. Startup reused the same 36-adjoint smoothness calibration,
weight 5.066584049455902, and selected MouthOpen first. It starts from the
approved neutral with zero activation rather than importing run 005's activation.

[Comet run](https://www.comet.com/liblaf/apple/6ab8105b66374eb180478ed0fb93572e).
The startup verification confirms MouthOpen is in the pose-only stage, the saved muscle activation is identically zero, the jaw has one coordinate, the initial forward/contact checks passed, and all runtime source hashes match the protocol. See `data/expression-fitting-006/startup-verification.json`. The long-running job remains unconverged; startup is not evidence of fitting success.
The runtime-only review publisher follows run 006 at
the live review (private preview omitted).
