# Why the complete-skull collision-off cold solve failed

The bounded diagnostic is
[`data/collision-off-curvature-diagnostic-001/summary.json`](../data/collision-off-curvature-diagnostic-001/summary.json)
(SHA-256 `37029811974dabefbb7fd30a17909f2fffcd79e416023a973c679f248274f696`,
[Comet](https://www.comet.com/liblaf/apple/9deaf77973d34a55ae74af982c463ae6)).
It reconstructs the exact collision-off extended model, candidate-002 repaired
displacement, and frozen Spatial80 update-111 materials at the **initial state**
of the failed cold forward. The unsaved seventh Newton iterate is not diagnosed,
and this probe performs no equilibrium update.

The exact free-DOF Hessian is symmetric to `7.26e-16` relative error, but it is
not positive definite. A 30-step Lanczos probe of the symmetrically
diagonal-preconditioned Hessian found a lowest Ritz value of `-0.69645`. The
reconstructed physical free-DOF direction independently gives
`vᵀHv=-0.6964477448` on two HVP evaluations, Euclidean Rayleigh quotient
`-3.8549e-7`, and relative quadratic scale `-0.5282`. This explicit negative
quadratic form proves local negative curvature; the `0.0061` Ritz residual does
not weaken that direct result. The replayable witness is
[`negative-curvature-witness.npz`](../data/collision-off-curvature-diagnostic-001/negative-curvature-witness.npz)
(SHA-256 `bffe44174d89bbe6096b700f58be6cc9b599b9f33b0f40c6cab00b80c94b9383`).

This establishes a concrete failure mechanism: the Newton implementation solves
the exact Hessian system with conjugate gradients, which assumes a positive
definite operator. That assumption is false at this cold state. The witness does
not prove that indefiniteness is the only cause of the 10,000-iteration CG
failure. The diagonal preconditioner is also weak evidence of scale difficulty:
all 597,177 free diagonal entries are finite, positive, and nonzero, but their
absolute dynamic range is `4.73e5`. Taking the reciprocal absolute diagonal does
not remove a negative mode created by coupled off-diagonal terms.

The force controls explain why disabling collision did not make the cold state an
equilibrium:

| Materials and displacement | Free-force norm |
| --- | ---: |
| Passive materials, zero displacement | `2.05e-20` |
| Frozen fitted preload, zero displacement | `6.687e-6` |
| Passive materials, repaired candidate | `1.138e-6` |
| Frozen fitted preload, repaired candidate | `6.784e-6` |

The passive zero-displacement result confirms that the constitutive reference is
numerically stress free. The repaired geometric displacement and the fitted
preload each create imbalance. The collision-off arm removed only the collision
potential; it retained both the repaired deformation and the fitted bulk/skin
preloads. Applying a cold exact Newton-CG solve to that combined nonequilibrium
state therefore encounters an indefinite Hessian before contact can be blamed.

The probe used 33 HVPs and finished in `0.383 s` inside its declared 120-second
cap. GPU contention affects this wall time, not the repeated negative quadratic
form.
