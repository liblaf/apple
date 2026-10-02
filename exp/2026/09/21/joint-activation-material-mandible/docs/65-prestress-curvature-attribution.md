# Does prestress cause the collision-off failure?

The fixed-mode ablation is
[`data/collision-off-prestress-curvature-ablation-001/summary.json`](../data/collision-off-prestress-curvature-ablation-001/summary.json)
(SHA-256 `3dc5c1764dfd5cf3ba5af88b27bdc21622d65648c5852921c522737dd19b46e3`,
[Comet](https://www.comet.com/liblaf/apple/7639866756b342fea6b5256276132ab0)).
It replays the exact negative-curvature direction from diagnostic 64 while holding
skin stiffness at the fitted multiplier `3.0` and toggling all fitted bulk and skin
baseline stresses. It performs no equilibrium solve.

| Geometry | Zero baseline stress | Fitted baseline stress | Baseline contribution |
| --- | ---: | ---: | ---: |
| Reference | `vᵀHv = +0.590116` | `+0.574887` | `-0.015229` |
| Repaired candidate | `-0.681219` | `-0.696448` | `-0.015229` |

Prestress does **not** create this negative mode. At the repaired geometry, the
mode remains negative when every fitted baseline stress is set to zero. Adding the
fitted stresses makes its quadratic form 2.24% more negative. Moving from reference
to repaired geometry changes the same fixed-mode quadratic form by `-1.271335`,
whereas fitted baseline stress contributes `-0.015229`, about 1.20% of that change.
This attributes the sign change of this witness primarily to the repaired
deformation. It does not prove how every Hessian mode behaves.

The force controls separate equilibrium imbalance from curvature:

| State | Free-force norm |
| --- | ---: |
| Passive stiffness 1, reference geometry | `2.05e-20` |
| Zero baseline, fitted stiffness 3, reference geometry | `5.61e-20` |
| Fitted baseline, fitted stiffness 3, reference geometry | `6.687e-6` |
| Zero baseline, fitted stiffness 3, repaired geometry | `1.138e-6` |
| Fitted baseline, fitted stiffness 3, repaired geometry | `6.784e-6` |

The constitutive reference is numerically force free. The fitted baseline stresses
produce a large net force even at reference geometry, while the repair deformation
also produces force with baseline stress removed. Collision-off therefore removed
only the contact potential; it did not remove the repaired deformation or the
fitted preload.

## How the prestress is solved

The fitted baseline stresses are outer inverse variables, not displacements found
by the inner mechanical solver. `SpatialSharedFieldParameters` maps the shared
coefficients to per-cell symmetric bulk stresses, a skin baseline resultant, and a
skin stiffness multiplier. For a fixed trial parameter vector, `JointPhysics`
inserts these values into the constitutive materials, and the inner equilibrium
solver seeks nodal displacements with vanishing free force. The implicit adjoint
then differentiates the neutral-shape objective and regularizers through that
equilibrium so the outer optimizer can update the shared material variables.

The update-111 stresses were fitted with the earlier partial-FEM collider and were
not equilibrated at the repaired complete-source initialization. Applying them in
one cold step combines a nonequilibrium repaired displacement, nonzero fitted
preload, and an indefinite tangent. The full-source workflow therefore needs a
passive equilibrium followed by staged prestress continuation, with equilibrium
and geometry/contact gates checked at every stage.
