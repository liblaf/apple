# Expression equilibrium runtime validation

This validates the fixed-material, eye-inclusive differentiable equilibrium used
by the all-expression activation and mandible-pose fit. The constitutive
reference, prescribed heterogeneous skin baseline, source skull, source eyes,
and numerical 10 nm CCD buffer are unchanged.

The zero-activation, zero-pose equilibrium returned through the initial-force
gate with `1.49980232014314e-10` code units, below the inherited
`1.5192003475221146e-10` threshold. A spatially constant, symmetric
`1e-8 MPa` active stress over every muscle tetrahedron and a nonzero six-DoF
pose converged in 159 PNCG steps to `1.4925448286559435e-10` code units.

The real-mesh implicit adjoint converged with relative residual
`9.791910130674421e-08` under the `1e-7` target. Both active-stress and jaw-pose
gradient norms were finite and nonzero. Terminal IPC had no intersections and
an active gap above 10 nm.

Runtime policy: strict Armijo `0.25`, initial damping `0.001`, restart every
200 steps, 0.5 mm maximum component, TightInclusionCCD tolerance 0.1 nm with
100,000 iterations, 0.95 safety on restrictive CCD steps, and rejection of
sub-10 nm Armijo trial states. These affect only trial feasibility; the IPC
energy and derivatives remain unchanged.

Evidence: [summary.json](../data/expression-equilibrium-validation-003/summary.json).
