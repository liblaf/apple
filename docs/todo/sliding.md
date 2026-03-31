# Sliding Constraint

## Conclusion

Sliding should be modeled as a scene-level kinematic constraint, not as a per-energy feature.
Energy implementations should continue to work in full displacement coordinates and should not
need any sliding-specific logic.

For one solve with frozen sliding directions, the constrained displacement can be written as:

```math
u = c + P q
```

where:

- `u` is the full displacement field
- `q` is the reduced solver vector that contains only admissible DOFs
- `c` is the affine offset introduced by constraints
- `P` maps reduced DOFs back to the full displacement field

Under this formulation:

- full-space energies still compute `value(u)`, `grad_u`, and `hess_prod_u`
- the scene or constraint layer applies the chain rule
- reduced solvers work only on `q`, not on the full displacement vector

This means sliding belongs in the shared constraint machinery owned by `Scene`, not in
`jax.Energy` or `warp.Energy`.

## Why It Is Deferred From V1

Although the reduced-DOF formulation is clean, it adds a nontrivial amount of orchestration work:

- a generalized constraint representation is needed, beyond fixed/free Dirichlet DOFs
- `Forward` must solve in reduced coordinates and map between reduced and full displacements
- `Inverse` and the adjoint solve must also operate in reduced coordinates
- preconditioning and Hessian-vector products need reduced-space handling

This is valuable, but it is not necessary for the V1 rewrite.

## V1 Scope

V1 should ignore sliding constraints entirely.

The V1 implementation should support only standard Dirichlet constraints:

- fixed DOFs are removed from the solver state
- free DOFs are packed into the solver vector
- energies remain unaware of constraints except through the current full displacement

This keeps the first rewrite focused on:

- multi-object scene assembly
- shared high-level orchestration
- backend-specific energy implementations
- stateful `Forward` / `Inverse` driven by `liblaf.peach`

## Revisit Later

When sliding is reintroduced after V1, the preferred direction is:

1. add a generalized scene-level kinematic constraint system
2. solve forward and adjoint systems in reduced coordinates
3. keep energy implementations unchanged

In other words, sliding is a future constraint-layer feature, not a V1 energy-layer feature.
