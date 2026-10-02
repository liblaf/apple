# Neutral-start 3D face: L2 versus surface-gradient matching

The user selected a comparison from the neutral face. Both branches start with
zero displacement, inactive muscles (`q=0`, `B=I`), and fresh Adam moments.
Neither branch uses a previous fit or previously fitted activation directions.

## Matched setup

- Use the frozen historical Smile fixture and the experiment-local corrected
  physical-volume energy from the September 14 full-face study: the volume term
  uses `J=det(F)`, while activation affects the norm of `F B`.
- Keep the full-resolution volume, passive material constants, skull fixation,
  no-skin-energy configuration, forward and adjoint tolerances, and the
  unrestricted six-component symmetric activation model identical.
- Optimize all 288,235 active tetrahedra with `B=I+sym(q)`. There is no activation
  smoothness penalty, spectral projection, or contraction-only restriction.
- Use the same 15,299 skin vertices and 29,899 triangles for both objectives.
  Three isolated points from the historical 15,302-point positional support
  are excluded from both branches because they are absent from this skin.
- Run 100 Adam updates per branch with learning rate 0.3, epsilon 0.01,
  betas (0.9, 0.999), and no learning-rate decay. This is a finite-budget pilot,
  not an inverse-convergence claim.

## Objectives

Let `e_i = u_i - u_target_i`, with displacements in meters. All areas, weights,
and triangle basis gradients are fixed on the reference skin.

The positional branch minimizes

`L2 = (10^6 / 3) sum_i w_i ||e_i||^2`,

where the normalized lumped vertex weights sum to one. The factor converts to
component mean squared error in square millimeters. Reported positional RMS
is the full vector RMS, `sqrt(3 L2)` millimeters.

The gradient-only branch minimizes

`Lgrad = c / A sum_t A_t ||sum_i e_i tensor grad(phi_i)||_F^2`.

The fixed positive scalar `c` matches the initial active-muscle-volume-weighted
Frobenius RMS of Adam's proposed change in `B` to the L2 branch. This accounts
for the effect of epsilon on differently scaled losses; it does not match
update directions or subsequent optimizer steps. Off-diagonal entries count
twice in the Frobenius norm. No positional term is added.

This measures the vector-valued tangential derivative of the displacement
residual on the corresponding reference triangles. It is translation-invariant
and is neither a normal-only loss nor a rotation-invariant shape distance.
The existing skull constraints are retained; no additional positional anchor
is introduced. Mean residual and centered positional RMS are reported to
expose the constant-translation nullspace.

## Gates and evidence

Before fitting, verify the surface operator's zero, translation, affine,
refinement, and autograd behavior. Then compare the complete implicit gradient
against central finite differences through the full equilibrium solver at
two perturbation sizes and in two directions. Require less than 2% derivative
error and perturbation-size disagreement.

Record successful forward and adjoint receipts for each evaluated state,
positional and gradient errors, motion magnitude, frozen 5 mm surface
high-pass diagnostics, physical determinants, and activation eigenvalues.
Save the last state, best objective, best noninverted state, and checkpoints.
Fail on solver or nonfinite errors and retain failure evidence.

Compare endpoints at the same update budget. An inverted endpoint must be
identified explicitly; an earlier noninverted state cannot silently replace
it in the primary comparison. Positive determinants and small force residuals
alone do not certify a mechanically stable equilibrium.

The target is known on the surface only. Render and export that target as a
surface; do not invent a target deformation for the interior volume.
