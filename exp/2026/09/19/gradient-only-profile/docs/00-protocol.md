# Gradient-only fitting of the 2D profile

The user requested gradient matching alone. The inverse objective is therefore

\[
L_g = \frac{1}{S}\sum_i\frac{\|e_{i+1}-e_i\|_2^2}{\Delta x_i},
\qquad e_i=u_i-(0,4h x_i(1-x_i)),\quad S=x_{\rm last}-x_{\rm first}=1.
\]

Differences use the full ordered reference top boundary, including both fixed
corners. The x and y components are both matched. The positional L2 coefficient
and activation-smoothness coefficient are exactly zero. Position error,
activation roughness, and the reference-coordinate second-derivative residual
are evaluation metrics only. Fixed corners remove the constant-displacement
nullspace without introducing another loss term. This is material-correspondence
matching, not a closest-curve or reparameterization-invariant metric.

## Controlled comparison

- Reuse the 100 x 10 layered rectangular mesh, central muscle band y=0.04..0.06,
  fixed side/bottom boundaries, and target heights 0.05 and 0.20.
- Reuse the corrected active Stable Neo-Hookean energy with physical J=det(F),
  muscle E=0.03, fat E=0.003, and nu=0.49.
- Run the same unrestricted symmetric, PSD contraction-only, learned-direction
  rank-one contraction, and fixed-x contraction models. All start at B=I.
- Reuse 1,200 raw-objective Adam updates, initial learning rate 0.03, decay 0.99,
  beta1=0.9, beta2=0.999, epsilon=1e-8, and the same feasibility projections.
- Reuse the saved unregularized L2 baselines under
  `exp/2026/09/15/activation-direction-smoothness/data/tune-w0`, after checking
  numerical source hashes. The large-target unrestricted L2 run stopped at
  update 262; show its actual endpoint and do not present it as a full-budget run.
- Preserve failed proposals and last valid states. Do not silently restart a
  failed case or change the material, forward solver, learning rate, or objective.

## Validation and interpretation

Check the displacement derivative and the entire implicit control derivative by
centered finite differences. Replay saved endpoints independently and verify
constraints, force residuals, physical determinants, and Hessian eigenvalues.
Compare position RMS, reference-slope RMS, second-derivative residual RMS, full
equal-scale profiles, and deformed internal wireframes. Do not compare different
loss scalar values as though they measured the same quantity.

This is a finite-budget test on a synthetic 2D demo. A completed budget does not
prove inverse convergence. Positive physical determinants and a small equilibrium
residual do not establish a stable minimum; report Hessian evidence separately.
The target need not be reachable under the prescribed physics and controls.
