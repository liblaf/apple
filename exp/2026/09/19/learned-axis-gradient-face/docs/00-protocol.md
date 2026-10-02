# Learned-axis contraction-only face fit, with and without smoothness

Continue the gradient-only surface loss comparison from an exactly neutral face.
The activation is `B = I + s n n^T`, with nonnegative strength `s` and unit axis
`n` independently optimized in each active tetrahedron. This is three physical
degrees of freedom per cell (one strength, two axis coordinates). The active
natural stretch along the axis is `1/(1+s)`; transverse active stretches are one.
Adam updates both stored variables, followed by clamping `s >= 0` and normalizing
`n`. No strength upper bound or magnitude penalty is introduced.

At neutral, `s=0`, `B=I`, and displacement is zero. Both branches use the same
axes: minimum-eigenvalue eigenvectors of the chosen data objective's symmetric
tensor gradient at neutral. Axis gradients vanish at zero strength, while strength
gradients can activate cells. These are effective target-derived axes, not measured
anatomical fibers. No prior fitted face supplies initialization.

The data term and corrected physical-volume energy are inherited from
`../gradient-only-face`: rest-surface derivative matching, scale
`99.56749008299767`, and physical `J=det(F)`. Positional L2 is evaluation-only.
The mesh, materials, constraints, surface support, and full forward/adjoint solver
are unchanged. The new regularizer is

```text
R = ell^2 / V_active * sum_(i,j) w_ij ||B_i - B_j||_F^2
ell = 0.005 m
```

Edges join shared-face tetrahedra within the same muscle label; conductances and
physical muscle-volume weights are fixed. Tensor differences make the penalty
invariant to the equivalent axes `n` and `-n`. The off objective is `L_data`; the
on objective is `L_data + lambda R`.

Before comparison, check algebra/projection/sign invariance on CPU, and finite
differences through the full implicit face solve for strength and tangent-axis
perturbations. Calibrate a base coefficient by equalizing strength-plus-axis
Euclidean gradient norms after an eight-update data-only probe. This is an
optimization-scale heuristic, not physiological calibration.

Restart at neutral for 20-update pilots at coefficients zero and 0.1, 1, and 10
times the base coefficient. Select the smallest tested positive coefficient that
reduces squared tensor roughness by at least 75% relative to the unregularized
pilot and has no inverted tetrahedra. Report its data-fit cost. If none qualifies,
extend the coefficient bracket explicitly rather than silently substituting one.

Restart both final branches at neutral with fresh Adam states: initial learning
rate 0.3, epsilon 0.01, identical step-based halving every 100 updates. The first
comparison uses 200 updates per branch (twice the preceding Raw6 comparison),
with saved checkpoints and solver receipts. This explicitly overrides the
runner's default 500-update budget. Branches are evaluated
in alternation at matched update counts. A branch may stop after two consecutive
25-update checks (starting at 100) with objective relative span below 0.001 and
physical rank-one projected-gradient RMS below 1% of its initial value. This
scale-dependent numerical stationarity diagnostic is not a proof of global
optimality or mechanical stability. Budget exhaustion is reported separately.

Evaluate surface position error, surface derivative error, retained motion,
regional 5 mm high-pass residuals, activation roughness, axis learning, strengths,
physical determinant minima/inversions, and optimization diagnostics. Contraction-
only activation guarantees the active tensor constraint; it does not guarantee
positive physical element volume or a stable mechanical equilibrium.
