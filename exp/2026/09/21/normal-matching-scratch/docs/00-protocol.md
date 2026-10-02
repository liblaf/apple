# From-scratch 4 by 2 by 2 factorial comparison

User-specified factors: four activation parameterizations, activation smoothness
on/off, and L2-only versus L2 plus target-normal matching. The user selected only
target height 0.20: exactly 16 runs. This replaces the earlier warm-continuation
design. Every run independently initializes control q=0, B=I, displacement u=0,
and Adam moments m=v=0. No saved fitted control, displacement or optimizer state
is loaded. Learned-axis initialization has s=0 and theta=0, with both trainable;
the angle derivative vanishes initially at zero strength.

Retain the historical 100 by 10 rectangular triangular mesh (width 1, thickness .1,
400 active muscle triangles in y=.04..06), fixed sides/bottom, muscle/fat Young's
moduli .03/.003, Poisson ratio .49, and corrected physical-J energy. Rows are
unrestricted symmetric (3 DoF), PSD contraction-only (3), learned-axis rank-one
contraction (2), and fixed-x contraction (1), independently per muscle element.

The objective is

    J = L2 + beta * (L2_neutral / normal_neutral) * N + weight * h^2 * R.

L2 is mean squared vector displacement error at free top vertices. N is the
validated fixed-reference-length-weighted oriented target-normal chord loss on
all deformed top edges (equivalent to unit-tangent matching). R is mean squared
Frobenius difference of neighboring activation tensors B in the muscle.
The two losses use beta=0 or .05; the latter is the weaker previously tested
normal weight. The two smoothness settings use weight=0 or 1; weight 1 is the
previous established smoothness-on setting. Both coefficients stay fixed, with
no tuning or winner selection during this factorial experiment.

Use the original neutral-start optimizer contract: 1200 Adam updates, learning
rate .03 times .99^step, betas .9/.999, epsilon 1e-8, raw objective gradient,
post-update activation feasibility projection, forward tolerance 1e-10, maximum
250 Newton iterations. Retain all failures and the last accepted states; no
fallback solver, restart, checkpoint resume, or shortened successful replacement.
The inverse budget and decaying learning rate do not establish convergence.

Each run stores its exact step 0 q/u/B/m/v state, per-step scalar trace, histories
every 10 updates plus last accepted, checkpoint and failed-proposal receipt.
For primary comparison within each activation row, use the latest saved step
common to its four factorial variants. Report full endpoints/failures separately.
Plot four rows by two smoothness columns, with both losses in each panel, target
and optional neutral profile. Keep physical scales equal and mark instability.
Report position RMS, target-normal angle, target-relative slope/D2 error,
target projection and peak/motion, activation roughness, physical J/inversions,
force and adjoint residuals, and projected-gradient history separately.

Before running, verify combined normal+smoothness derivatives for all four modes,
exact neutral initialization, and equivalence of beta 0 objectives/gradients to
the historical L2/smoothness implementation. Freeze/hash resolved numerical
sources. Afterward independently verify step 0 state, saved endpoints and common
steps, constraints, determinant and equilibrium residual; inspect forward
Hessian at all 16 common-step states. Positive J alone does not imply stability.
