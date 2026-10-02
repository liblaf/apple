# Target-normal refinement in the historical 4 by 2 demo

The four rows are unrestricted symmetric activation (3 DoF), PSD contraction-only
(3 DoF), learned-axis rank-one contraction (2 DoF), and fixed-x contraction (1 DoF).
The columns are parabola target heights 0.05 and 0.20. Retain the 100 by 10 mesh,
width 1, thickness 0.1, muscle band y=0.04..0.06, all 400 active triangles, fixed
sides/bottom, materials, and corrected physical-volume J=det(F) energy.

This is a continuation comparison. All five variants in a cell start from the
same saved L2 update 200 in the September 15 `tune-w0` history. All eight start
states have no top-edge backtracking and minimum physical J >= 0.1089. In
particular, do not start from the large-target free-model endpoint at step 262,
which approached collapse before its next forward solve failed.

Compare continued L2, L2 plus gradient weights beta=0.05 and 0.25, and L2 plus
target-normal weights beta=0.05 and 0.25. No activation smoothness is added.
Fixed neutral-state values normalize the added terms while retaining coefficient
one on the original positional loss:

    J = L2 + beta * (L2_neutral / shape_neutral) * L_shape.

L2 is the original mean squared displacement error at free top vertices. Gradient
matching is the existing full-vector reference-arclength derivative residual on
the entire top chain, including fixed corners. Normal matching compares oriented
unit normals of corresponding deformed and target segments, using fixed reference
edge-length weights. Its loss is sum w_i (1-n_i dot n_i_target), evaluated as half
the squared unit-tangent difference. It does not penalize neighboring normals or
smooth the target. The derivative includes edge-length normalization.

Every continuation resets Adam moments and uses the same 600-update budget,
learning rate 0.003 times 0.995^step, betas 0.9/0.999 and epsilon 1e-8. The common
reset avoids inheriting the historical near-zero late learning rate. Forward
tolerance remains 1e-10, maximum iterations 250, with unchanged adjoints and
post-update activation feasibility projection. This is not a reproduction of
the original neutral-start optimizer trajectory. Normalized values do not imply
identical initial control steps across objectives; record their magnitudes.

Preserve any failed proposal and last accepted state; do not restart or silently
alter a solver. Save scalar traces at every step and displacement/control histories
every 10 steps and at the last accepted state. Primary within-cell comparisons
use the latest update present in all five saved histories. Show short/failed
runs explicitly, with full endpoints in a separate table. The primary comparison
must not silently compare different continuation budgets.

For each added loss, select the tested beta with lowest target-normal error among
candidates whose position RMS is at most 1.05 times matched-step continued L2 and
whose full-target displacement projection is at least 0.95 times that control.
Require all trace states through that comparison to remain inversion-free.
If no candidate qualifies, say so; do not substitute a failed candidate. These
5% limits are proposed fit-preservation tolerances, not validated universal values.
Report slope and second-derivative residuals, normal angular error, position,
motion, peak displacement, target projection, tensor roughness, physical J and
force/adjoint residuals separately. A lower normal loss alone is not a claim of
recovering target amplitude or eliminating every bump.

Before fitting, check direct normal-loss derivatives, target/translation/scaling
identities and full implicit derivatives for all four activation models. Check
reused numerical source hashes and replay all eight starting states. Afterward,
independently recompute endpoint and shared-step metrics, deformation determinants,
activation constraints and force residuals; report smallest Hessian eigenvalues
for selected comparisons if available. Include equal-scale 4 by 2 profiles,
internal mesh views, and objective plus projected-gradient histories. Fixed-budget
completion or a decayed-learning-rate plateau does not establish inverse convergence.
