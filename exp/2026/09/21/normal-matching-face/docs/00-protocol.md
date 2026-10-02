# Neutral-start unrestricted 3D face: L2 and normal matching

The user requested the successful 2D factorial design on the human face, using
unrestricted activation. Run exactly four main comparisons: activation tensor
smoothness off/on by L2-only/L2 plus oriented normal matching. Every branch starts
independently at q=0, B=I, u=0 with fresh zero Adam moments. No fitted face,
learned axes, optimizer checkpoint or continuation initializes these runs.
Clear the cached adjoint solution between branches so each first adjoint solve
also starts at zero. Within a branch, the established adjoint warm start remains.

Use the frozen historical Smile fixture and corrected physical-volume Raw6
energy of the September 19 gradient-only face study. Each of the 288,235 active
tetrahedra has six independent symmetric components B=I+sym(q). There is no PSD,
contraction-only or eigenvalue projection. Keep the full mesh, passive materials,
skull constraints, zero skin energy, and forward/adjoint solvers unchanged.
The common skin support has 15,299 vertices and 29,899 corresponding triangles.

The fixed loss is

    J = L2 + beta * (L20 / N0) * N + lambda * R.

L2 is reference-area-lumped mean squared vector displacement residual in mm²,
divided by three (component MSE). L20 is its neutral value. N is the mean squared
chord difference of oriented unit normals of corresponding deformed and target
triangles, weighted by fixed reference triangle areas. N0 is its neutral value.
The normal branch uses beta=.05 and L2-only uses beta=0. Normals depend on actual
deformed triangle geometry, not reference displacement derivatives. Collapsed
triangles fail visibly; there is no denominator clamp hiding degeneracy.

R is the existing same-muscle activation-tensor prior:

    R = ell² / V_active * sum_(i,j) w_ij ||B_i-B_j||_F²,

with ell=5 mm and shared-face conductance equal to face area over centroid
distance times harmonic muscle fraction. Off-diagonal tensor entries count
twice. No edge crosses a muscle label. Lambda is 0 or 0.003214147722027223, the
existing conservative Raw6 coefficient recorded in the September 9 activation
smoothness study. It is fixed before runs and is not selected against outcomes.
The older coefficient was calibrated with uniform positional weights; this
study has area-lumped weights. Therefore it is a legacy scale reference, not a
newly calibrated optimum or numerically equivalent 2D/3D smoothing strength.
The initial draft considered normalized weight one, but the setup audit found
that would be approximately 2,693 times this historical coefficient. No
optimization used that draft.

Use the previous unrestricted face budget: 100 Adam updates, constant learning
rate .3, epsilon .01, betas .9/.999. Keep training forward tolerances at relative
5e-4, absolute 1e-10 and adjoint relative tolerance 5e-4. No inverse restart,
line search, additional determinant penalty or automatic coefficient search.
Physics includes the existing recorded adjoint CG/MinRes fallback; it is not
introduced by this experiment.

Before fitting, test the normal implementation and regularizer algebra/autograd
on CPU and check full implicit normal-only and combined derivatives through
the 3D forward solve. Finite difference directions and perturbations are
recorded; require under 2% relative error and perturbation-size disagreement.
Use tighter equilibrium settings for derivative validation if necessary and
record the distinction from training tolerances. Run a short all-four smoke
test, then restart all four main runs from neutral.

Save exact neutral controls/displacement/Adam moments, scalar traces and solver
receipts at every evaluation, checkpoints every ten updates, last accepted
state, best objective and best noninverted state. The shared latest saved step
is the primary comparison if a branch fails. Full endpoints are separate.
Report physical det(F), inversions and non-SPD activation tensors even when a
solve converges. Retain inverted states explicitly; never silently replace an
endpoint with an earlier noninverted checkpoint. This comparison cannot certify
mechanical stability without a separate Hessian check.

Report position RMS, normal-angle RMS, surface-gradient residual, target-relative
5 mm high-pass residual, motion, activation roughness and inverse gradient
history separately. Budget completion or a flat curve is not convergence.
Use shared cameras, scales, no displacement exaggeration, and flat-shaded face
and mouth-region plots. Export the target skin only; target volume deformation
is unknown.
