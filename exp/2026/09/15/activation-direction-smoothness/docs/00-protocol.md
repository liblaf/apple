# Four activation models with and without spatial smoothness

## Fixed comparison

This extends the 2026-09-14 2D parabola demo. The user selected the existing muscle band: the 1 × 0.1 rectangle has a muscle band at y∈[0.04,0.06] and passive fat above and below it. All 400 muscle triangles are independently controlled. The same 100 × 10 mesh, bottom/side fixation, E_muscle=0.03, E_fat=0.003, ν=0.49, and top-displacement targets h=0.05 and 0.20 are retained. Material parameters are fixed.

All cases use the experiment-local corrected Stable Neo-Hookean energy with physical J=det(F):

    W = μ/2 (||F B||²_F − 2) − μ(J−1) + λ/2 (J−1)², B=A⁻¹.

No constitutive model or determinant convention is varied.

| Mode | Inverse activation B | DoF per muscle triangle |
| --- | --- | ---: |
| Free activation | I+S, S unrestricted symmetric | 3 |
| Contraction-only, free directions | I+S, S symmetric positive semidefinite | 3 |
| Contraction-only, learned direction | I+s n(θ)n(θ)ᵀ, s≥0 | 2 |
| Contraction-only, fixed x-direction | diag(1+s,1), s≥0 | 1 |

The learned model has one axis per element, with both its strength and angle trainable. Its transverse active stretch is exactly one; the free-directions model can contract in both principal directions. All models start at B=I. Learned angles start at θ=0, and angle gradients are zero until strength becomes nonzero. This is an initialization choice, not anatomy.

## Loss and tuning

The data loss is mean free-top-node squared Euclidean displacement error (vector L2/MSE, with no extra factor 1/2 or coordinate averaging). The optimizer minimizes:

    L_total = L_data + α h² R,
    R = mean_(i,j shared edge within muscle) ||B_i−B_j||²_F.

R smooths the same tensor field in every model, counts both off-diagonal entries, and excludes muscle/fat boundary edges. The penalty is invariant to replacing a learned axis n with −n. This graph normalization is specific to this mesh.

The initial full-budget bracket is α∈{0,0.01,0.1,1}. Every candidate uses the same historical Adam settings: 1,200 updates, initial learning rate 0.03, decay 0.99 per update, betas (0.9,0.999), epsilon 1e−8. Only feasibility is projected. No amplitude cap or inverse loss-based step rejection is added. Sweeps start independently at identity.

An effective weight should reduce mean squared neighboring tensor differences by at least 75% relative to the same model without regularization; fitting-error changes must be displayed alongside that reduction. Prefer the smallest common weight that is effective for all models and both targets. Expand the bracket if necessary. The previously observed unregularized free-model forward failure at the larger target means shared-step comparisons are required, as well as explicitly labelled last-valid endpoints. Weight selection will be recorded after reading the runs; failed or short runs will not be silently compared as equally converged endpoints.

## Evidence and limits

Save full traces, displacement/control snapshots every 10 steps, final optimizer checkpoints, source hashes, forward failures, physical J, activation eigenvalues/singular values, raw data loss and total objective separately, R, tensor magnitude, and relative roughness. Validate control and smoothness derivatives, the full implicit objective gradient, all saved constrained activations, and final equilibrium residuals. Show equal-scale mesh views and both principal activation modes.

This compares finite Adam trajectories with different parameterizations. It does not establish inverse convergence, intrinsic globally optimal capacity, anatomical fiber directions, or physically bounded muscle strengths. Activation contraction does not preclude elastic expansion in F. The fixed-boundary near-incompressible target is a model/constraint conflict; it is not labelled mathematically unreachable without proof.
