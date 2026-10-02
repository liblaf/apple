# Smile inverse solver review — 2026-09-23

Follow-up: the [fixes and verification](2026-09-23-smile-inverse-fixes.md) are now recorded separately. The findings below describe the code at review time.

Review only: no face solves, inverse experiments, solver edits, or historical artifact changes were made. Inspection used the current working tree at HEAD `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, including its existing uncommitted changes. Small CPU derivative and analytic checks were run with the repository interpreter (Torch 2.12.0).

**Recommendation:** reuse the loss and stress maps from `exp/2026/09/21/stress-activation-loss`, integrate the current forward solver explicitly, and address the issues below before launching the two four-stage chains. Neither historical runner is ready to run unchanged.

## Findings before experiments

### 1. P1: the reusable implicit backward uses mutable forward state

[`_diff_forward.py`](../../src/liblaf/apple/inverse/_diff_forward.py) saves only material tensors in `setup_context` (lines 124–130). Its adjoint and mixed derivative use `ctx.forward.state` (lines 150–153), which belongs to the most recent solve. The returned output also aliases `forward.state.u` (line 120).

Consequently, a second forward before the first backward can produce the wrong derivative. A CPU analytic reproduction used `E(u, theta) = u^3/3 - theta*u`, on the positive equilibrium `u = sqrt(theta)`. At `theta = 4`, immediate backward returned the correct `du/dtheta = 0.25`; after a second forward at `theta = 9`, backward of the first output returned `0.16666666666666666`.

Save an independent solved displacement and the relevant boundary state per autograd invocation; construct the adjoint at that saved state and restore mutable materials/boundaries afterward. The newer [`joint_equilibrium.py`](../../exp/2026/09/21/joint-activation-material-mandible/src/joint_equilibrium.py), lines 328–350, already demonstrates this pattern, including reconstruction of contact state.

Scope: the old stress chain currently runs backward immediately for every accepted trial, so this reproduction does **not** establish that its saved results have corrupted gradients. It is a correctness issue in the reusable API and a blocker for batching evaluations or separate component-gradient calculations across forwards.

### 2. P1: solver success and accepted-state residuals need to be enforced at the inverse boundary

The reusable wrapper ignores the result of `forward.step()` (lines 118–120) and consumes `solution.params` without checking adjoint success (lines 150–153). CPU mocks with either solver reporting `success=False` still returned a gradient without raising. Check success, finiteness, and the true residual before using the result; handle a zero adjoint right-hand side explicitly.

The historical stress adapter partially protects callers: [`stress_physics.py`](../../exp/2026/09/21/stress-activation-loss/src/stress_physics.py), lines 350–368 and 382–403, checks forward success and an independently evaluated adjoint residual. However, its forward receipt copies the PNCG convergence state's gradient. The current [`Pncg.step`](../../src/liblaf/apple/solvers/optim/pncg/_pncg.py), lines 156–169, updates that criterion using the gradient computed **before** the accepted displacement update. It does not establish the force norm at the returned equilibrium.

The new inverse adapter should recompute `r = grad(E)` at the returned displacement, use one recorded acceptance threshold, and check geometry/contact separately. Use the physical, unshifted Hessian for the implicit derivative. Forward Newton damping is a search device and must not be differentiated as physical stiffness.

### 3. P1: the requested final gradient ratio cannot generally be a convergence condition

For an unconstrained, interior L2-only optimum of `J = L2 + eta R`,

    grad(L2) + eta grad(R) = 0.

If those component gradients are nonzero, their norm ratio is exactly **1**, in any common norm. If both vanish, the ratio is undefined. Thus a nonzero final ratio of 0.1 and exact interior stationarity are incompatible. A normal term or active constraints can change this balance.

Treat 0.1 as the intended regularization strength at a declared, nonzero-stress pilot checkpoint, or as a diagnostic at a finite-budget endpoint. Freeze the coefficient before the compared chains, and report the achieved terminal ratio separately from stationarity. Do not continuously rebalance the loss, and do not promise that all eight stationary endpoints will have ratio 0.1.

The existing [`20-calibrate-smoothness.py`](../../exp/2026/09/21/stress-activation-loss/src/20-calibrate-smoothness.py), lines 40–76, uses gradients from zero stress, a different probe state for the regularizer gradient, and candidate multipliers 3, 30, 300, and 3000. It selects strong roughness reduction, not the requested weak-gradient criterion. Replace this calibration.

### 4. P2: a learned-axis zero-amplitude cell can remain stuck

The existing [`activation_models.py`](../../exp/2026/09/21/stress-activation-loss/src/activation_models.py), lines 103–115 and 145–180, correctly uses separate amplitude and unit-axis controls. This is preferable to `Q = v v^T`, whose entire derivative vanishes at `v = 0`. Nevertheless, at amplitude zero the axis derivative vanishes, and the amplitude derivative tests only the current axis.

A CPU reproduction with `J(Q) = diag(1,-1,1):Q`, zero amplitude, and the template's default x-axis gives control gradient `(1,0,0,0)`. A projected gradient step returns the identical zero tensor indefinitely, although `Q = 0.1 e_y e_y^T` is feasible and lowers the loss to `-0.1`.

For stage 4, preserve the parent axis when available. At zero amplitude, check the smallest eigenvalue of the symmetric total tensor gradient; a negative eigenvalue identifies an activating descent axis. Rotating the axis at exactly zero amplitude preserves the warm-start tensor exactly. Record such activation events. Do not call native-control stationarity a full rank-one stationarity certificate without this check. Stage 3 intentionally keeps its axis fixed and should not perform that axis update.

### 5. P2: neither runner binds the requested mechanics and new solver automatically

The closest template is the old no-skin [`stress-activation-loss`](../../exp/2026/09/21/stress-activation-loss/docs/00-protocol.md) study. Its maps, loss definitions, fresh stage optimizers, and physical tensor transfers are reusable. Its runtime is not current: `stress_physics.py`, lines 15–18, imports `liblaf.peach`, which is absent from the current environment; lines 203–209 hardcode old tolerances; lines 272–282 install old PNCG. Its source-hash validation must be regenerated for a new implementation, not bypassed.

The newer [`20-fit-smile.py`](../../exp/2026/09/22/solver-performance/src/20-fit-smile.py) instead loads a skin-on model, uses a target-motion-normalized L2 objective, and has no normal term. It always optimizes jaw pose, projects stress eigenvalues into `[0,10]`, and inherits a 36-expression smoothness weight. `pose_first=False` and `jaw_weight=0` do not freeze the jaw. Its `install_arm_runtime` (line 331) installs the older hybrid adapter; it does not select the new neutral Newton driver's backend.

The old no-skin fixture also disables contact and uses a different stress reference/material setup. **Removing skin does not authorize silently removing contact or changing fixation/materials.** Reuse its loss/maps with an explicit model manifest: no membrane energy or membrane prestress, the intended fixed passive materials and constraints, and an explicit collision policy. Keep the outer surface as an observation mesh. Recommended jaw setting for this stress comparison is fixed neutral; user preference was requested separately and is not needed to complete this review.

### 6. P2: outer step acceptance must distinguish stagnation from equilibrium error

The old [`run_support.py`](../../exp/2026/09/21/stress-activation-loss/src/run_support.py), lines 390–443, checks raw Armijo decrease and only then computes the accepted trial's adjoint. This saves rejected-trial adjoints, but it does not establish that tiny loss changes exceed forward-solve error. The old physics hardcodes relative forward/adjoint tolerances of `5e-4`.

Reuse bounded proposals and transactional acceptance with the new forward adapter. Before accepting tiny progress or declaring stationarity, repeat the relevant evaluation with tighter equilibrium/adjoint accuracy and compare the loss and directional gradient. A residual correction `p dot r` can be a useful first-order diagnostic; it is not an error certificate. Exhausted line search should retain the accepted checkpoint and report failure, rather than imply convergence or silently promote a child stage.

## Loss definition and checked balance

Retain the existing frozen reference-area weights and oriented triangle unit-normal chord loss. Let

    P = (1/3) sum_v w_v ||x_v - target_v||^2, in mm^2
    N = sum_t w_t ||n_t - target_normal_t||^2
    L2 = P / l_ref^2
    J = L2 + beta N + eta R, beta in {0, 1}.

The positional error is a **3D vector RMS** when specifying 2 mm. Then

    l_ref = 2 / sqrt(3 * (2 - 2 cos(5 degrees)))
          = 13.236093032531715 mm.

This agrees with [`stress_study.py`](../../exp/2026/09/21/stress-activation-loss/src/stress_study.py), lines 21 and 93–99, and [`surface_normal.py`](../../exp/2026/09/21/normal-matching-face/src/surface_normal.py), lines 57–66. A direct CPU check gave:

| Reference error | Weighted loss |
| --- | ---: |
| 2 mm positional vector RMS | 0.007610603816508936 |
| 5 degree triangle normal error | 0.007610603816508934 |

The equality calibrates these reference errors; an arbitrary distribution with 5 degree angular RMS need not have exactly the same average chord loss. Report normal-angle RMS in degrees separately. Use the chord loss for optimization, avoiding an `acos` objective near aligned normals. Freeze correspondence, triangle winding, and reference-area weights across all eight fits.

## Tensor smoothness and gradient reporting

Use the existing within-muscle reference graph:

    R(Qhat) = ell^2 / V_active * sum_(i,j) c_ij ||Qhat_i - Qhat_j||_F^2
    Qhat = Q / Qref, ell = 5 mm.

Conductance is shared-face area divided by centroid distance, multiplied by the harmonic mean of muscle fractions. The graph joins same-muscle neighbors. Penalize the full tensor in all stages, rather than scalar amplitudes or signed axis coordinates. No magnitude penalty is requested.

At the same valid calibration state, evaluate the L2-only and unweighted R gradients in common Mandel stress coordinates. A declared mesh-aware choice is the dual effective-volume norm, with `m_i = V_i / V_active`:

    ||g||_(M^-1) = sqrt(sum_i ||g_i||^2 / m_i)
    eta = 0.1 * ||grad_Qhat L2||_(M^-1) / ||grad_Qhat R||_(M^-1).

Do not calibrate at zero stress: the quadratic regularizer gradient there is zero. Freeze one eta across the eight fits for the primary comparison. Record the actual ambient tensor-gradient ratio at every endpoint and constrained stationarity separately. Raw 6-, 1-, and 4-coordinate gradient norms are not comparable. An ambient ratio also does not describe only the feasible directions of a constrained stage.

## Four stages, separately for each loss column

In this stress convention `W_active = Q:(F^T F - I)/2`, so `P_active = F Q`; positive-semidefinite Q has the intended contraction sign. This constrains the activation tensor, not the actual deformation gradient.

| Stage | Tensor and independent controls per active tetrahedron               | Parent transfer                                                           |
| ----- | -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1     | Arbitrary symmetric Q; 6 Mandel coordinates                          | Zero stress, freshly verified no-skin equilibrium                         |
| 2     | Q positive semidefinite; 6 coordinates, PSD projection, no upper cap | Clip negative eigenvalues of the same-loss stage 1 endpoint               |
| 3     | Q = a n n^T, a >= 0; 1 scalar, fixed n                               | Keep the largest eigenvalue/eigenvector of the same-loss stage 2 endpoint |
| 4     | Q = a n n^T, a >= 0,                                                 |                                                                           |

Carry displacement only as a forward seed. Re-equilibrate and recompute gradients after each transfer. Reset Adam moments and convergence history at each stage. Save parent hashes, projection error, minimum eigenvalue, zero-amplitude count, and eigenvalue-gap diagnostics. A repeated largest eigenvalue makes the stage-3 axis nonunique and should be recorded. The two loss columns have independent parent chains.

## Validation completed and remaining

Completed in this review:

- CPU double-precision `torch.autograd.gradcheck` passed for all four stress maps at smooth interior controls.
- The oriented normal-loss CPU gradcheck passed on a nondegenerate triangle rotated 5 degrees.
- PSD projection returned minimum eigenvalue `-5.56e-18` (roundoff); stage-3 to stage-4 tensor transfer had zero maximum error in the checked fixture.
- The physical loss balance above matched within `1.74e-18`.
- The delayed-backward, ignored-solver-failure, and zero-amplitude stationary-control counterexamples reproduced.
- Environment lookup confirmed `liblaf.peach` is absent and `liblaf.apple.solvers` is available.

These checks do not validate the full-face implicit gradient with the new solver. Before the production chains: add regression coverage for saved backward state and failures, validate accepted-state force and unshifted adjoint residuals, then check L2, normal, and smoothness directional derivatives on the actual no-skin model at neutral and a valid nonzero-stress state. Use a finite-difference step/tolerance sweep, including feasible one-sided checks at active constraints. Check projected warm starts and zero-amplitude axis release independently. Count forward/adjoint evaluations and solver work alongside time and fit; equal outer steps do not imply equal computational cost.

No old optimizer learning rate or warm-checkpoint performance result is treated as validated for this new objective, model, or parameterization.
