# Stable Neo-Hookean 2D unreachable-target inverse experiment

## Conclusion

This experiment reproduces the current no-skin inverse-physics pipeline in a controlled 2D plane-strain setting and separates three effects that can all appear as a “bumpy solution”:

1. The broad mound is the expected continuum-scale response of a localized muscle patch beneath a stiff SMAS, with the bottom and sides fixed, while the objective requests an almost uniform upward motion of the free top.
2. Fine structure appears before inversion because the activation-to-surface map is poorly identifiable. Per-triangle controls add spatial degrees of freedom much faster than top-boundary observations are added, and many control combinations have very weak surface response.
3. The largest late-stage bumps are nonphysical branch exploitation. Stable Neo-Hookean remains finite through singular and inverted states; “stable” does not mean that it enforces positive determinants.

The experiment does **not** prove that the nonlinear target is globally unreachable. It shows that the tested Adam trajectories did not reach it before determinant failure, and it supplies local tangent evidence explaining why the inverse problem is poorly conditioned.

The relationship between resolution and bumpiness is not monotone in the three tested free-control meshes. At closely matched data loss, the 100×10 mesh is bumpier than both 50×5 and 200×20. All finest-pair roughness convergence gates fail. By contrast, the regularized family becomes smoother under refinement at matched loss. Resolution therefore changes which weak inverse modes Adam selects; it is not merely “more elements produce more bumps.”

## Corresponding 3D experiments

The current algorithmic reference is the no-skin stretch baseline in [`unreachable-toy-skin-tetwild`](../../../../06/10/unreachable-toy-skin-tetwild/src/_toy_skin_tetwild.py), with its recorded [baseline result](../../../../06/10/unreachable-toy-skin-tetwild/data/20-stretch-lr001/baseline/20-toy-tetwild-stretch-lr001-l2-no_skin-prestrain0-activation-per_tet-summary.json). It uses:

- no skin energy;
- a `+0.1` displacement target on the free top;
- unrestricted six-component symmetric `activation_inv` per active tetrahedron;
- full three-component mean-squared displacement error;
- nonlinear forward equilibrium and an implicit adjoint;
- Adam with learning rate `0.05`, at most 160 updates, patience 20, and minimum loss improvement `5e-6`.

The likely historical source of the original unreachable-target visualization is [`30-inverse-unreachable-stable-neo-hookean.py`](../../../../01/28/smas/src/30-inverse-unreachable-stable-neo-hookean.py). That January prototype uses the older activation/material API, six activation components per muscle cell, a surface-wide target, and `lambda / mu = 3`. It is useful provenance, but it is not the current reference pipeline.

Apple’s current constitutive and adjoint implementations are in:

- `src/liblaf/apple/warp/fem/_stable_neo_hookean.py`;
- `src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py`;
- `src/liblaf/apple/warp/fem/func/_misc.py`;
- `src/liblaf/apple/inverse/_diff_forward.py`.

### Exact matches and deliberate deviations

| Feature | Current June 3D no-skin baseline | This 2D experiment | Assessment |
| --- | --- | --- | --- |
| Skin | Disabled | Absent | Exact intent match |
| Target | Free top, `(0, +0.1, 0)` | Free top interior, `(0, +0.1, 0)` | Exact dimensional analogue |
| Loss | Mean squared three-component displacement residual | Residual embedded as `(ux, uy, 0)` and divided by `3 N_top` | Exact normalization match |
| Constitutive law | Apple `StableNeoHookean` and `StableNeoHookeanActive` | Exact plane-strain restriction | Exact energy match |
| Active kinematics | `G = F @ Ainv` | `G_2 = F_2 @ Ainv_2` | Exact plane-strain restriction |
| Baseline controls | Six symmetric components per active tetrahedron | Three in-plane symmetric components per muscle triangle | Deliberate user-requested restriction |
| Baseline smoothing/bounds | None | None in `free` | Match |
| Inverse optimizer | Adam, LR 0.05, 160 steps, patience 20, delta `5e-6` | Same | Match |
| Gradient | Nonlinear equilibrium plus implicit adjoint | Same equations, explicitly assembled | Algorithmic match |
| Forward solver | Native Warp tetrahedra with PNCG | P1 triangles with damped Newton, Armijo line search, and sparse direct solves | Important backend/solver deviation |
| Poisson ratio | Fat/muscle 0.49; SMAS 0.35 | Fat/muscle/SMAS all 0.49 | Deliberate requested material change |
| Material interfaces | TetWild fraction sampling | Sharp mesh-aligned triangle labels | Controlled simplification |
| Geometry | 3D `1 × 0.1 × 1` volume | 2D `1 × 0.1` strip with a smaller near-edge muscle patch | Equivalent experiment, not a literal geometric section |

Thus, the constitutive energy, loss scaling, optimizer, and implicit-differentiation semantics match the current pipeline. The result should not be described as a literal execution of Apple’s tetrahedron-only Warp FEM.

## Mathematical formulation

### Exact plane-strain Stable Neo-Hookean energy

Embed the in-plane deformation and activation as

\[
\bar F =
\begin{bmatrix}
F & 0 \\
0 & 1
\end{bmatrix},
\qquad
\bar A^{-1} =
\begin{bmatrix}
A^{-1} & 0 \\
0 & 1
\end{bmatrix}.
\]

For a muscle triangle, the three activation degrees of freedom are

\[
A^{-1} = I +
\begin{bmatrix}
a_{xx} & a_{xy} \\
a_{xy} & a_{yy}
\end{bmatrix}.
\]

The off-diagonal basis changes both symmetric entries by the same amount. It is tensor shear, not engineering shear with an extra factor of one half. In Apple’s six-component order, the retained components are `[xx, yy, xy]`, corresponding to indices `[0, 1, 3]`; the out-of-plane components are zero.

Let

\[
G = F A^{-1}, \qquad J = \det G.
\]

The per-unit-reference-area energy is

\[
\psi(G) =
\frac{\mu}{2}\left(\lVert G\rVert_F^2 - 2\right)

- \mu(J - 1)
- \frac{\lambda}{2}(J - 1)^2.
\]

This is the exact plane-strain restriction of Apple’s 3D expression because

\[
\lVert \bar G\rVert_F^2 - 3 = \lVert G\rVert_F^2 - 2,
\qquad
\det \bar G = \det G.
\]

The Lamé parameters use the 3D/plane-strain conversion

\[
\mu = \frac{E}{2(1+\nu)},
\qquad
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}.
\]

All materials use `nu = 0.49`, so

\[
\frac{\lambda}{\mu} = \frac{2\nu}{1-2\nu} = 49.
\]

The Young’s moduli are 0.003 MPa for fat, 0.030 MPa for muscle, and 0.100 MPa for SMAS.

### Equilibrium, loss, and adjoint

The discrete equilibrium is

\[
R(u,a) = \partial_u \Pi(u,a) = 0.
\]

Only the non-fixed top nodes are observed. If their requested displacement is `(0, 0.1, 0)`, the data loss is

\[
L_{\mathrm{data}} =
\frac{1}{3N_{\mathrm{top}}}
\sum_{i=1}^{N_{\mathrm{top}}}
\left[u_{x,i}^2 + (u_{y,i}-0.1)^2\right].
\]

The factor of three deliberately matches the current 3D loss. The native top vector-error RMS therefore satisfies

\[
e_{\mathrm{RMS}} = \sqrt{3 L_{\mathrm{data}}}.
\]

With

\[
H = \partial_u R,
\qquad
C = \partial_a R,
\]

the adjoint and inverse gradient are

\[
H p = -\partial_u L,
\qquad
\nabla_a L = C^T p + \eta Q a.
\]

The free and tied baselines have `eta = 0`. The regularized variant uses `eta = 1e-4` and

\[
\frac{\eta}{2}
\sum_{K\sim L}
\frac{|e_{KL}|}{d_{KL}}
\lVert a_K-a_L\rVert^2,
\]

which is a mesh-aware graph approximation to an activation `H1` seminorm.

## Geometry, discretization, and control spaces

The domain is `[0,1] × [0,0.1]`. SMAS occupies `0.04 <= y <= 0.06`. Muscle replaces SMAS in `0.06 <= x <= 0.22`, `0.04 <= y <= 0.06`. The bottom, left, and right boundaries are fixed in both in-plane components. The top interior is free and observed. Each square is split along the same southwest-to-northeast diagonal.

Three nested meshes and three control variants were run:

- `free`: one unrestricted three-component tensor per muscle triangle;
- `tied`: every refined mesh is tied to the 16 triangle groups of the 50×5 muscle partition, so all meshes have 48 activation DoFs;
- `regularized`: the same per-triangle controls as `free`, plus the graph-H1 penalty above.

The user-requested control contract applies exactly to `free` and `regularized`: every muscle triangle independently owns `a_xx`, `a_yy`, and `a_xy`. The `tied` family intentionally violates per-triangle independence by sharing those three components within coarse groups; it is included only as a diagnostic control-space ablation, not as a primary realization of the requested inverse model.

| Mesh | Spacing | Total triangles | Muscle triangles | Free state DoFs | Free/regularized activation DoFs | Tied shared-group DoFs | Effective in-plane top outputs | Loss entries including zero z |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50×5 | 0.020 | 500 | 16 | 490 | 48 | 48 | 98 | 147 |
| 100×10 | 0.010 | 2,000 | 64 | 1,980 | 192 | 48 | 198 | 297 |
| 200×20 | 0.005 | 8,000 | 256 | 7,960 | 768 | 48 | 398 | 597 |

The control count of the free family grows as area, `O(h^-2)`, while independent boundary outputs grow as length, `O(h^-1)`. The embedded zero z component changes loss normalization but supplies no additional rank.

![Material setup](../data/20-paraview/setup-materials.png)

## Numerical validation

The summary is complete: the requested Poisson ratio was used, all nine selected comparison states passed the determinant/equilibrium selection checks, and every free or regularized muscle triangle owns exactly three activation DoFs.

| Check | Relative error | Result |
| --- | ---: | --- |
| Element energy directional derivative | `5.531e-10` | pass |
| Element Hessian action | `5.642e-10` | pass |
| Mixed activation derivative | `1.450e-10` | pass |
| Assembled single-component implicit adjoint | `1.104e-7` | pass |
| Independent random-direction implicit gradient | `5.353e-9` | pass |

The assembled adjoint check had a maximum perturbed-equilibrium residual of `1.931e-14`. The independently seeded direction check used `epsilon = 2e-5`.

All selected comparison states also passed a post-analysis positive-definiteness check on `Ainv`; there were no nonpositive activation eigenvalues. The smallest selected `Ainv` eigenvalue was `0.02907` in the 50×5 free/tied state.

Some non-selected evaluations near or after determinant failure did not satisfy the `1e-9` forward tolerance. Consequently the global diagnostics “all forward solves converged” and “all pre-inversion solves equilibrated” are false. This does not change the selected-state tables below, whose reported residuals all satisfy the tolerance. It does mean that post-failure animation frames are diagnostic failure evidence rather than valid equilibria.

## Selected determinant-admissible results

For each case, the comparison state is the minimum-objective, strictly equilibrated, orientation-preserving evaluated state before the first observed orientation failure. This is a prefix of sampled evaluations; it is not a proof that the continuous path between Adam iterates remained admissible.

The primary free-control results are:

| Mesh | Selected step | First invalid step | Data loss | Top error RMS | Common-grid high-pass RMS | Activation jump RMS | min det F | min det G | min det Ainv | Equilibrium residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50×5 | 11 | 12 | `2.978459e-3` | `9.452712e-2` | `1.293011e-3` | `1.036750e-1` | 0.6652 | 0.1834 | 0.02660 | `1.524e-11` |
| 100×10 | 9 | 10 | `2.841382e-3` | `9.232630e-2` | `3.364920e-3` | `2.012695e-1` | 0.5589 | 0.5589 | 0.13849 | `9.063e-12` |
| 200×20 | 9 | 11 | `2.877740e-3` | `9.291513e-2` | `1.752960e-3` | `8.540031e-2` | 0.5322 | 0.4446 | 0.14260 | `8.471e-10` |

The initial loss is `0.0033333333`, corresponding to a `0.1` vector-error RMS. Even the selected free states retain about 92–95% of that target magnitude as RMS error. The tested admissible trajectories improve the fit but remain far from the requested uniform lift.

The 100×10 selected states illustrate the effect of the control model:

| Variant | Activation DoFs | Step | Data loss | Regularizer | Top error RMS | Native high-pass RMS | Activation jump RMS | First invalid step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Free | 192 | 9 | `2.841382e-3` | 0 | `9.232630e-2` | `3.612746e-3` | 0.20127 | 10 |
| Tied | 48 | 9 | `2.755753e-3` | 0 | `9.092446e-2` | `3.377022e-3` | 0.13686 | 11 |
| Regularized | 192 | 16 | `2.643802e-3` | `9.313257e-5` | `8.905844e-2` | `2.706447e-3` | 0.06192 | 17 |

This endpoint table is useful but not a pure roughness comparison: the cases reach their selected prefixes at different optimizer progress and different data loss. The matched-loss analysis below controls that confound.

![Selected free-control resolution comparison](../data/20-paraview/free-resolution-geometry.png)

![Signed shear ActivationXY](../data/20-paraview/free-resolution-signed-activation-xy.png)

## Resolution versus bumpiness at matched loss

All profiles were interpolated onto the same 1,921-point grid over `x = 0.02...0.98`. Bumpiness is the RMS difference from a Gaussian-smoothed top profile with physical standard deviation `0.02`. Fourier power uses fixed spatial-frequency bands; `12...24` cycles per unit length remains below the coarsest native Nyquist frequency.

To compare equivalent optimizer progress, the analysis defines a target loss 75% of the way from the common initial loss to the 50×5 selected-prefix loss for each variant, then selects the closest equilibrated, determinant-admissible evaluation on every mesh. The relative loss mismatch is below 1% in all nine rows.

| Variant | Mesh | Step | Actual data loss | Loss mismatch | Common-grid target RMS | High-pass RMS | PSD 12–24 | Activation jump RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Free | 50×5 | 9 | `3.056062e-3` | 0.362% | 0.095714 | `1.153229e-3` | `6.391111e-7` | 0.05539 |
| Free | 100×10 | 7 | `3.055786e-3` | 0.371% | 0.095655 | `1.428235e-3` | `5.365331e-7` | 0.15400 |
| Free | 200×20 | 7 | `3.057972e-3` | 0.300% | 0.095738 | `1.021426e-3` | `3.330026e-7` | 0.06041 |
| Tied | 50×5 | 9 | `3.056062e-3` | 0.362% | 0.095714 | `1.153229e-3` | `6.391111e-7` | 0.05539 |
| Tied | 100×10 | 7 | `3.042808e-3` | 0.795% | 0.095444 | `1.769996e-3` | `8.133099e-7` | 0.10542 |
| Tied | 200×20 | 7 | `3.037534e-3` | 0.966% | 0.095406 | `1.144873e-3` | `2.192066e-7` | 0.03327 |
| Regularized | 50×5 | 8 | `3.078105e-3` | 0.311% | 0.096031 | `6.838468e-4` | `1.619178e-7` | 0.02534 |
| Regularized | 100×10 | 8 | `3.062886e-3` | 0.185% | 0.095759 | `5.467157e-4` | `3.716322e-8` | 0.01660 |
| Regularized | 200×20 | 8 | `3.059919e-3` | 0.282% | 0.095718 | `4.084092e-4` | `1.379252e-8` | 0.01699 |

The evidence supports four restrained conclusions:

1. Free-control bumpiness is resolution-sensitive but nonmonotone: the middle mesh is roughest at matched loss.
2. Tying the control space to 48 DoFs does not remove the middle-mesh peak. Expanding the control dimension is therefore not sufficient to explain it. State discretization, nearly incompressible stiffness, the common diagonal, and optimizer path remain confounded candidate contributors; this experiment does not isolate their individual effects.
3. The graph-H1 penalty consistently lowers both high-pass displacement and activation jumps at matched loss. Its high-pass RMS decreases from `6.84e-4` to `4.08e-4` as the mesh is refined.
4. One fixed regularization weight is only a diagnostic. It is not a calibrated regularization path or a proof that `1e-4` is optimal.

The selected-prefix endpoints themselves are not converged. For the 100×10 to 200×20 pair, the high-pass change is 92.0% for free, 73.1% for regularized, and 101.9% for tied; the corresponding profile correlations are 0.920, 0.992, and 0.842. Every predeclared gate requiring at most 5% changes and correlation at least 0.995 fails.

Free-control restriction to the next coarser physical partition leaves 20.5% subgrid activation at 100×10 and 15.5% at 200×20. The control fields therefore contain substantial spatial content that a coarser parameterization cannot represent.

![Matched-loss bumpiness and activation jumps](../data/25-paraview-analysis/matched-loss-bumpiness-progress-0p75.png)

## Local tangent identifiability

At zero activation, define the top-boundary sensitivity

\[
S_h = -P H_h^{-1} C_h.
\]

The data-only Gauss–Newton curvature is proportional to `S_h^T S_h`; a small singular value of `S_h` is therefore a nearly flat inverse direction. Elliptic propagation from a buried muscle patch to the top attenuates short-wavelength control changes, and neighboring changes can cancel at the observation boundary.

The following ranks use a relative singular-value tolerance of `1e-10`. “Physically resolved rank” counts a unit activation response larger than the declared displacement resolution `1e-4`.

| Mesh | Controls | In-plane outputs | Dimension-guaranteed nullity | Numerical rank | Numerical nullity | Physically resolved rank | Target projection residual fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50×5 | 48 | 98 | 0 | 26 | 22 | 21 | 0.7390 |
| 100×10 | 192 | 198 | 0 | 54 | 138 | 39 | 0.6747 |
| 200×20 | 768 | 398 | 370 | 100 | 668 | 62 | 0.6811 |

The finest mesh has at least 370 null directions from dimension alone and 668 at the stated numerical tolerance. Even the coarser systems are rank deficient because the forward map is smoothing and because only the top boundary is observed.

The projection residual shows that the requested uniform lift is not in the range of the **initial linearized** response. This is a local identifiability statement, not a nonlinear global reachability certificate.

![Initial tangent singular-value spectra](../data/25-paraview-analysis/initial-tangent-singular-value-spectra.png)

## Inverted global optima and double inversion

The polynomial Stable Neo-Hookean determinant term is

\[
-\mu(J-1) + \frac{\lambda}{2}(J-1)^2.
\]

It has no `-log J` term or other barrier as `J` approaches zero from above. The energy is finite at `J = 0` and defined for `J < 0`. Robust evaluation through inversion is distinct from enforcing physical orientation.

In muscle,

\[
\det G = \det F\,\det A^{-1}.
\]

Consequently, simultaneous sign reversals of `F` and `Ainv` can leave `det G` positive. The 50×5 global best demonstrates this directly:

| Mesh | Global-best step | Data loss | High-pass RMS | min det F | min det G | min det Ainv | Residual | Use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 50×5 | 138 | `2.533200e-3` | `5.777214e-3` | -17.4238 | +0.3015 | -0.2648 | `1.707e-13` | Equilibrated double-inversion failure evidence |
| 100×10 | 10 | `2.704414e-3` | `4.200899e-3` | -0.1647 | -0.1647 | +0.0568 | `1.033e-10` | Equilibrated deformation-inversion evidence |
| 200×20 | 11 | `2.772025e-3` | `2.378569e-3` | -65.4054 | -0.1523 | -0.0120 | `2.512e-6` | Invalid and non-equilibrated; diagnostic only |

For 50×5, accepting the invalid branch lowers the loss by `4.45e-4` relative to the selected prefix while increasing high-pass RMS from `1.59e-3` to `5.78e-3`. The small equilibrium residual rules out forward nonconvergence as the explanation for that particular invalid state. The optimizer has found an equilibrated but nonphysical branch.

The inverse controls are unbounded and are updated by Adam without an `Ainv` positive-definiteness parameterization. The forward line search decreases elastic energy for fixed controls; it does not constrain an Adam update in control space. Smoothness alone also does not enforce determinant positivity, which is why every regularized trajectory eventually crosses an orientation boundary.

![Determinant history](../data/20-paraview/determinant-history.png)

![Admissible prefix versus invalid global best](../data/20-paraview/admissibility-transition.png)

## Cause of the bumps

### Continuum-scale mound

The actuator occupies only a short segment near the left edge and is separated from the observed top by fat. A localized eigenstrain cannot directly prescribe an independent vertical displacement at every top point. The fixed bottom and sides further restrict the deformation. A broad mound or arch is therefore expected even for a smooth activation field; this part should not be labeled a numerical artifact.

### Pre-inversion fine structure

Each free muscle triangle carries a discontinuous, piecewise-constant tensor. Refining both directions by two multiplies the free activation count by four, but only doubles the number of independent top samples. The tangent spectra show a rapidly growing null/weak subspace. Adam can populate these weak directions because they change the loss little; alternating triangle controls can partly cancel before their response reaches the top.

The matched-loss experiment supports this mechanism. Activation jump RMS and top high-pass RMS move together qualitatively, and regularization reduces both without sacrificing the matched target RMS. However, the nonmonotone free and tied curves show that control-space dimension alone is insufficient as an explanation.

### Post-inversion amplification

As `det Ainv`, `det F`, or `det G` approaches zero, the forward tangent becomes difficult and Adam can cross into a different equilibrium branch. Because Stable Neo-Hookean has no determinant barrier, the objective may continue to improve on that branch. The video shows a sharp distinction between step 9, the 100×10 comparison state, and step 10, the first orientation-invalid evaluation. Later frames are failure evolution, not physical inverse solutions.

### Near-incompressibility and locking

With `nu = 0.49`, `lambda / mu = 49`. Displacement-only P1 triangles have elementwise constant strain and are susceptible to volumetric locking in nearly incompressible plane strain. Locking can make the strip artificially stiff, worsen conditioning, amplify the common mesh-diagonal bias, and force the inverse optimizer toward larger or more heterogeneous activations.

Locking is a plausible contributing factor, not an isolated cause in this experiment. Poisson ratio and element formulation were held fixed. There is no independent pressure degree of freedom, so the observed pattern should not be called pressure checkerboarding. A causal locking study would repeat the matched-loss experiment with a Poisson-ratio sweep and a mixed or higher-order element, and should also reverse or alternate the triangle diagonal.

## ParaView visualization

All geometry screenshots, charts, and animation frames were rendered through ParaView 6.1.1. The VTK files store reference coordinates and a displacement array; the renderer applies `WarpByVector` once. The receipts record input hashes and output dimensions:

- [primary render receipt](../data/20-paraview/render-receipt.json);
- [analysis render receipt](../data/25-paraview-analysis/render-receipt.json).

Primary assets:

- [primary step-by-step inverse evolution video](../data/20-paraview/evolution-step-by-step.mp4): 31 exact ParaView evaluation frames, steps 0–30, 2 fps, 15.5 s, 1800×1200 H.264/yuv420p, with no temporal interpolation; recorded as `evolution.step_by_step_video` in the receipt;
- [10 fps evolution preview](../data/20-paraview/evolution.mp4): the same 31 exact frames in 3.1 s;
- [initial frame](../data/20-paraview/evolution-initial.png), [middle frame](../data/20-paraview/evolution-middle.png), and [final frame](../data/20-paraview/evolution-final.png);
- [free-resolution geometry](../data/20-paraview/free-resolution-geometry.png);
- [signed shear `ActivationXY`](../data/20-paraview/free-resolution-signed-activation-xy.png);
- [free/tied/regularized comparison](../data/20-paraview/free-tied-regularized-control-comparison.png);
- [top profiles and fixed-scale high-pass signals](../data/20-paraview/free-resolution-top-profiles.png);
- [spatial spectra](../data/20-paraview/free-resolution-spatial-spectra.png);
- [determinant history](../data/20-paraview/determinant-history.png);
- [admissibility transition](../data/20-paraview/admissibility-transition.png);
- [matched-loss bumpiness](../data/25-paraview-analysis/matched-loss-bumpiness-progress-0p75.png);
- [initial tangent spectra](../data/25-paraview-analysis/initial-tangent-singular-value-spectra.png).

Each PNG has a `.pvsm` sidecar in the same directory for inspection and further ParaView editing.

## Limitations

- “Unreachable” is descriptive of the tested solves. No global nonlinear impossibility proof was attempted.
- Tangent ranks and projection residuals are local at zero activation.
- The selected prefix is checked only at stored evaluation states. It does not certify the interpolated path between Adam steps or every internal Newton iterate.
- Selected `Ainv` tensors were verified positive definite after the run, but the optimizer does not enforce this property by construction.
- Free refinement changes both the state discretization and inverse control space. Tied controls isolate the latter confound only partially because optimizer paths and admissibility margins still differ.
- The single regularization weight is diagnostic rather than calibrated at a full regularization path.
- The P1 triangle/Newton implementation matches Apple’s constitutive and adjoint equations but not its tetrahedral Warp/PNCG execution path.
- Sharp aligned material interfaces replace the fraction-sampled 3D geometry.
- The fixed triangle diagonal, Poisson ratio, element type, target magnitude, optimizer, and learning rate were not swept.
- The 2D model is plane strain per unit out-of-plane thickness and does not reproduce 3D lateral load spreading.

Recommended next controlled experiments are: an `Ainv = exp(S)` or Cholesky parameterization that guarantees positive definiteness; a regularization-weight path compared at matched data loss; a `nu` sweep; a mixed/higher-order displacement-pressure formulation; reversed/alternating mesh diagonals; and optimizer/learning-rate repeats.

## Outputs

Numerical source and evidence:

- [numerical runner](../src/10-run-stable-neo-hookean-resolution.py);
- [numerical summary](../data/10-stable-neo-hookean-resolution/summary.json);
- [analysis source](../src/15-analyze-stable-neo-hookean-resolution.py);
- [analysis summary](../data/15-resolution-analysis/analysis.json);
- [matched-loss table](../data/15-resolution-analysis/matched-data-loss-evolution.csv);
- [evolution diagnostics](../data/15-resolution-analysis/evolution-diagnostics.csv).

The numerical run completed in 422.32 seconds with NumPy 2.4.6, SciPy 1.17.1, and PyVista 0.48.4. The validated production run used Cherries debug mode and did not create a remote Comet record or an automatic Git commit.

## Reproduction

Run from the experiment group. The commands below deliberately use fresh `*-rerun` directories. The numerical, analysis, and rendering scripts refuse to overwrite nonempty output directories, which protects the validated evidence above.

```bash
cd exp/2026/08/24/unreachable-stable-neo-hookean-2d-resolution

DEBUG=1 \
CHERRIES_NAME='Stable Neo-Hookean 2D unreachable target resolution study' \
CHERRIES_TAGS='inverse,stable-neo-hookean,plane-strain,mesh-resolution,bumpiness,paraview' \
uv run python src/10-run-stable-neo-hookean-resolution.py \
  --output-dir data/10-stable-neo-hookean-resolution-rerun

DEBUG=1 \
CHERRIES_NAME='Analyze Stable Neo-Hookean 2D resolution study' \
CHERRIES_TAGS='analysis,mesh-resolution,bumpiness,identifiability' \
uv run python src/15-analyze-stable-neo-hookean-resolution.py \
  --input-summary data/10-stable-neo-hookean-resolution-rerun/summary.json \
  --output-dir data/15-resolution-analysis-rerun

pvpython src/20-render-stable-neo-hookean-resolution-paraview.py \
  --summary data/10-stable-neo-hookean-resolution-rerun/summary.json \
  --output-dir data/20-paraview-rerun \
  --fps 10

pvpython src/25-render-stable-neo-hookean-resolution-analysis-paraview.py \
  --analysis-dir data/15-resolution-analysis-rerun \
  --numerical-dir data/10-stable-neo-hookean-resolution-rerun \
  --output-dir data/25-paraview-analysis-rerun
```

## References

- [Stable Neo-Hookean Flesh Simulation (2018)](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf), for the constitutive model’s robustness under large deformation and inversion.
- [Oñate et al., linear triangles/tetrahedra for incompressible solids](https://www.scipedia.com/public/Onate_et_al_2004c), for the volumetric-locking limitation of low-order displacement elements.
- [Numerical analysis of an inverse elasticity problem and Tikhonov regularization](https://www.cambridge.org/core/journals/communications-in-computational-physics/article/abs/numerical-analysis-of-inverse-elasticity-problemwith-signorinis-condition/1B138749177D46854F355E8D8543C836), for regularization of ill-posed inverse elasticity.
- [Non-regularised inverse finite-element identification](https://upcommons.upc.edu/server/api/core/bitstreams/d4803107-e3db-4a5a-9ce7-57e21254dca5/content), for rank and conditioning issues in inverse finite-element systems.
