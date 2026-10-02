# Constraining muscle activation to reduce bumpy inverse-physics artifacts

Status: experiment design, 2026-09-06. No new physics runs or numerical results are included. Repository inspected at `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.

Start with **bounded contraction along a fixed fiber field, plus a small-activation penalty and within-muscle spatial regularization**. Test each restriction separately before combining them. The intended outcome is a useful tradeoff between surface quality and target fit; a less accurate but smooth, mechanically admissible reconstruction may be preferable. A smooth nearly motionless result is not sufficient evidence of success.

**What the existing implementation establishes.** The current face runner optimizes six independent numbers per active tetrahedron and uses only surface displacement MSE. Those numbers form the symmetric offset `ActivationInv = A_inv - I`; the elastic deformation is `G = F @ A_inv`. They are not six physiological muscle activation channels. The unrestricted map can become singular or orientation-reversing. The saved upper-mouth report documents negative `det(F)` and `det(A_inv)` in Orbicularis oris and explicitly labels the historical inverse result nonconverged. Spatial oscillation, excessive magnitude, and inadmissible activation maps therefore need separate diagnostics.

The existing fat-transfer experiment supplies a suitable 3-D block and a volume-preserving fiber activation map. Its saved results support transmission of a prescribed activation mode to the surface, not the effectiveness of inverse regularization. Keep fat thickness, passive materials, skin settings, constraints, target loss, and observation mask fixed in the first comparison.

**Activation model.** Let `f_e` be a unit fiber in the reference configuration, `P_e = f_e f_e^T`, and `Q_e = I - P_e`. Use the scalar log contraction `a_e`:

$$
 A_e = e^{-a_e}P_e + e^{a_e/2}Q_e,
 \qquad A_e^{-1}=e^{a_e}P_e+e^{-a_e/2}Q_e,
 \qquad 0\le a_e\le a_{\max}.
$$

Pass the packed symmetric components of `A_inv - I` to the existing constitutive model. This gives one control per tetrahedron, positive active stretches, no independently fitted shear or transverse activation, and `det(A) = 1`. A fiber direction is an axis: `f` and `-f` give the same map. Store it in reference coordinates; do not re-estimate it from each deformed state.

For example, `a = -log(0.8)` prescribes 20% natural fiber shortening and transverse stretches `1/sqrt(0.8) ≈ 1.118`. This is a prescribed internal active strain, not necessarily the actual tissue shortening, because loading and constraints determine `F`.

Use 35% maximum natural shortening (`a_max = -log(0.65)`) as an initial **experimental cap**, then test 10%, 20%, and 35% on finalists. These values are sensitivity settings, not calibrated physiological limits. The multiplicative active-strain interpretation and its dependence on the passive law are discussed in [A Physiology-Guided Classification of Active-Stress and Active-Strain Approaches](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.685531/full).

The user's literal “minimal perpendicular activation” alternative should also be tested:

$$
 A_e(\gamma)=e^{-a_e}P_e+e^{\gamma a_e}Q_e,
 \qquad \gamma\in\{0,1/2\}.
$$

`gamma = 0` leaves transverse natural lengths unchanged but reduces active volume; `gamma = 1/2` preserves active volume. Compare these with the same fiber shortening and regularizers on representative targets. Neither has independently controlled transverse activation. Do not conflate the volume difference with the benefit of a fiber constraint. Active stress is a later constitutive-model comparison, since changing it in the first matrix would change both the control space and the mechanical law.

**Magnitude and spatial penalties.** For the scalar model use

$$
 \min_a\; E_{\mathrm{fit}}(u(a),u^*)
       +\lambda_m R_m(a)+\lambda_s R_s(a),
 \quad \text{subject to equilibrium and }0\le a\le a_{\max},
$$

$$
 E_{\mathrm{fit}}=
 \frac{\sum_{i\in\mathcal O}w_i\|u_i-u_i^*\|^2}
      {D^2\sum_{i\in\mathcal O}w_i},
 \qquad
 R_m=\frac{\sum_e V_e a_e^2}{V_m a_{\rm ref}^2},
$$

$$
 R_s=\frac{\ell^2}{V_m a_{\rm ref}^2}
       \sum_{(e,j)\in\mathcal E}
       \frac{S_{ej}}{d_{ej}}(a_e-a_j)^2.
$$

Here `w_i` is a reference surface-area weight, `V_m = sum V_e` is active reference volume, `S_ej` is shared-face area, and `d_ej` is reference centroid distance. Each unordered edge appears once. The graph connects face-sharing active tetrahedra within the same muscle/compartment, never through fat or between different muscles. There is no imposed zero-activation boundary at the muscle surface. For the block choose `a_ref = -log(0.8)` and `ell = 0.1 L`, where `L` is the footprint width. Fix `D` to the clean target's displacement RMS across its clean/noisy/mismatched variants.

The graph term is a physically scaled discrete smoothness penalty; on a general tetrahedral mesh it is not automatically a consistent continuum Laplacian. Check refinement sensitivity. On mixed-material face cells use muscle-volume weighting `V_e * MuscleFraction_e` for magnitude statistics and document the corresponding interface weighting. Preserve the active mask across methods, and report results by muscle fraction so that small-fraction cells cannot hide extreme controls.

Use L2 magnitude first: it discourages large peaks. L1 would instead encourage sparsity and can concentrate activity; reserve it for a separate hypothesis. Spatial smoothness alone permits a large constant field, while magnitude alone permits small rapid oscillations, so both ablations are needed. A graph penalty only encourages agreement between piecewise-constant tet values; it does not impose exact continuity. If strict continuity is useful, compare against bounded nodal scalar activation interpolated with continuous linear basis functions, with separate nodes at compartment boundaries.

**Matched comparison matrix.** A direct comparison of raw six-DoF controls with the fiber model changes positivity, volume freedom, magnitude bounds, and direction simultaneously. Use two additional general-tensor definitions to interpret that comparison:

- `G6`: `A_inv = exp(H)`, with symmetric `H` and `||H||_F <= sqrt(3/2) a_max`. This retains six DoF with a positive definite, bounded activation map.
- `G5`: the same model with `trace(H) = 0`, giving five DoF and unit active determinant. The fiber model has `H = a(P - Q/2)` and is a subset of G5 under the same norm cap.

Run the following eight rows on the same constant-fiber block. The general-tensor magnitude and smoothness terms use `||H||_F^2/(3/2)` and `||H_e-H_j||_F^2/(3/2)` in place of `a^2` and `(a_e-a_j)^2`. They exactly agree on the fiber subspace when the fiber is constant.

| Row | Control space | Small magnitude | Spatial smoothness | Question |
| --- | --- | --- | --- | --- |
| G | G5 per tet | off | off | What remains after bounding the isochoric tensor? |
| G-M | G5 per tet | on | off | Does magnitude control suffice? |
| G-S | G5 per tet | off | on | Does spatial regularity suffice? |
| G-MS | G5 per tet | on | on | Can a general tensor be regularized adequately? |
| F | Fixed fiber, scalar per tet | off | off | Does the restricted fiber space suffice? |
| F-M | Fixed fiber, scalar per tet | on | off | What does magnitude add? |
| F-S | Fixed fiber, scalar per tet | off | on | What does spatial regularity add? |
| F-MS | Fixed fiber, scalar per tet | on | on | Does the combination justify lost fit? |

Add three reference rows: the historical raw unbounded six-DoF model; G6 without regularizers; and one bounded scalar shared over the entire muscle. The last is an intentionally restrictive endpoint. All rows in the new matrix use the same area-weighted target loss; do not compare a rerun against an old endpoint with different settings. Raw6 versus G6 measures the combined admissibility/bound intervention; unregularized G6 versus G5 isolates active-volume freedom within that bounded parameterization. The eight-row matrix tests the remaining fiber, magnitude, and smoothness effects under a shared isochoric assumption. Include G6-MS in the finalist strength sweep to check whether regularized volume freedom changes the practical recommendation. Differences in optimization difficulty must still be reported.

For the first screen use `lambda_m = lambda_s = 0.01` when a term is on; these are pilot settings, not conclusions. On G-MS and the promising fiber rows, trace the fit/roughness frontier with weights in `{0, 1e-3, 1e-2, 1e-1, 1}`. Start with one-axis sweeps and expand around nondominated points. A negative conclusion about a penalty requires this strength sweep, not failure at one weight. Freeze selected weights before testing new noise realizations or face targets.

**Experiment 1: verify the mechanism without inverse optimization.** Reuse the deterministic 3-D layered block from the fat-transfer study: footprint `1 x 1`, bottom fat thickness `0.04`, muscle thickness `0.02`, and top fat thickness `0.04`; bottom fixed and sides/top free; no skin or external pressure. Keep the existing fat/muscle moduli `0.003/0.03 MPa` and `nu = 0.49`. Use `f = (1,0,0)` throughout the muscle.

At a common mean log contraction `a0 = 0.15`, prescribe `a = a0 + b p_k(x,z)`, with `p_k` a sampled cosine-product field normalized to zero volume-weighted mean and unit RMS. Compare `k = 1` and `k = 4` at `b = 0.03`; both have the same mean and RMS activation. Include uniform `a0` and repeat the two patterns at `b = 0.015`. Verify all fields remain within the bounds without clipping them. Measure surface and muscle-interface responses relative to the uniform solve. This tests whether changing activation frequency changes surface roughness at comparable activation magnitude, independently of inverse fitting.

Use the existing `48 x 48` lateral grid with `0.01` vertical spacing for these probes; it resolves the prescribed `k = 4` pattern. Keep the source and surface response metrics separate, including modal transfer where defined. A failure to observe the expected effect is useful and must not be encoded as a failed numerical assertion.

**Experiment 2: invert controlled targets.** Use the same block and three target conditions:

| Target | Construction | What it tests |
| --- | --- | --- |
| Clean, attainable | Generate a strict forward equilibrium with `a_true = 0.10 + 0.10 exp(-((x-.5)^2+(z-.5)^2)/(2*.2^2))` and the fixed fiber model. Observe only the top. | Can the restriction retain intended motion when its assumptions are correct? |
| Noisy | Add scalar noise along the reference surface normal, using random combinations of lateral Fourier modes with wave numbers 4–6 and area-weighted RMS `0.02 D`. Retain the clean target privately for evaluation. | Does regularization prevent fitting target noise? |
| Demanding smooth mismatch | Add `delta * 16 x(1-x)z(1-z)` in the upward direction to the clean target, with the added field normalized to RMS `0.5 D`. | Does the method trade residual error for admissible smooth activation when asked for extra motion? |

Coordinates in these formulas are normalized by `L`. The mismatch is deliberately demanding; do not call it mathematically unreachable without a certificate. Its smooth shape avoids building artificial bumps into the target. On finalists increase the mismatch to RMS `D` and test a material mismatch separately; do not vary several sources of mismatch at once.

Start the inverse screen on a `24 x 24` lateral grid with `0.01` vertical spacing. Generate the clean target on that same grid for an implementation/identifiability control. The 11 rows and three targets make **33 initial inverse fits** at one common initialization. Compare activation tensors to truth for general models and scalar activation to truth for fiber models, as well as unobserved interior displacement. A nonunique surface-to-activation inverse need not recover the exact generating field, even with noiseless data.

For each finalist at a frozen weight setting, cross five independent noise fields, two RMS levels (`0.02 D` and `0.05 D`), and three optimization initializations: 30 noisy fits per method. The clean and demanding targets each get the same three initializations, adding six fits per method. Reuse qualifying pilot cases rather than duplicating them. Generate the spatial noise in reference coordinates so that it is the same physical field across meshes. Run a separate target from a refined mesh and interpolate observations to the inversion mesh; the same-mesh clean test alone is not external validation. Refine both lateral and vertical spacing for the numerical check. Keep activation correlation lengths and evaluation wavelengths fixed in physical units. Also solve on the refined physical mesh with the same coarse continuous activation basis, to separate forward discretization from increased control capacity.

**Experiment 3: test the fiber assumption.** Keep this separate from the first matrix so that an unknown fiber field cannot explain away every result.

- In the straight block, perturb the known direction coherently by 10 and 25 degrees. Add spatially varying orientation error at 10 degrees RMS with a fixed correlation length, to test whether a noisy fiber field creates tensor roughness even with smooth scalar activation.
- For a muscle with identified origin/insertion attachments, compare a single attachment-to-attachment direction with a Laplace-derived field. Solve `Delta phi = 0` with balanced inflow/outflow flux on attachments, no flux on the remaining boundary, and a fixed potential gauge; normalize `grad(phi)`. This follows the attachment-based construction in [Choi and Blemker, 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0077576). Geometry alone does not identify the correct attachment patches or guarantee anatomical truth.
- For a ring toy and later Orbicularis oris, use tangents to an explicitly defined reference centerline as the initial hypothesis. A single straight principal axis is a poor ring model. Circumferential fiber contraction can produce inward radial surface motion; fiber direction and displacement direction are different quantities.

The current face preparation does not supply a verified fiber field. Inspect any historical orientation assets for mesh identity and coordinate registration before reuse. Record attachment/centerline annotations, field construction, and uncertain regions. Zero or unresolved gradients require repairing the construction; do not silently substitute a world axis. Freeze the field for the primary inverse comparison. Jointly fitting a direction in every tetrahedron would restore much of the freedom being removed.

On curved fibers, smooth `a`, not raw global tensor entries. Constant scalar activation legitimately produces changing tensors as fibers turn. Report the roughness of `P = ff^T` separately; do not penalize anatomical curvature as activation noise. If uncertainty makes the fixed-field model too restrictive, test a small number of shared directional corrections or a bounded residual tensor only as a later ablation.

**Experiment 4: transfer only the supported methods to the face.** Reuse one fixed prepared face mesh, expression target, muscle mask, attachment conditions, passive material setup, and skin/prestrain choice. Begin with the saved no-skin setting implicated in the upper-mouth artifact and evaluate both the whole observed face and the fixed upper-lip/Orbicularis-oris region. Fit the whole model; a cropped ROI used for metrics must not introduce new cut-boundary mechanics.

Compare raw6, G6, G-MS, G6-MS, F, and F-MS; add F-S if it was competitive in the toy study. Treat the general tensor smoothness result on curved anatomy as a separate comparison because its global tensor penalty is no longer identical to scalar smoothing. Test `gamma = 0` versus `1/2` after fixing the other settings. Then repeat the finalists with the project's chosen skin/prestrain setup, as a separate block of cases. Freeze each setup before comparing activations; do not transfer a fitted activation blindly between materials or geometry.

Use a second expression as a held-out robustness test if a compatible target is available. The decision should be phrased as improved regularity and artifact reduction under an anatomical prior. Shape fit alone cannot validate physiological activation: [Eskes et al., 2017](https://www.nature.com/articles/s41598-017-17790-4) found that position tracking could be good while activation estimates were poor. EMG or independently measured internal motion would be additional evidence, not a prerequisite for the initial computational study.

**Measurements and decision rule.** Do not optimize a surface-smoothing loss in the primary study; use surface roughness only for evaluation so the proposed activation mechanism is actually tested.

| Dimension | Report |
| --- | --- |
| Fit and retained motion | Area-weighted target RMS, p95/max error, clean-target error for noisy cases, low-frequency target amplitude/correlation, unobserved interior error in synthetic cases. |
| Surface artifacts | Area-weighted high-pass normal-displacement RMS, curvature/Laplacian diagnostics, and spatial maps. Report high-pass vector displacement as a secondary diagnostic. |
| Activation size | Muscle-volume-weighted RMS and p95/max log strain for positive definite maps; singular-value condition number of `A_inv`, active determinant, and fraction at the bound. Report prescribed fiber shortening for fiber rows. |
| Activation continuity | Scalar graph energy and jump p95/max; tensor graph energy where appropriate; frequency spectrum on the block. |
| Mechanical validity | Minima and volume-weighted fractions for `det(F) <= 0`, `det(A_inv) <= 0`, and `det(G) <= 0`, throughout the accepted trajectory and at the final state; self-contact/interpenetration where relevant. |
| Numerics and robustness | Forward residual, inverse projected-gradient/KKT residual, adjoint/gradient checks, failures, initialization sensitivity, mesh sensitivity, solve counts, and wall time. |

Define the high-pass operator once in reference geometry with a fixed physical smoothing length; start with the existing block setting `0.06 L` and show `0.03 L` and `0.12 L` sensitivity. Do not redefine the cutoff in numbers of vertices as the mesh changes. Evaluate both the whole top and a predeclared interior crop. On synthetic cases the principal excess-bump metric is `RMS_area(HP[(u - u_clean) dot n0])`; also show the raw high-pass surface displacement, since the intended clean shape can contain real high-frequency features. On faces report the target's own roughness and distinguish lips/creases from an artifact ROI.

For a raw6 map that is not positive definite, mark the symmetric log strain undefined; report `||A_inv-I||_F`, singular values, and determinant instead. A singular map has infinite condition number. Do not compute a complex matrix logarithm or silently repair the map for metrics.

Plot roughness against fit error for every method/weight, with invalid and nonstationary cases marked. Compare at overlapping fit-error levels; never extrapolate a frontier to manufacture a matched-fit point. If there is no overlap, state the extra error paid for the improvement. Include zero activation and one-shared-scalar endpoints as visual references. The zero-activation equilibrium is one additional non-optimized forward baseline per fixed physical setup, reused across targets and excluded from the 33 inverse-fit count.

Define retained motion relative to that equilibrium `u0`: `q = LP(u_clean-u0)` and `r = LP(u-u0)`, where `LP = I-HP` uses the same fixed physical filter. Report the area-weighted projection amplitude `<r,q>/<q,q>`, its residual, the mean displacement, and the corresponding amplitude/residual after removing the spatial mean. This prevents retention of only mean motion from hiding loss of the intended spatial pattern. For the mismatch case use the prescribed noiseless target in place of `u_clean`; for noisy cases retain the clean reference. If a reference component has zero norm its amplitude is undefined, not zero.

A proposed practical screening criterion is at least **30% less excess surface roughness while retaining at least 80% of both the total and spatially varying low-frequency projection amplitudes**, with no inverted final tetrahedra. Inspect projection residuals as well; a large amplitude with the wrong pattern is not retained motion. These percentages are provisional engineering preferences, not scientific constants. Show the full frontier so another fit tolerance can be chosen. A method can still be informative if it misses this criterion. For the clean target, reject a setting that destroys intended deformation; for noisy targets, prefer better clean-target/generalization error even when noisy training fit worsens.

Use the ablations to distinguish outcomes: F improves over G if the fiber restriction helps; G-S improves over G if spatial roughness is sufficient to explain part of the effect; G-M improves over G if excessive magnitude is important; F-MS improves over its single-penalty variants if they complement one another. If activations become regular but the surface remains bumpy or folded, investigate mechanics, contact, target/model mismatch, or branch selection instead of increasing penalties indefinitely. Even a positive definite activation tensor does not guarantee positive `det(F)`.

**Execution and reporting contract.** Check the activation adapter first: identity at zero, the intended fiber shortening, unit determinant for the isochoric model, correct six-component packing, and derivative agreement on a small stable fixture. Use a constrained optimizer or projection with the appropriate projected-gradient stationarity measure; do not confuse a small gradient caused by sigmoid saturation with convergence. For G5 use an orthonormal trace-free tensor basis and a norm-ball constraint, so optimization coordinates have a defined scale.

Keep all methods on the same declared equilibrium-branch policy. In the toy study, audit gradients by finite differences with local continuation, because previous block experiments show that reseeding forward solves can switch branches even when both solves converge. Report branch disagreement separately from forward failure. Use comparable normalized objectives and effort budgets, then extend representative finalists until their local stationarity is established or explicitly remains budget-limited. Equal Adam step counts alone do not establish a fair converged comparison.

Do not silently insert a determinant barrier, contact term, skin energy, post-hoc activation filter, or mesh repair into only one method. Initially record mechanical invalidity under the same forward law; invalid outputs cannot establish successful mitigation. If the candidate priors still require a determinant safeguard, make that a separately declared, shared intervention. Preserve failed cases and rejected-trial diagnostics rather than replacing their outputs with a plausible default.

Keep implementation local to this experiment group before promoting a reusable module. Proposed future entrypoints, **not yet implemented**, are `src/10-forward-frequency-probe.py`, `src/20-inverse-constraint-matrix.py`, `src/30-fiber-sensitivity.py`, `src/40-face-validation.py`, and `src/50-analyze-results.py`. Follow Cherries configuration/output conventions and record the run name, tags, complete parameters, input hashes, source revision, seeds, dependency versions, observation masks, muscle graph, fiber field, and convergence receipt. Save accepted control/deformation checkpoints for replay. Store scalar metrics and case manifests independently of videos; render selected trajectories and separate reusable PNG/PVSM assets with fixed cameras and scales.

The first deliverable should be the forward mechanism check and the 33-fit toy screen, followed by the strength frontier for finalists. Proceed to fiber uncertainty and the face only when those results distinguish regularization effects from trivial loss of motion or solver failure. No simulation runtime is estimated until a pilot establishes actual forward/adjoint costs.

**Local source pointers.** These were inspected for this design; the links identify reusable code and saved evidence, not new validation results.

- [Activation matrix packing](../../../../../../src/liblaf/apple/warp/fem/func/_misc.py:48) and [active constitutive energy](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py:20).
- [Current face parameters](../../../../06/17/human-face-skin-prestrain/src/_human_face_runtime.py:145), [surface-only loss](../../../../06/17/human-face-skin-prestrain/src/_human_face_loop.py:214), and [mesh activation fields](../../../../06/17/human-face-skin-prestrain/src/_human_face_mesh.py:117).
- [Existing isochoric block runner](../../../../08/19/fat-thickness-bumpy-activation-transfer/src/10-run-bumpy-activation-transfer.py:299) and [saved transfer report](../../../../08/19/fat-thickness-bumpy-activation-transfer/docs/10-bumpy-activation-transfer.md).
- [Saved face folding evidence](../../../02/human-face-upper-mouth-muscle-folding/docs/10-upper-mouth-muscle-folding.md) and [block convergence/branch-selection limitations](../../../../08/31/unreachable-pork-factor-study/docs/10-unreachable-pork-factor-study.md).
- [Historical magnitude and same-muscle smoothness precedent](../../../../../2025/10/22/inverse-flame/src/20-inverse-adam.py:149). Reuse the objective idea, not the older JAX/PHACE plumbing.
