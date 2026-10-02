# Learned-axis contraction and spatial smoothness

**Proposed plan for review. Execution remains paused.** This round schedules three new fits: learned-axis smoothing off/on with one initialization, and corrected Raw6 with smoothing. The completed unsmoothed Raw6 run supplies its paired control; existing PSD results are reused. Seed sensitivity is deferred. The original calibration is complete, but the revised validation, calibration, and primary fits have not started.

## 1. Questions and scope

The primary question is whether spatial regularization of a learned uniaxial contraction field reduces uneven facial deformation while preserving fitting quality and intended expression motion. The corrected Raw6 comparison tests whether smoothing also helps the more flexible baseline. Sensitivity to the starting axis field will be examined in a later study.

The new experiment provides a learned-axis off/on pair at one initialization and a corrected Raw6 off/on pair using the recorded Raw6 starting state. The latter reuses the completed unsmoothed arm. Existing tensile PSD results supply a previous controlled pair and longer-run context. Cross-model comparisons describe attained fit, motion, and roughness under the recorded protocols; they do not establish an equal-budget ranking or isolate the effect of the admissible activation space.

Use the existing smile target and anatomical mesh. Skin membrane energy, contact, magnitude penalties, and rank penalties are zero. The activation increment C has rank at most one by construction; B remains full rank. No additional contraction or stress cap is imposed. The volume-preserving active-strain alternative and additional parameterizations are outside this plan.

## 2. Activation model

For each active tetrahedron e, optimize three unrestricted real coordinates vₑ:

$$
C_e=v_ev_e^\top,\qquad B_e=A_e^{-1}=I+C_e,
\qquad s_e=\|v_e\|^2.
$$

The active map, learned axis, and shortening fraction are

$$
A_e=I-\frac{v_ev_e^\top}{1+s_e},\qquad
f_e=\frac{v_e}{\|v_e\|}\quad(v_e\ne0),\qquad
c_e=\frac{s_e}{1+s_e}.
$$

The active stretches are (1/(1+s), 1, 1): contraction along one learned axis and unchanged transverse active lengths. No sign constraint on v is needed; v and −v represent the same activation. This constrains prescribed activation, while mechanically coupled tissue can still stretch. The learned direction is an effective actuation axis, not a validated anatomical fiber direction.

Use the corrected physical-volume muscle energy:

$$
W(F;B)=\frac{\mu}{2}(\|FB\|_F^2-3)-\mu(J-1)
+\frac{\lambda_L}{2}(J-1)^2,\qquad J=\det F.
$$

Thus the volumetric terms use physical det(F). Retain the verified material fractions, fixation, and classical-Lamé convention:

| Phase | Young's modulus (MPa) | Poisson ratio |
| --- | ---: | ---: |
| Fat | 0.003 | 0.49 |
| Muscle | 0.03 | 0.49 |
| Aponeurosis | 0.1 | 0.35 |

The fixture contains 228,660 vertices, 1,146,517 tetrahedra, 288,235 active tetrahedra, and 15,302 finite `IsFace` observations. The new model has 864,705 scalar controls. The skin surface remains available for observation and rendering, with its membrane energy disabled.

## 3. How spatial regularization is applied

The fitting loss is the existing uniform displacement-coordinate MSE:

$$
L_{\mathrm{fit}}=\frac{10^6}{3N}\sum_{p\in\mathrm{IsFace}}
\|u_p-u_p^*\|^2.
$$

Displacements are in meters. L_fit is in mm² and equals one third of squared vertex-vector fit RMS in millimeters.

Regularize **C = vvᵀ = B − I** during optimization:

$$
L=L_{\mathrm{fit}}+\lambda_s S_C,\qquad
S_C=\frac{\ell^2}{V}\sum_{(i,j)\in E}
w_{ij}\|C_i-C_j\|_F^2.
$$

Use the existing 501,409 unique same-muscle shared-face edges. Each weight is shared-face area divided by centroid distance, multiplied by the harmonic mean of the two muscle fractions. V is total rest tetrahedron volume weighted by muscle fraction, and ℓ = 5 mm. No edge crosses muscle labels. S_C is dimensionless and λ_s has units mm². The length ℓ normalizes the penalty; it is not a prescribed surface smoothing radius.

Writing Cᵢ = sᵢ fᵢfᵢᵀ makes the treatment explicit:

$$
\|C_i-C_j\|_F^2=s_i^2+s_j^2
-2s_is_j(f_i^\top f_j)^2.
$$

The penalty discourages abrupt changes in both contraction strength and axis direction. Opposite vectors describe the same axis and are not penalized as different. Orientation matters less when contraction strength is small. A constant tensor on a connected muscle component has zero penalty, so spatial smoothing does not bound overall activation magnitude.

Autograd adds λ_s ∇ᵥS_C to the fitting gradient before each Adam update. The fitting gradient comes through the equilibrium adjoint; the graph penalty differentiates directly through C = vvᵀ and needs no second mechanics adjoint. For a cell i, its penalty-gradient contribution is

$$
\nabla_{v_i}S_C=\frac{4\ell^2}{V}
\sum_{j\in\mathcal N(i)}w_{ij}(C_i-C_j)v_i.
$$

Here N(i) contains all same-muscle neighbors incident to cell i; each graph edge contributes to both cells' gradients.

Keep the effective stress field distinct:

$$
Z=BB^\top-I=(2+s)C,\qquad Q_{\mathrm{eff}}=\mu Z,
\qquad \mu=0.010067114093959731\ \mathrm{MPa}.
$$

S(Z) remains a common physical-actuation diagnostic across models. It is not the learned-axis penalty. No constant coefficient conversion makes S(C) and S(Z) equivalent when strength varies spatially. The earlier shared-Z coefficient must therefore not be copied into λ_s.

For the new Raw6 smoothing arm, use the same graph functional on its unrestricted symmetric C = B − I, with a separately calibrated coefficient λ_b. Reconstruct its physical field using Z = 2C + C²; the simplification Z = (2+s)C applies only to the learned-axis rank-one case. Raw6 has a representation ambiguity: different signs of B can preserve BBᵀ while changing S(C). The recorded starting state already has 2,573 cells with a nonpositive B eigenvalue. Treat this as a within-Raw6 control-space intervention, retain B-sign and S(Z) diagnostics, and do not interpret the two models' coefficients as directly comparable physical strengths.

## 4. Existing evidence to reuse

Recompute needed diagnostics from saved states without repeating their inverse optimization. A reuse manifest will record source paths, fixture hashes, checkpoint steps, material law, optimizer history, penalty definition, and original verification receipts.

| Source | Reuse | Limits |
| --- | --- | --- |
| [Corrected no-skin Raw6 re-fit](../../../08/local-skin-prestrain/data/30-refit-no-skin/) | Reused unsmoothed control for the new Raw6 smoothing arm; final uniform fit RMS 1.397767 mm. | Begins from trained canonical step-200 controls, followed by 200 updates with fresh Adam at lr 0.3. New Raw6 smoothing must reproduce that starting state and protocol. |
| [PSD off](../../../07/tensor-active-stress/data/21-psd/) and [PSD smooth](../../../07/tensor-active-stress/data/22-psd-smooth/) | Completed equal-budget controlled 64-update pair: same start/settings except smoothing. | Both start at zero, use fresh Adam at lr 0.3 and eps 0.01, and have their own penalty/weight. Their endpoints are not assumed to match in fit/motion. |
| [PSD step 1024](../../../07/tensor-active-stress/data/102-fit1024/) and its recorded parents | Longer-run fit/roughness context; final uniform fit RMS 1.542084 mm. | Inherited Adam history, lr 0.3 then 0.6; no matched smoothed continuation. |
| [Completed calibration](../data/15-calibration/summary.json), [mechanics probe](../data/15-initial-probe/summary.json), and original validation | Physical step scales, solver timing, unchanged-component checks, and failed-rate evidence. | These do not supply a learned-axis learning rate or C-smoothness weight. |
| Fixture, graph construction, ROIs, normals, filter, cameras, and section planes | Reuse with unchanged definitions and provenance. | Model-dependent quantities are recomputed from the saved states. |

The old PSD fixture hashes, passive constants, no-skin setting, graph construction, and tensor constitutive implementation match this study. Saved q contains six Frobenius-orthonormal symmetric coordinates per cell. With M(q) denoting their symmetric matrix unpacking, Q = Q_ref M(q), Q_ref = 3μ, and Z = 3M(q) = Q/μ. Consequently S(M(q)) = S(Z)/9; the old coordinate penalty uses exactly that Frobenius norm. Its smoothing weight 5.9051714684188505 is equivalent to λ_Z = 0.65613016315765, an existing physical-weight point rather than a matching λ_s for S(C).

The old PSD upper cap was 0.3020134228 MPa. Upper clipping is zero throughout the audited trajectory, whose maximum evaluated eigenvalue is 0.2868469517 MPa. Describe this as a recorded nonbinding upper cap; lower PSD projection remained active.

Map saved Raw6 controls to Z = BBᵀ − I and PSD controls to Z = Q/μ. Use full u and control checkpoints for complete measurements: surface-only files omit three fitted vertices. Keep uniform and area-weighted fit measures distinct; the main global fit measure is uniform vertex-vector RMS.

Older Raw6 trajectories whose volumetric terms use det(FB), skin-enabled runs, rank-penalized PSD runs, and the four-update Ainv-smoothing continuation do not replace primary cases. The recorded no-skin re-fit is a paired control only for the new Raw6 arm that copies its initial state and settings. It remains a mixed-protocol reference in comparisons with learned-axis or PSD results.

## 5. Three new runs and initialization

| Case | Initial state | Smoothness | Updates | Output |
| --- | --- | --- | ---: | --- |
| Axis-off | Seed 20260909 | λ_s = 0 | 256 target | `data/24-learned-axis/` |
| Axis-on | Same seed 20260909 | Selected λ_s > 0 | 256 target | `data/25-learned-axis-smooth/` |
| Raw6-on | Recorded canonical controls and forward seed | Selected λ_b > 0 | 200 | `data/28-raw6-smooth/` |

Raw6-off is the already completed 200-update no-skin re-fit. No second learned-axis seed is scheduled.

For the learned-axis pair, set s₀ = 0.001, giving approximately 0.10% prescribed shortening. With seed 20260909, sample one isotropic random unit vector per muscle label and initialize v₀ = √s₀ f. Each cell subsequently optimizes independently. The off/on pair has identical controls, zero displacement solve seeds, and fresh Adam state. A zero vector cannot be used because the derivative of vvᵀ vanishes there.

Freeze one selected learning rate for both learned-axis runs and one positive λ_s for Axis-on. Across all new runs, use betas = (0.9, 0.999), eps = 0.01, weight_decay = 0, amsgrad = False, and maximize = False. Use float64 on the same GPU and record Torch versions and actual foreach/fused options. Pilot states are discarded. Historical fitted tensors and Adam moments do not initialize the learned-axis pair.

For Raw6-on, copy the exact canonical step-200 q used by the recorded off arm and use [the saved no-skin forward state](../../../08/local-skin-prestrain/data/20-forward-no-skin/final.npz)'s u as the solve seed. Create fresh Adam at lr = 0.3 for 200 updates; do not inherit the canonical moments or start from the off arm's final re-fit state. The forward file has SHA-256 `9fb5c34a361328c4b90ef32983dc2f77a6fae3d1ac813ee08016142e7fadab13`. Record file/array hashes, initial-control identity, and the distinct semantics of copying controls versus resuming an optimizer.

Obtain the learned-axis pair through 128 updates and complete the Raw6 smoothing control before extending the learned-axis pair to 256. Report equal update counts within each model's off/on pair. This round cannot establish initialization robustness; the unused seed is reserved for a later sensitivity study, not an automatic follow-up run.

## 6. Validation and bounded calibration

### Preparation and checks

The prepared runner currently penalizes Z. Before fitting, implement S_C for each new model with its declared control map, retain S(Z) as a diagnostic, bind actual Adam settings to frozen receipts, and preserve complete traces and best states across continuation segments. Add an archived-state initialization mode for Raw6 that copies q/u but creates fresh moments; the existing resume path restores moments and is unsuitable for this purpose. These changes are pending approval and implementation.

Complete CPU checks for raw6 packing, B/C/Z identities, inverse stretches, sign invariance, rank, and finite-difference gradients of the maps and S_C. Verify force/Hessian equivalence at mapped controls, graph normalization, and the penalty gradient above. Reuse completed checks for unchanged mechanics.

At the proposed learned-axis nonzero start, run a small face comparison against equivalent Raw6 controls and check the displacement-loss chain rule through v. Axis-off/on step-0 C, Z, equilibrium, and fit must agree. Their first optimizer update must also agree because the per-muscle constant C₀ gives S_C = 0 and ∇S_C = 0.

For the Raw6 pair, verify exact starting controls and seed provenance, then check the re-equilibrated no-update state against the archived off-state fit and mechanics before tuning. An evaluated u need not equal the input solver seed bitwise. The first smoothed update is expected to differ because the trained Raw6 starting field has nonzero variation.

### Learned-axis Adam learning rate

Use eight-update pilots on seed 20260909. The previous physical ΔZ RMS values 0.01881737, 0.05669795, and 0.19279357 define three candidate first-step scales. They are ΔZ targets, not ΔC targets, and do not imply equal later optimizer geometry. Raw6 keeps its recorded lr = 0.3 so its completed off arm remains a controlled comparison.

For the initial learned-axis gradient g, form the first Adam direction d = −g/(|g| + 0.01). Find the rate giving each target physical ΔZ on the first increasing branch. Record actual volume-weighted RMS and maximum physical changes.

Test all three rates. Eligibility requires successful finite forward/adjoint solves, positive fitting progress, and final fitting loss within 1% of the best post-update loss. Score a rate by fractional loss reduction (L₀ − L₈)/L₀, with L₀ > 0; choose the smallest rate within 5% of the best score. Do not expand above the largest physical target; if that candidate wins, report that the best tested rate lies at the search boundary. If every candidate fails, allow one predeclared half-of-smallest-rate pilot, then stop calibration if it also fails.

### Learned-axis C-smoothness weight

Reuse the selected fit-only pilot endpoint. Compute the separate gradients with respect to v and set

$$
\lambda_0=0.25
\frac{\|\nabla_v L_{\mathrm{fit}}\|_2}
{\|\nabla_v S_C\|_2}.
$$

Require finite gradients and a nonzero denominator. This provides a search scale in the fixed v coordinates, not an invariant physical calibration. Test λ₀/4, λ₀, and 4λ₀ with eight fresh updates at the selected learning rate and seed 20260909.

Each candidate must have successful solves, an improving total objective ending within 1% of its best post-update value, at least 80% of its fit-only pilot's fitting progress, at least 90% of its incremental target-direction projection progress, and reduced S_C. These are operational selection rules, not physiological limits.

Fitting retention is (L₀ − L_on,8)/(L₀ − L_off,8), and projection retention is (p_on,8 − p₀)/(p_off,8 − p₀). Here p is the global unweighted `target_projection` metric, not the later low-frequency regional projection. The shared step-0 state supplies L₀ and p₀. Require positive reference progress in both denominators; otherwise that candidate is ineligible. Relative roughness reduction is 1 − S_C,on,8/S_C,off,8, requiring S_C,off,8 > 0.

Score eligible weights by relative S_C reduction. Choose the smallest weight within five percentage points of the best score. If no weight passes, report the failed calibration instead of substituting zero or an unverified value. Any additional search requires an explicit documented revision to this bounded plan.

The learned-axis grid uses 55 solve/adjoint evaluations: one initial state plus six nine-state pilots, with the selected fit-only endpoint reused. The optional lower-rate pilot raises this to 64; mapped-mechanics validation solves are additional. At the earlier measured mean of about 17.45 seconds/state, this is roughly 16–19 minutes before setup and validation. Use actual learned-axis timings to update the estimate. This setting is tuned and evaluated on one initial field; no seed-robustness conclusion is intended.

### Corrected Raw6 C-smoothness weight

Keep lr = 0.3, eps = 0.01, and the recorded starting q/u. Use the existing off trajectory's first ten updates as the unregularized reference, including its saved step-10 full controls. At the matched starting state, compute separate q-coordinate gradients of L_fit and S_C and set λ_b,0 = 0.25 ‖∇q L_fit‖/‖∇q S_C‖. Test λ_b,0/4, λ_b,0, and 4λ_b,0 for ten fresh updates from that same state.

Apply the same solver, total-objective, 80% fitting-progress, 90% incremental projection-progress, and positive S_C-reduction rules, replacing step 8 by step 10 and comparing against the recorded Raw6-off reference. Choose the smallest eligible weight within five percentage points of the largest S_C reduction. Check reference denominators and numerical starting-state agreement before using these ratios. Do not copy the learned-axis coefficient or infer equivalent physical prior strengths from their numerical values.

This adds approximately 34 solve/adjoint evaluations, including the initial gradient check, without rerunning the completed off branch. Allow about one hour for both models' preparation/calibration together, then refine the estimate from observed costs.

Freeze the learned-axis rate, λ_s, seed, and selection receipts in `data/18-learned-axis-calibration/summary.json`; save the independent baseline coefficient and archive-initialization receipts in `data/19-raw6-smooth-calibration/summary.json`. Freeze both before their primary runs, retain failed pilots, and do not retune a primary case after seeing its outcome.

## 7. Execution and time allocation

The original 12-hour window is September 9, 2026, 01:21–13:21 Asia/Shanghai. At the 03:30 revision approximately 9 h 51 min remained; review time reduces that window. The deadline is unchanged.

| Priority | Allocation |
| --- | --- |
| Validate prepared code and calibrate both models' settings | Target one hour after approval; no second-seed pilots. |
| Complete the learned-axis pair through 128 updates and Raw6-on through 200 | First fitting priorities; the Raw6-off trajectory is already complete. |
| Extend the learned-axis pair through 256 updates | Main learned-axis target: 512 total updates, about 2 h 51 min at an assumed 20 seconds/update. The previous Raw6 200-update off run took about 31 minutes; budget 45–60 minutes for its smoothed counterpart pending pilot timing. |
| Diagnose an actual failure or unresolved discrepancy | Use remaining simulation time for a concrete issue while preserving failed/partial evidence. |
| Independent verification, rendering, and concise report | Reserve the final 90 minutes, beginning no later than 11:51. |

Use one GPU fitting process at a time. Continue a started run from exact controls, Adam moments/counter, and solver seeds; distinguish this from the fresh-moment Raw6 initialization above. After each segment, estimate remaining time from observed costs; begin the next complete learned-axis round only if it fits before the analysis reserve. Report the largest completed common update count within each model's off/on pair and longer partial trajectories separately. No automatic extension beyond 256 learned-axis updates, 200 Raw6 updates, or an additional seed/PSD study is scheduled.

Forward settings remain maximum 5,000 iterations, rtol = 5×10⁻⁴, atol = 10⁻¹⁰, and line-search limit 10. Retain the existing adjoint solvers and rtol = 5×10⁻⁴. A failed solve or nonfinite field stops its trajectory with a failure receipt. Do not relax tolerances or silently lower that case's rate to label it completed. Other valid cases may continue within the budget.

Record inversions, det(F), shortening, and stress spectra as diagnostics rather than extra rejection gates. Accepted equilibria meet the declared numerical criteria; they are not certified physiological states. Fixed update counts and the deadline do not establish inverse stationarity.

## 8. Measurements and interpretation

Use the existing frozen mouth-corner, lateral-cheek, lower-cheek/jaw, and nasolabial supports, rest normals, and 5 mm high-pass operator. Keep these definitions fixed before fitting.

| Measurement | Purpose |
| --- | --- |
| Uniform vector fit RMS, global and regional | Target agreement. |
| Face motion RMS, target-direction projection, and low-frequency/mean motion retention | Detect weakening or loss of the intended expression. |
| Area-weighted high-pass RMS of normal target residual | Primary measure of small-scale surface error. |
| Companion high-pass RMS of normal displacement | Explain how small-scale motion changes. |
| S_C, S(Z), volume-weighted stress magnitude, and shortening percentiles | Separate the optimized prior from physical actuation and activation strength. |
| Nasolabial fit and exact triangle-plane sections | Inspect groove retention; vector fit is not fold depth. |
| det(F), inversion counts, solve cost, and solver receipts | Mechanical and computational diagnostics. |

The primary surface score is area-weighted 5 mm normal-residual high-pass RMS over the union U of the three frozen cheek/mouth ROIs, counting each vertex once. Report each ROI separately as well, together with the protected nasolabial region. Add and validate this aggregate before fitting; it must use the same frozen operator and supports as the existing regional metrics.

For the low-frequency retention criterion, define one scalar on that same union. Let d = n₀·u and d* = n₀·u* be the rest-normal displacement and target fields. Apply the frozen low-pass LP on the full skin before restricting to U: LP(d) = (M + tK)⁻¹Md, with rest lumped-area mass M, the existing stiffness K, and t = (0.005 m)²/4. Then use

$$
P_U=\frac{\sum_{p\in U}M_{pp}\,\mathrm{LP}(d)_p\,\mathrm{LP}(d^*)_p}
{\sum_{p\in U}M_{pp}\,\mathrm{LP}(d^*)_p^2}.
$$

Validate the target denominator before fitting. This fixes the retention scalar rather than choosing a favorable region or averaging regional ratios afterward.

For the learned-axis pair and the corrected Raw6 pair separately, report off/on changes at their common final update count and through the trajectory. Separately select actual post-update states satisfying both |Δfit RMS| ≤ 0.05 mm and |Δmotion RMS| ≤ 0.05 mm. Exclude step zero. Among eligible pairs choose lowest mean fit error, then smallest combined fit/motion mismatch, then earliest steps. If none qualifies, report no matched comparison. Do not interpolate geometry or relax the thresholds.

Predeclare a useful matched surface effect as at least a 10% reduction in the aggregate normal-residual score, with lower S_C and P_U,on/P_U,off ≥ 0.90. Apply this ratio only when P_U,off > 0; otherwise report the values and withhold that preservation claim. A zero reference residual score leaves no relative improvement to claim. The 10% threshold is an operational effect-size criterion, not a statistical or physiological boundary. Report absolute changes too. Companion displacement roughness and nasolabial geometry explain the tradeoff rather than being hidden behind a single score.

Interpret the learned-axis outcome as evidence for one specified initial field. Report the Raw6 smoothing effect separately; it does not substitute for a second learned-axis seed. No eligible matched pair leaves that model's matched-state question unresolved, while its equal-update result is still reported. A lower S_C without a surface benefit is a negative result for the proposed surface-quality mechanism, even if the penalty works numerically. Local deterioration or loss of the target nasolabial shape limits broader claims.

The reused Raw6-off arm belongs in its controlled Raw6 comparison after starting-state verification. All cross-model tables/panels retain initialization, optimizer-history, and penalty labels; historical PSD trajectories remain contextual comparisons. Matching fit/motion does not equalize those protocols. No conclusion about seed robustness, anatomical fibers, other expressions, or converged model capacity follows from this round.

## 9. Evidence and deliverables

Save every evaluated surface and trace row; full u, model-specific controls (v or raw6 q), C, and Z states every 16 learned-axis updates or ten Raw6 updates and at phase boundaries; best fit, best total objective, and final states; and an atomic latest Adam checkpoint at every evaluated state. Record how many updates have already been applied. Preserve the complete earlier trace and global best states on continuation. Save failed controls separately from accepted geometry.

Archive fixture/source hashes, exact commands, versions, seeds, calibration decisions, solver receipts, and Cherries/Comet logs with readable run names and tags. Local artifacts remain the numerical evidence if remote logging fails. The reuse manifest binds historical states to their source records. No commit, push, or publication is included.

Independent verification checks C/Z reconstruction, learned-axis rank/contraction identities, penalty values, surface measurements, deformation diagnostics on full checkpoints, Adam update counts, fresh archive-state initialization versus continuation lineage, and source/input hashes. Raw6 is not subject to the learned-axis rank or contraction constraints.

Deliver a concise report in `docs/40-results.md`, a compact numeric table, and separately reusable PNG assets:

1. Learned-axis and corrected Raw6 off/on fitting and activation-variation curves, plus fit–surface-roughness plots with labeled PSD references.
2. Actual off/on surfaces and target at matched states where available, with equal-update views labeled separately. Use the frozen face, cheek, mouth, and nasolabial views with shared cameras, lighting, and scale.
3. Exact nasolabial sections for the learned-axis and Raw6 pairs.
4. A selected muscle-region view of unoriented learned axes colored by shortening, using identical cell sampling across panels, if useful for explaining direction changes.

Render actual saved geometry without geometric smoothing or deformation exaggeration. Each asset records case, initialization/seed, step, source checkpoint/hash, and camera. The report explains the observed effect, fit/motion tradeoff, reused evidence, failures, and limits, explicitly noting that seed sensitivity is deferred. Link the detailed protocol and verification receipt as supporting material.
