# Activation space and smoothness with no skin

**Superseded proposal.** The current plan for review is [Learned-axis contraction and spatial smoothness](12-learned-axis-plan.md). The six-fresh-case design below is retained as proposal history and is not scheduled for execution.

Status: proposed revision, paused for user review at 2026-09-09 02:34 Asia/Shanghai. The original two-model calibration is complete. Learned-axis control/validation code and the six-case runner have been drafted; extended validation, revised calibration, and all primary runs have not started. Execution awaits approval.

This study compares the physical-volume Raw6 baseline, additive positive-semidefinite tensile stress, and a learned contraction axis, each with and without spatial smoothness. The original working budget is 12 hours: 2026-09-09 01:21–13:21 Asia/Shanghai. The plan was expanded after reading the session “Design activation regularization,” including its final clarification that each tetrahedron's contraction direction can be optimized. The primary planned budget is 256 Adam updates per case; completion at that count does not establish stationarity.

| Case | Controls | Smoothness | Added skin membrane |
| --- | --- | --- | --- |
| baseline | historical Raw6 coordinates for symmetric B | off | off |
| baseline-smooth | identical Raw6 coordinates | on | off |
| tensile | direct orthonormal coordinates for PSD Z | off | off |
| tensile-smooth | identical direct PSD coordinates | on | off |
| learned-axis | three coordinates v, B = I + vvᵀ | off | off |
| learned-axis-smooth | identical learned-axis coordinates | on | off |

Magnitude and rank penalties are zero. No upper stress cap is applied. Negative eigenvalues are projected out of the direct tensile controls after each Adam update, outside automatic differentiation. The learned-axis model has rank at most one by construction; it does not need a rank penalty or PSD projection.

Every primary case starts from the same effective physical field Z₀ = B₀B₀ᵀ − I, using the canonical map B₀ = I + s₀ ffᵀ, where s₀ = 0.001 and f is one seeded isotropic random unit axis per muscle label. Raw6 and learned-axis use this map directly; tensile controls use its equivalent additive stress Q₀ = μZ₀. Seed 20260909 is fixed before primary runs. Axes are initially constant inside a muscle but subsequently learn independently per tetrahedron. Each model receives its exact coordinates for this shared field, a zero displacement solve seed, and fresh Adam moments. The small nonzero activation is necessary because v = 0 has zero derivative. Its prescribed axial shortening is s₀/(1+s₀), approximately 0.10%.

## Shared mechanics and prior

The baseline uses B = I + sym(q), with physical det(F) in the volumetric terms. Its effective stress field is Z = B Bᵀ − I. The tensile model uses Q = μ Z with Z positive semidefinite. At mapped controls these models have identical forces and deformation Hessians; their energies differ by the control-dependent constant tr(Q)/2. The learned-axis model uses the same physical-volume constitutive law as Raw6, with B = I + vvᵀ and Z = (2 + ‖v‖²)vvᵀ. Its active stretches are (1/(1 + ‖v‖²), 1, 1): one contracting direction and unchanged transverse active lengths. The volume-preserving variant discussed in the other session is not the selected model.

The effective feasible spaces are nested: Raw6 contains the tensile PSD space, which contains the learned-axis rank-one PSD space. These runs compare both a restriction in expressive capacity and different optimization coordinates. A better finite-budget result for a restricted model is not evidence that it can represent fields unavailable to Raw6. The learned direction is an effective actuation axis; surface fitting does not establish agreement with anatomical fibers.

The fitting loss is uniform finite-IsFace displacement-coordinate MSE multiplied by 10⁶. It equals one third of squared vector fit RMS in millimeters. The common dimensionless smoothness functional is

S(Z) = ℓ² / V × Σ₍ᵢ,ⱼ₎ wᵢⱼ ‖Zᵢ − Zⱼ‖²_F,

where ℓ = 5 mm, V is summed rest tetrahedron volume weighted by muscle fraction, and w is shared-face area divided by centroid distance, weighted by the harmonic mean of the two muscle fractions. Edges connect cells with the same muscle label only. The objective is fitting loss + λ S. The same positive λ is used in all three regularized cases. For learned axes, this avoids penalizing the arbitrary sign of v and preserves the common physical prior. The graph permits constant fields on each connected muscle component; smoothness does not bound activation magnitude.

The target, tetrahedral mesh, fixation, material fractions, passive material constants, physical-volume convention, and forward/adjoint tolerances are shared. The surface mesh remains available for observations and rendering while its membrane energy is disabled.

## Calibration before primary runs

The CPU control validation checks coordinate identities, PSD projection, derivatives, common smoothness, and constitutive equivalence. Extended validation checks learned-axis rank, contraction spectrum, derivatives, and the shared initialization. An initial face probe already checked the two original solver pipelines at inactive and matched nonzero controls. The six-case calibration checks all three pipelines at the new common start, including the learned-axis chain rule.

The completed original calibration used discarded pilots starting from inactivity and fresh physics/Adam state. It selected Raw6 lr = 0.9, tensile lr = 27.1350289781, and λ = 0.06856144437. Those artifacts are retained as preliminary evidence. The revised calibration checks the two rates again with eight updates from the new common physical start. Learned-axis candidates match the volume-weighted Frobenius RMS of the first physical ΔZ produced by Raw6 rates 0.3, 0.9, and 3.0, then select actual stable fitting progress; if the best remains at the upper edge, the physical equivalent of Raw6 9.0 is probed. The smallest rate within 5% of the best stable progress wins. A valid pilot must improve fitting and finish within 1% of its best post-update objective. The selected rate is shared by its smooth and unregularized cases; eps = 0.01 and betas = (0.9, 0.999) remain fixed. Initial stress-step matching does not imply identical later optimizer geometry.

The original weight grid was scaled from λ₀ = 0.25 ‖∇_Z L_fit‖ / ‖∇_Z S‖ at a discarded tensile endpoint. The revised calibration tests the selected shared weight in all three models. Selection requires at least 80% of unregularized fitting progress, 90% of unregularized target-direction motion, reduced activation variation, and a stable total objective in every model. If the weight fails, the previous weight divided by 4 and then 16 are checked in all three models; the largest passing weight is selected. Failure of every candidate stops calibration. Both selection protocols and every pilot trajectory are retained. The revised frozen settings are written to data/17-six-case-calibration/summary.json before primary endpoints exist.

## Remaining budget and priorities

At the 02:34 pause, approximately 10 h 47 min remained in the original window. The budget targets below are estimates based on the completed pilots, not guaranteed solver durations. Review time reduces the remaining window if the original 13:21 deadline is retained.

| Work after approval | Target allocation |
| --- | --- |
| Validate prepared code, common-start mechanics, and revised calibration | about 1 hour |
| Six primary cases: 128 updates each, then continue the whole cohort toward 256 | up to 7 hours total |
| Alternate learned-axis seed 20260910, smoothing off/on, 128 updates each | about 1 hour |
| Independent saved-state checks, figures, and concise report | final 90 minutes |

First obtain all six 128-update trajectories. Next obtain the alternate learned-axis pair at the same 128-update count. Use the remaining simulation allocation to extend all six primary cases to 256 with exact Adam-state continuation. Before using continuation, finish the parent-state verification and preserve the complete trace and best states across segments; the current prepared runner reports segment-local best states. This is implementation work to complete only after approval.

The six-case comparison and two-start sensitivity take priority over longer continuation, broader weight sweeps, or other parameterizations. Use the highest completed common update count for the primary table and report partial extra trajectories separately. The second seed is sensitivity evidence, not a claim of seed invariance. Reserve the final 90 minutes for independent saved-state checks, reusable PNG figures, and the concise report. Numerical failures remain visible and cannot be relabeled as completed cases. No convergence claim is based on the time limit.

## Evidence and interpretation

Save every evaluated outer surface, full states every 10 updates, and an atomic latest Adam checkpoint at every evaluated step. Each checkpoint records the number of updates already applied; all stored geometries come from evaluated equilibria. Select the best unpenalized fit and best total objective separately, and retain the equal-update final state.

Primary measurements are fitting error, residual and displacement surface bumpiness, target-direction and low-frequency motion retention, common stress magnitude/variation, and local nasolabial geometry. Surface roughness uses the existing frozen rest-normal 5 mm high-pass operator and frozen anatomical-region boxes. Nasolabial vector fit error is not a measurement of fold depth.

All forward and adjoint solves must pass their declared tolerances. Nonfinite fields or failed solves invalidate a trial. Inversions and stress spectra are recorded as diagnostics, without claiming physiological admissibility. Compare actual states at matched fit and motion where available; do not interpolate or smooth geometry to manufacture a match.

The final report will be concise, include reusable visual assets, and distinguish fixed-budget observations from convergence, physiological, or generalization claims. Source snapshots, input hashes, commands, Cherries/Comet receipts, calibration records, and saved-state checks provide the reproducibility evidence.
