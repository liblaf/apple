# Preregistered PSD optimizer-continuation protocol

## Scope and prerequisite

This experiment asks whether the fit-only PSD face inverse at global step 64 is limited by the current Adam scaling, while preserving the accepted constitutive model and feasible set. It does not test signed or otherwise unconstrained `Q`, and it cannot use Raw6 to claim model capacity.

Execution may start only after a separate, small known-reachable block recovery shows that the proposed smaller Adam epsilon reaches the known solution with finite gradients and successful solves. That study fixes `epsilon_reduced` before any face calibration. Its result, source hash, and acceptance receipt must be inputs to this run; face results cannot be used to choose epsilon.

### Small-recovery method amendment

The preregistered reduced-epsilon small recovery did not pass after 4,096 updates: its relative displacement error was `0.06443`. All controls remained strictly interior, while `99.876%` of the remaining displacement-residual energy lay in the weakest displacement-Jacobian mode. Thus epsilon reduction alone did not finish the known-reachable recovery, and the evidence supports a conditioning-limited final phase rather than clipping at the control bounds.

The small study may make one documented learning-rate recalibration at update 4,096. It must preserve the exact `q/u`, both Adam moments, scalar Adam counter `t`, target, loss, and `epsilon = 1e-6`; the recalibration restores the originally prescribed physical stress-update size and is then frozen for at most 4,096 additional updates. No reset, repeated schedule, fit-selected rate, or other optimizer change is allowed. The recovery gate must be evaluated on the resulting real solved state.

If that bounded amended phase passes, the face experiment transfers only the independently validated `epsilon = 1e-6`. It does not inherit the small study's learning rate or a hidden learning-rate schedule: source 73 still calibrates the face learning rate once at global step 64 from the frozen face gradient, Adam moments, projection, and physical `d_Q` rule below. If the amended small recovery does not pass, the face experiment remains blocked.

The immutable face source is `data/21-psd/optimizer-latest.pt`, SHA-256 `adf99d031850d423979feb11a0d844ee343757ffaa749ab77fe4aa4fd0de490d`, at global step 64. The associated `step-0064.npz` and `final.npz` are byte-identical, SHA-256 `df5642c97d774bdaacc7bb5fc8c361e352301edab736ac9d92e0306c69903596`. The source-20 and tensor-control hashes are `706a48576c7d781543525c428225de15ca7662f9ce86623a5f206d52ada975a9` and `fef2b003da6316d8d6b8f36c7a09724a05c1c954493e9377d5e7564d30e7f45e`.

All continuations retain the 288,235 active tetrahedra and all 1,729,410 orthonormal tensor coordinates. The fixture, passive materials, `Qref = 0.030201342281879193 MPa`, spectral constraint `0 <= Q/Qref <= 10 I`, projection implementation, boundary conditions, target, forward and adjoint solvers and tolerances, and lack of geometry rejection remain unchanged.

## Exact resumption contract

Source 20 evaluates and saves global step 64 before an update at that state. The checkpoint contains the evaluated `q64`, converged displacement `u64`, Adam `exp_avg`, `exp_avg_sq`, parameter group, and scalar Adam counter 64. Those moments resulted from the 64 updates that produced states 1 through 64; the next continuation update uses the newly recomputed gradient at state 64 and produces global state 65. No `q.grad` is serialized.

Before calibration, the continuation runner must:

1. require checkpoint keys `step`, `q`, `u`, `optimizer`, and `config`; require `step == 64`, `q.shape == (288235, 6)`, `u.shape == (228660, 3)`, one Adam state, moment shapes equal to `q`, and Adam counter 64;
2. prove exact equality of checkpoint `q/u` with both `step-0064.npz` and `final.npz`, and verify the saved step-64 solver receipt, active-cell ordering, configuration, and source/input hashes;
3. create the parameter, load the complete Adam state, and only then change a parameter-group epsilon or learning rate, because `load_state_dict` restores the saved hyperparameters;
4. set `q.grad = None`, solve from `u64`, and recompute the true step-64 objective and adjoint gradient; require finite values and successful forward and adjoint solves;
5. record the re-evaluated state as local update 0 / global step 64 without applying an extra update or counting the old trace row as new progress.

The re-evaluation must reproduce the saved unweighted fit, area-weighted fit and motion within `1e-6 mm`, and retain projected-feasible `Q` to the source-20 eigenvalue tolerance of `1e-10` in normalized coordinates. Since the data objective is the unweighted squared displacement RMS divided by three, its absolute replay tolerance is `(2 * F_parent * 1e-6 + 1e-12) / 3 mm^2`; the identity itself must agree within `1e-12 mm^2`. A mismatch stops the experiment rather than switching checkpoints or resetting state.

Before either face probe ran, the first source-73 calibration exposed an inconsistent scalar check: the original independently chosen `1e-9 mm^2` loss threshold rejected a `6.15137e-9 mm^2` change, although unweighted fit differed by `2.41529e-9 mm`, area fit by `3.71486e-10 mm`, and motion by `5.72322e-9 mm`. The failed run is preserved as `data/73-face-step-calibration`. The corrected squared-loss threshold above is derived from the already-declared RMS tolerance, rather than a new solver tolerance or an outcome-selected fit tolerance. Subsequent runs retain the original checkpoint and seed and record every replay difference.

## One-gradient calibration

Calibration uses only the common recomputed fit gradient at `q64`; it may not inspect a state-65 solve, later fit, motion, inversion, or surface metric. Every shadow update starts from a fresh exact clone of `q64`, `m64`, `v64`, and counter 64. Trial learning rates must never advance a shared optimizer state.

First compute the baseline shadow update with `epsilon = 0.01` and `learning_rate = 0.3`: apply the next Adam update with the common gradient, project by the unchanged spectral box, and measure

\[
d_Q = \left(\frac{1}{N}\sum_{i=1}^{N}
\lVert Q_i^{\mathrm{after}}-Q_i^{\mathrm{before}}\rVert_F^2\right)^{1/2}
= Q_{\mathrm{ref}}\left(\frac{1}{N}\sum_i\lVert\Delta q_i\rVert_2^2\right)^{1/2}.
\]

This is the accepted, post-projection physical tensor update in MPa. Also record the proposed pre-projection norm, projection change, lower/upper eigenvalue clipping fractions, and the existing scalar-coordinate `actual_update_rms`; its ratio between branches equals the ratio of `d_Q`.

Set the reduced-epsilon target to **2.0 times** the baseline `d_Q`. This gives a material acceleration over the current step while the 16-update screening bounds exposure. With `epsilon = epsilon_reduced`, search only the scalar learning rate in `(0, 0.3]`. Evaluate the exact projected shadow map from the same cloned state at every trial, find the first monotone crossing, and bisect until `d_Q_reduced / d_Q_baseline` is within 1% of 2.0. Freeze that learning rate before final trials. If no crossing exists, the mapping is nonfinite/nonmonotone near the crossing, or the tolerance is not met, stop and write the calibration failure; do not use an unprojected norm or a fallback learning rate.

The calibration receipt must contain epsilon, learning rate, state/gradient/moment hashes, counter before and after the shadow update, both proposed and projected coordinate and physical norms, clipping fractions, search trace, and selected ratio. The first final update must replay the shadow update within maximum normalized-coordinate error `1e-10` and relative `d_Q` error `1e-5`; bitwise identity is not required across the closed-form calculation, installed Adam kernel, and repeated CUDA gradient evaluation.

## Matched 16-update branch screen

Fork the verified step-64 state into exactly two final screening branches:

- **baseline:** epsilon `0.01`, learning rate `0.3`;
- **reduced epsilon:** the independently validated epsilon and the frozen calibrated learning rate.

Both branches start with identical `q/u/m/v`, counter 64, and the common true gradient, then perform exactly 16 updates to global step 80. Each evaluated state must come from a real nonlinear equilibrium solve and successful adjoint. Record real area-weighted fit RMS, motion RMS, objective, solver iterations and residual, `detF_min/max`, inverted-tetrahedron count, tensor amplitude/cap diagnostics, proposed and accepted update norms, cumulative accepted `d_Q` path length, and net `d_Q(q_k, q64)`. No linearized fit or motion may replace a solve.

The reduced-epsilon branch is selected only if both branches complete all 16 updates successfully, both improve on the common step-64 fit, and

\[
F_{80}^{\mathrm{baseline}}-F_{80}^{\mathrm{reduced}}
\ge \max(0.01\ \mathrm{mm},\ 0.05\,[F_{64}-F_{80}^{\mathrm{baseline}}]).
\]

This is a meaningful improvement beyond a numerically positive difference. Motion and inversion are reported side by side but do not silently alter this optimizer-selection rule. Any solve/adjoint failure makes that branch ineligible; no retry, changed tolerance, alternate seed, best-state substitution, or shorter successful prefix is allowed. A selected branch with worsening inversion remains an optimization result with limited physical interpretation.

If the reduced-epsilon branch does not meet this rule, continue the eligible baseline branch. If the baseline branch is ineligible, stop without selecting either branch.

## Continuation and stopping

Continue only the selected step-80 branch, preserving its `q/u/m/v`, counter, epsilon, and learning rate. Run blocks to global steps 144 and 208, then a final bounded block to global step 256. Save full checkpoints at least every 16 updates.

Raw gradient size is never a stopping rule. At every evaluated checkpoint, compute the fixed-step projected-gradient mapping of the current branch objective in normalized tensor coordinates,

\[
G_{1}(q)=q-P_{[0,10I]}(q-\nabla f(q)),
\]

using a clone and without changing Adam state. Record its RMS and maximum absolute coordinate. The fixed `eta = 1.0` is a diagnostic scale shared across all branches; it is not the calibrated Adam learning rate.

Stop only at a block boundary, before global 256, if either:

- area-fit improvement over that block is less than `max(0.01 mm, 0.5% of the block-start fit)`; label this **progress-limited**, not converged; or
- both projected-gradient RMS and maximum are no more than 5% of their common step-64 values; label this **relative projected-gradient criterion met at eta=1.0**, without claiming convergence or stationarity.

Otherwise stop at global 256 and state that the budget ended without an inverse-convergence claim. Preserve every completed state and any failure receipt.

## Smoothness and conditional rank follow-ups

After the fit-only continuation endpoint is frozen, fork it into three 16-update arms: a fit-only control, a weak-smoothness arm with `smoothness_weight = 1.4762928671047126` (one quarter of the frozen full strength), and a full-smoothness arm with `smoothness_weight = 5.9051714684188505`. Preserve `q/u/m/v`, counter, selected epsilon/learning rate, projection, and physics; the smoothness weight is the only difference among the arms. Save actual `Q/u/VTU` evidence at every step.

Match each smoothness arm separately to the fit-only control using only recorded states with positive local step in both arms; local step 0 is excluded so the common fork cannot become a trivial match. Report equal-step comparisons as well. A match is admissible only when fit differs by at most `max(0.02 mm, 0.5%)` and motion by at most `max(0.02 mm, 1%)`; for each smoothness strength, select the admissible pair with minimum squared distance after scaling by those two tolerances, breaking ties by the earlier smooth-branch step and then the earlier control step. If neither strength has a qualifying pair, report that no comparable-fit/motion smoothness claim is available.

Run a rank follow-up last, and only if a smoothness arm has an admissible match, lowers both tensor smoothness and the preregistered 5 mm full-face surface high-pass ratio by at least 10% at that match, and retains a rank-mixing fraction above `0.05`. If both strengths qualify, choose the weaker `1.4762928671047126` arm; otherwise choose the sole qualifying strength. Then fork that exact smooth checkpoint into matched 16-update smooth and smooth-plus-rank arms, adding only the previously frozen `rank_weight = 119.62893380703123`. Otherwise record why rank was skipped.

Changing epsilon changes the coordinate-wise denominator `sqrt(v_hat) + epsilon`; even after matching a global projected `d_Q`, it changes the update direction and its interaction with PSD projection. Changing the loss changes the incoming gradients and hence future Adam moments. The matched forks make these algorithmic effects auditable, but they are not pure scalar step-size comparisons. Better continuation fit demonstrates further progress for this bounded PSD inverse under the tested optimizer. It does not prove that PSD active stress can attain the target, and comparison with the differently parameterized Raw6 reference cannot establish that claim.
