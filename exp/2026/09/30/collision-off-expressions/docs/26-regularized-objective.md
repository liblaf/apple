# Calibrated position, normal and activation-smoothness objective

The user requested that MouthOpen and Smile use L2 position error, surface-normal matching and spatial activation regularization, with weights calibrated for this 3D face model. This supersedes future L2-only continuations. Existing L2-only results, including strict MouthOpen001, remain immutable provenance and are not evidence of fitting the combined objective.

## Objective and units

The planned objective is `J = Lpos + beta N + eta R`. `Lpos` preserves the authoritative loaded-neutral skin-area-weighted squared position error divided by weighted squared target motion `D2`. `N` compares corresponding oriented skin triangle normals using fixed loaded-neutral area weights. `R` penalizes Frobenius jumps of the symmetric activation tensor over shared faces between retained active tetrahedra with the same MuscleId. Cross-muscle edges are excluded; this is not an activation-magnitude penalty.

The initial normal-weight calibration hypothesis equates a 2 mm vector position RMS with a uniform 5 degree normal error: `beta = (0.002 m)^2 / (D2 * (1 - cos(5 degrees)))`. Exact coefficients must be recomputed and saved from each expression's loaded target and neutral; the old 2D coefficient is inapplicable. The normal anchor is a declared testable scale choice, not a claim of an optimal weight.

Graph conductance uses face geometry and harmonic muscle fraction. Its geometry-derived length normalization must be distinguished from a specified physical correlation length. The reference retained graph has 288,172 active cells, 501,313 same-muscle edges and 103 MuscleId labels; 406 cells have no same-muscle neighbor. Final implementation and receipts must independently verify those counts and all IDs.

At each certified starting state, an unshifted adjoint will measure the material gradient of `Lpos + beta N`. A volume/Frobenius dual norm comparison to the direct gradient of `R` sets a reference positive `eta` with smooth/data gradient ratio 0.1. This is a calibration rule; its useful range is tested by matched finite pilots.

## Matched pilot selection

Each expression starts every candidate from the same q, exact normalized pose, equilibrium, four Adam moments, counters and original convergence reference/history. MouthOpen uses audited strict001 at 75/75. Smile uses the unchanged Smile004 parameters and optimizer state at 86/86, with the already certified 1e-12 equilibrium from baseline diagnostic82 recorded as a separate zero-update refinement.

The initial pilot candidates use smooth-weight multipliers 0, 0.3, 1 and 3, with the same normal coefficient and at most 25 updates each. Zero is an explicitly labelled calibration control only. Each endpoint requires an independent rebuilt audit under the original force, inversion, IsFixed, free-lip, exact reference/skin and collision-disabled policy. A finite pilot budget is not inverse convergence.

The initial selection screen requires a positive candidate to reduce activation roughness at least 20% relative to the smooth-off control, with position RMS and normal-angle RMS each no more than 5% worse than that control. If no positive candidate passes, refine positive weights or report the measured tradeoff as a concrete decision; do not silently use zero smoothness as the final requested objective. A stalled or invalid control cannot establish a reliable matched comparison.

## Evidence and implementation status

New source modules and runners are being implemented additively. Normal and smoothness directional derivatives, tensor conventions, zero-gradient cases, graph provenance and physical-unit normal calibration require checks before GPU pilots. Combined gradients must include the direct activation-regularizer derivative as well as the implicit displacement response. The accepted line search must use the combined objective. Position RMS must use only the positional term; separate normal, roughness and weighted contributions must be saved.

The objective transition must be explicit in checkpoints, receipts and plots. Preserve original convergence metadata as provenance and do not join old L2 losses to new combined losses as if they were comparable. Independent post-run convergence probes remain necessary.

Old L2-only source85 is superseded and must not launch. Already started finite Smile84 diagnostic522602 remains under controller521121; no cancellation signal was sent. Its completion/recovery and GPU ownership must be verified before any new numerical launch. See `data/regularized-objective-transition-001.json` and `data/operations.json` for process state and implementation status.

## User deadline and revised allocation

At approximately 11:03 Asia/Shanghai, the user requested that numerical computation finish before 14:00 today (2026-09-30 06:00 UTC), maximizing progress on both MouthOpen and Smile before starting visualization. This supersedes the indefinite computation plan and the full matched pilot sweep for this deadline. The initial positive smoothness coefficient will use the declared 0.1 gradient-ratio calibration; the sensitivity screen remains explicitly incomplete. Numerical fits stop by 05:45 UTC with independent audits and owned-worker cleanup reserved before 06:00 UTC. Both expressions receive bounded allocations on the single GPU. Saved accepted checkpoints and the original physical policy remain required; the time cutoff cannot count as inverse convergence. Additive controller88 will keep the superseded L2 queue stopped after computation so it cannot restart beyond the cutoff. Start full anatomy/lineage visualization after numerical work ends.
