# MouthOpen with a limited number of inverted cells

The user clarified that a few inverted cells are acceptable for this exploratory MouthOpen trial. This supersedes the zero-inversion gate of the first pruned-domain experiment. All earlier checkpoints, sources and results remain preserved. Numerical convergence, orientation and boundary intersections are recorded as separate properties.

The working interpretation of “a few” is at most 0.1% of the 1,144,268 retained cells: 1,144 cells. This is an experimental stop rule, not a physical validity threshold. Every accepted state must still have finite coordinates and Jacobians and a free-force norm at most `1e-10` in the existing code units. The same historical bulk materials and prescribed, chin-derived jaw pose are used. Contact energy remains off.

## Continuation stages

`src/45-forward-allow-inversions.py` resumed the strict run's 4.2679% jaw pose. It allowed negative determinants while retaining complete FEM boundary CCD. It reached 5.1579% with one inverted cell before the collision step bound reached the minimum pose increment.

The independent CPU audit `src/46-audit-local-folds.py` reproduced that final rejected carry's full-boundary CCD fraction of 0.75. Excluding the 10 boundary faces incident to vertices of the inverted cell made the fraction 1.0. For an earlier larger carry, excluding 14 faces only increased the fraction from 0.22265625 to 0.875. The evidence supports relaxing the final local fold obstruction while retaining collision checks elsewhere.

`src/48-forward-local-folds.py` resumes the 45 checkpoint. For every proposed linear move, it identifies inverted tetrahedra at the current and trial endpoints, excludes boundary triangles incident to their vertices, compacts the remaining surface, and runs CCD there. Accepted endpoints must also have no detected intersections on the surface outside the current inverted-cell neighborhoods. Complete FEM boundary intersections and excluded-face counts are reported separately. The excluded patch is geometrically unvalidated; this procedure cannot establish collision-free or physically valid tissue. The extracted boundary already contains 28 nonmanifold edges at rest. No bone-obstacle or containment check is claimed.

The pose increments, Newton and CG settings, force threshold and rollback semantics match the earlier trials. The continuation budget is 1,200 seconds after initialization, with at most 100 Newton steps and 3,000 CG iterations per pose attempt. Reaching a partial pose is reported as partial completion, even if its force solve converges.

The local-fold continuation stopped at 7.5581% pose with two inverted cells and converged forces because its retained-surface CCD still restricted the prescribed carry. `src/49-forward-contact-off.py` resumes that state using the original historical contact-off model, without the added CCD feasibility constraint. Complete FEM boundary intersections remain an endpoint diagnostic. This changes the exploratory geometric acceptance policy; it does not relax force convergence or the 0.1% inverted-cell limit. It is not a collision-free tissue simulation. The policy change was stated to the user before the new trial.

## Activation fit, conditional on reaching the full pose

`src/55-fit-mouthopen.py` requires a saved full-pose equilibrium before starting. It uses zero initial activation and fresh Adam moments, contraction-only full symmetric tensors (`psd6`), 200 attempted updates, an initial learning rate of 0.05, normalized L2 position loss, normal weight 1, and smoothness weight `7.2e-6` (the previous sweep's 10× coefficient). Effective active volumes and the adjacency graph are rebuilt on the pruned domain: 288,172 active cells and 501,313 same-region edges. The coefficient is reused, not recalibrated after pruning.

Both forward force convergence and adjoint relative residual at most `1e-7` are required. Failed proposals restore the activation and Adam moments, halve the learning rate, and count as skipped attempts. Allowing inverted cells does not enable approximate or unconverged gradients. The fixed mandible pose stays prescribed throughout fitting; the runtime DOF map is checked against the surviving `IsFixed` mask.

The activation fit follows stage 49's contact-off model. Complete FEM boundary intersections are saved as diagnostics rather than being used to reject updates. Inverted or intersecting results carry no physical validity claim.

The first fitting trial, stage 55, exposed repeated search-direction stabilization resets: four proposals exhausted the 100-step force-solve budget, repeatedly resetting the shift near `0.002766`. It was interrupted after nine completed proposals (five accepted, four skipped); the in-progress tenth proposal was discarded. Its last accepted state, matching Adam checkpoint, original running summary, interruption receipt, sources and logs are preserved. No unconverged proposal was accepted.

Stage 56 repeats the trial from the same stage-49 full-pose displacement, zero activation and fresh Adam. Its only solver-policy change is to reuse the Newton stabilization shift within each forward solve, matching the successful stage-49 continuation. The original energy, gradient, physical Hessian, unshifted adjoint, force and adjoint thresholds, 100-step Newton limit, 200-proposal budget and inversion cap stay fixed. The adapter is local to the new script; it does not modify the prior numerical sources. This is a search-policy trial, not a relaxation of numerical convergence.

The final gradient balance is `||eta dR/dS|| / ||dL2/dS||`, using full symmetric tensor covectors and the dual norm of normalized effective active volumes. The normal-loss gradient is excluded from the denominator. Final fit, normal RMS, activation roughness, inversion count and severity, boundary diagnostics and solver residuals must accompany any fit result.

Every executed stage uses a distinct Cherries output directory and stores source and input hashes. No automatic Git commits or pushes are made. Results and figures are written from completed run artifacts, with the geometric and mechanical limitations stated alongside them.
