# Continue unrestricted face fits from update 100 to 200

The user requested continuation after the four neutral-start, 100-update Adam
fits. Extend all four conditions to update 200, retaining each branch's controls,
equilibrium displacement, Adam first and second moments, and update counter.
Keep the existing physics, loss weights, normalization, learning rate .3,
epsilon .01, betas .9/.999 and solver tolerances. Do not restart from neutral,
replace the inverted endpoints with earlier states, or tune a coefficient.

The original `data/10-comparison` remains unchanged. Save continuation artifacts
under `data/40-continuation`, including the untouched history 0..100 and newly
accepted states 101..200. Copy old neutral and geometry receipts for full-history
auditing; label the run explicitly as continuation, not another neutral start.

The step-100 checkpoint stores q100/u100 and the Adam state after exactly 100
updates. Recompute the gradient at q100 using u100 as the equilibrium seed, then
apply the next Adam update to obtain q101. Do not append a duplicate trace or
solver-receipt row at step 100. Save its replay separately. The old adjoint cache
was not serialized; clear it between branches, then retain within-branch warm
starts. Solver replay is not promised to be bitwise identical to the original
evaluation, which used u99 as its initial guess.

Before main continuation, require source/fixture hashes to match the original
run, verify checkpoint controls/displacements and optimizer states on CPU, and
check the restored Adam update against its explicit bias-corrected formula.
Run a one-update smoke for all four branches from the original step-100
checkpoints, then audit both the inherited history and actual first new update.
The main run resumes the same original checkpoints independently of that smoke.
Replay allows a maximum vertex displacement difference of 1e-6 m, position-RMS
difference of .001 mm, normal-angle RMS difference of .01 degrees, determinant
minimum difference of .001, and physical gradient RMS agreement within 2%.
Objective differences must stay below 1e-4 times max(1, the original value),
and normal loss must agree within 1e-4 relative error. Activation variation
must agree to a scaled 1e-12 and inversion
counts exactly. Controls and Adam moments/counter remain exactly equal before
the next update. Save all replay errors and bounds before checking them.

The first smoke used a 1e-9 m displacement threshold and stopped before any
updates: reevaluating the finite-tolerance equilibrium shifted vertices by
3.50e-8 to 7.92e-8 m. That threshold was tighter than the repeatability of the
existing solver. Its receipts are preserved as `data/39-smoke-initial-replay-threshold`.
Only the replay comparison bounds were revised; physics and optimization
settings were unchanged. The revised bounds assess negligible changes in the
reported fit and shape metrics rather than claim bitwise equilibrium replay.
Only the original gradient RMS was saved, so this check cannot certify equality
of the old and replayed gradient directions.

Save checkpoints every ten updates, the first resumed state, the last accepted
state, best objective, best noninverted state, full traces and solver receipts.
All four existing endpoints have one inverted tetrahedron. Continue reporting
physical determinant minima, inversion counts and non-SPD activation counts;
no new determinant barrier or projection is introduced. If a forward/adjoint
solve fails, preserve that proposal and the last accepted checkpoint, and let
the independent remaining branches continue. A failed branch does not count
as completed. Do not alter tolerances or resume through a failed solve silently.

Compare updates 100 and 200 (or an explicitly labeled latest shared saved
checkpoint if any branch fails). Report position RMS, normal-angle RMS,
target-relative surface residuals, activation variation, motion and validity
separately. Inspect final gradients and recent objective changes; completing
200 updates is not a convergence certificate. Reuse the verified inversion-free
comparison and show the full 0..200 histories with the continuation boundary.
