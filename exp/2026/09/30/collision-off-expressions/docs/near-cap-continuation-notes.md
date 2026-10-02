# Notes for a future inversion-boundary stall

These are source-review findings, not evidence that the current solve is blocked. At the 04:00 CST check, iteration 8 was still being corrected after a successful pose projection. Do not interrupt a progressing solve to apply these ideas.

- The projection trigger is count >=95, and its constraints protect currently positive retained tets with 0<J<=0.05. It is not an equilibrium certificate or a bound on total inverted rest volume. Accepted forward force, inversion count and inverted volume remain the authority.
- `10-fit.py` can reject the tangent seed for exceeding the inversion allowance before the nonlinear corrector runs. A predictor is not an equilibrium. Exhausting only these seed rejections does not prove the target parameter trial is physically infeasible. In an additive continuation, record the predictor geometry and consider allowing the strict collision-off corrector to run, while leaving every final equilibrium gate unchanged.
- The small jaw QP is built for the full q proposal before the scalar trial backtracking. Failure of that full-step QP does not rule out a smaller coupled step. If this occurs, either try a smaller q proposal in an explicit continuation (preserving optimizer moments/counters), or form the projection for each scaled trial direction. Keep exact normalized pose coordinates and recheck descent after projection.
- Do not call a budget, seed rejection, QP failure, tiny accepted step or line-search stall convergence. The shifted adjoint and fixed-state pullback check do not establish reduced-objective stationarity.

## Updated evidence at 04:10 CST

At accepted step 12, larger fully corrected candidates fail the final inversion gate. This is no longer just the seed rejection issue. The accepted state has 99 inversions, RMS 4.090055492896568 mm, alpha 0.00048828125, and scaled gradient 0.05404318442301799 against threshold 0.0001375144569358037. The tiny accepted steps are not a stationarity certificate.

After the first chunk endpoint audit, compare the current relative adjoint shift 0.001 with 0.0001 (and an unshifted solve only if its residual contract succeeds). Record native unshifted residuals and changes in the joint directional derivative. Fully re-equilibrate reduced-objective probes at two decreasing scales along the projected joint direction. Record loss, force, retained count/fraction/minJ and the seed-versus-corrected determinants of newly inverted original cell IDs. Tighten force tolerance if needed to resolve objective differences; never loosen it or the inversion policy.

A feasible probe with a resolved loss decrease disproves constrained stationarity. Invalid probes or objective differences below numerical resolution leave the boundary unresolved and require further diagnosis; they do not prove convergence or impossibility. All proposed diagnostics remain pending, and no running source or process was changed.

## Force-threshold discontinuity in steps 18-20

Saved trial receipts show that an initial force just above 1e-8 invokes additional primal correction. For iteration 20 trial 1, alpha is 1.1920928955078125e-7, initial force 1.0000005316710734e-8, and the first PNCG coordinate displacement is 0.00030879056410851546 m. The final force is 9.373707641370253e-9, but 125 retained cells are inverted, so this candidate is rejected. Accepted tiny steps can remain just below the force stopping threshold and bypass those corrections. This is evidence of numerical resolution sensitivity, not proof that the physical model has no feasible equilibrium. See data/force-threshold-diagnostic.json for compact, source-bound observations.

The first diagnostic should re-equilibrate the unchanged endpoint parameters at stricter force tolerance and save displacement, loss, and geometry changes. If that baseline cannot satisfy the existing geometry gates, mark the baseline unresolved and retain the diagnostic evidence; do not run derivative probes as though a valid stricter baseline existed.

## Scheduling a diagnostic if the serial queue does not become idle

Read-only controller review at 04:47 CST recommends allowing Smile-001 and MouthOpen-002 to finish first. A tiny step does not satisfy the gradient monitor. With the current minimum_trial_alpha=0, roundoff may produce non-descent trials and eventually line_search_stalled; do not expect a positive resolution floor to stop it.

If finite-budget cycling persists, a reversible reservation can stop only the verified v1 queue parent while leaving its identified active fit child running. Before any signal, record and verify PID, starttime, cmdline, queue-state ownership and direct parent-child relation. Once the same parent is stopped, wait for that exact fit child to exit or become a zombie, all descendants to stop, and GPU compute occupancy to become empty. Only then run the already staged 70 probe against previously audited MouthOpen-001 in a new directory. No second queue is needed. After the probe exits and GPU is empty, resume that same parent; its original Popen.wait then reaps the child, runs the normal audit, and updates the queue. A dedicated controller must record every phase and provide finally/recovery logic for SIGCONT, with process identity rechecked before resuming. Verify the original queue advances afterward. A transient GPU gap without parent reservation is unsafe. This is a reviewed future operating plan only: no signal or process control has been performed.
