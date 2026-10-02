# Adjoint accuracy and forward optimization protocol

Authorized follow-up to the Smile solver recap, 22 September 2026. Test adjoint tolerances at fixed saved states, then compare shift reuse and GPU contact Hessian products. All timed numerical runs are sequential on the same Paratera RTX 4090 with eight IPC threads. This is a bounded saved-state study, not a new 20-update inverse trajectory.

## Frozen inputs and unchanged physics

Use the completed regularized hybrid checkpoint and the last valid zero-smoothness checkpoints from `smile-fit-adam03-unconditional-004` and `smile-fit-adam03-no-smoothness-005`. Preserve the input manifests, active stress, material fields, jaw/PSD bounds, full cranium/mandible/eyeball collision, forward threshold `1e-8`, and unshifted physical Hessian for implicit differentiation. The baseline uses the original CPU contact product and Newton shift-reset policy.

## Adjoint-only sweep

At each selected saved state, hold displacement, materials, loss and initial adjoint fixed; compare relative tolerances `1e-7`, `1e-6`, `1e-5`, and `1e-4` to `1e-8`. Use identical zero initial adjoints because previous-step adjoints are not checkpointed. Report this cold-start timing scope explicitly. Do not rerun the primal or modify the accepted checkpoint. Measure parameter-data gradient errors separately from total regularized gradients, cosine, actual next projected-Adam update differences with saved optimizer moments, true linear residual, HVP counts and time. A lower residual is not assumed to equal the parameter-gradient error.

Provisional useful-accuracy gates for selecting a candidate: data-gradient relative L2 at most `1e-3`, total-gradient relative L2 at most `1e-3`, and projected stress-step relative L2 at most `1e-3`; record jaw absolute/relative errors separately when its projected step is zero. These are engineering comparison targets, not a convergence theorem. Keep `1e-7` for historical results. A passing candidate is an opt-in setting until an inverse trajectory validates accumulated effects.

## GPU contact product

CPU-assemble the same exact sparse contact Hessian and upload once per owned state. Compare deterministic full-model HVP directions to the CPU route (relative L2 at most `1e-10`), with first-use upload and subsequent cached timings separated. Validate state-update invalidation, switching back to an earlier owned state, and installation/removal on the actual collision object. Preserve CPU broad phase and CCD. This does not claim a full GPU IPC implementation.

After the forward replays exposed endpoint variability, add an explicitly scoped follow-up: install the GPU adapter only inside `runtime.solver.solve`, then restore the CPU contact product before the physical residual and fixed-boundary derivative checks. Compare its saved `1e-8` and `1e-4` output vectors directly against newly saved CPU `1e-8` references on both states. Preserve the earlier whole-adjoint and forward experiments as separate evidence. The fitter defaults to this `adjoint` scope when GPU contact is requested; GPU contact itself remains opt-in.

## Newton shift reuse

Retain the original `reset` policy as the control. The opt-in `reuse` policy starts each new Newton iteration at one tenth of the previous accepted shift normalized by mean absolute Hessian diagonal. Shifts below the existing `1e-6` relative floor return to zero. Following the recorded easy-state failure in `shift-reuse-easy-001`, revision2 also resets to the original zero-shift trial when force is within three times the final tolerance. This is an empirical guard; agreement is still measured rather than assumed. Every CG, descent, Armijo, force and contact check remains unchanged. Reuse is local to a single solve; it does not alter the energy or adjoint Hessian. CPU tests verify the same SPD solution and nonconvex equilibrium with fewer retries.

## Matched forward replay

Reconstruct one full Adam0.3 proposal from saved activation/jaw gradients and optimizer moments at checkpoint15. Compare `reset_cpu`, `reuse_cpu`, `reset_gpu`, and `reuse_gpu` from exactly the same parameters and displacement seed. Record all nonlinear/linear counts, shift attempts, forward times, force/contact/inversion checks, and endpoint displacement differences. Compare both an easy regularized state and a difficult zero-smoothing state if runtime permits. Declare endpoint agreement at observed-skin RMS <= `0.001 mm` and maximum full-node difference <= `0.01 mm`. Preserve and report failed variants; do not silently relax tolerances or change physical gates.

## Reporting

Freeze sources and input hashes in each run. Exclude model construction, transfer and rendering from solve timings, but retain setup/upload timings separately. Use local Cherries debug receipts consistent with this task's existing experiments. Report single-run and repeated timing scope accurately. No code or setting is called faster or sufficiently accurate solely from CPU tests.

An unchanged hard `reset_cpu` repeat is included to assess baseline variability. It uses the identical checkpoint and Adam proposal; compare its endpoint to the first control, without treating its automatic self-comparison in the replay summary as cross-run agreement.
