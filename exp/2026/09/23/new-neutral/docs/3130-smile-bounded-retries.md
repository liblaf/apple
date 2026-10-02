# Continuing Smile with bounded trial failures

The user explicitly requested continuing Smile after the completed deadline publication and allowed a few forward or adjoint failures. The old 14:00 deadline is superseded for this continuation. The published audited result remains the baseline until a new endpoint is independently audited.

The continuation starts from audited Smile006, preserving q, exact normalized and physical pose, displacement, four Adam moments and q/pose counters 44. It uses the same collision-on L2 position, oriented normal and same-muscle activation smoothness objective. The force, collision/CCD, determinant, retained inversion-count/volume, adjoint residual, directional check and nonlinear Armijo gates remain active.

The new 3130 worker rejects only the archived bounded-dual Newton line-search assertion when its saved certificate identifies that exact failure. It permits three such retries per outer iteration, three forward failures and one known sparse-adjoint failure, each at half alpha. Each failed trial is retained in evidence, and no failed trial commits optimizer moments, coordinates or counters. Unrelated errors remain visible and terminal. The gradient re-evaluation must also meet Armijo before committing a step.

A CPU replay of the frozen failed QP certified alpha 0.00048828125 with the unchanged margins, trust limit and descent requirement. This supplies the initial step size; it does not establish a new physical endpoint or inverse convergence. Full forward, CCD and adjoint checks are required in the running batch.

The additive 3140 wrapper passed CPU startup checks against the complete verified006 mirror. Exact-state and source binding checks passed with CUDA uninitialized. Independent source review and all four numerical-failure classifier tests passed. The worker asserts that newly built objective terms match the parent before recording objective continuity.

The planned008 batch permits at most 25 new accepted updates and one hour of fitting, followed by the independent180 endpoint audit. The3150 supervisor owns only its fitter and auditor. The3160 launcher requires a fresh explicit cutoff and source hashes;3170 retrieves verified receipts and terminal bundles. No new renderer is part of the restart.

Smile008 launched at 2026-09-30T07:01:45.005423+00:00 and accepted four new updates, reaching q/pose counters48. Its terminal endpoint has loss1.4583670817080778, RMS4.944834033817685mm and force0.0008681875855114987N. The independent audit passed with99 retained inversions inside the declared100-cell allowance. Inverse convergence remains unproven.

At iteration5, the bounded dual produced the distinct assertion `Bounded dual Newton direction unresolved`. The narrow3130 classifier left that failure visible and terminal. The supervisor then audited the last durable endpoint. At 2026-09-30T07:11:22.543197+00:00, all owned processes had exited and the GPU was idle. Current receipts and the complete terminal bundle are collected with `src/3170-sync-smile-retries.py`. The next additive continuation requires evidence for this specific numerical trial failure and a CPU certificate from008's own frozen direction cache.
