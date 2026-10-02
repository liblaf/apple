# Where inverse-physics time goes

Measured complete-update profiles, 22 September 2026. All four numerical measurements ran sequentially on the same Paratera RTX 4090, with eight IPC threads.

## Main findings

The dominant cost changes with the state. In the regularized Smile fit, speeding up the adjoint exposes forward Newton-CG as the main remaining cost. In the difficult zero-smoothing case, nonlinear forward convergence remains the bottleneck. CCD is significant but does not explain most of the runtime. Rebuilding discrete contact, PNCG directional curvature and repeated Hessian products also consume substantial time.

The optimized variant changes only the adjoint: relative tolerance `1e-4` and CUDA contact products during its linear solve. The previous variant uses `1e-7` and CPU contact products. Both retain the same hybrid forward solver, shift reset, final force threshold `1e-8`, active stress, full cranium/mandible/eyeball collision, Adam 0.3, and no magnitude/jaw prior or outer rejection. Each case retains its original smoothing weight.

## Full inverse-update duration

| Case / adjoint | Full update | Forward | Adjoint | Other |
| --- | --- | --- | --- | --- |
| Regularized / previous | 19.44 s | 7.42 s (38.2%) | 9.56 s (49.2%) | 2.46 s (12.6%) |
| Regularized / optimized | 11.31 s | 7.45 s (65.9%) | 2.70 s (23.9%) | 1.16 s (10.2%) |
| Zero smoothing / previous | 140.78 s | 121.22 s (86.1%) | 17.04 s (12.1%) | 2.51 s (1.8%) |
| Zero smoothing / optimized | 81.86 s | 72.00 s (88.0%) | 8.63 s (10.5%) | 1.23 s (1.5%) |

Each measurement is the actual `Fitter.fit_step` advancing saved step15 to step16. It includes the operative checkpoint load, projected Adam proposal, objective/gradient evaluation, diagnostics and checkpoint/log writes. Model creation, preparation copies and reconstruction of the initial warm adjoint are excluded. Both variants receive the same reconstructed adjoint at the saved step15 state. These are warm-started update profiles, unlike the previous zero-initial-adjoint microbenchmarks.

“Other” includes optimizer work, parameter projections, regularizers, metrics, I/O and remaining Python/dispatch overhead. Its residual is larger on the first profile in each process, consistent with first-use setup; that residual is not fully attributed. We do not treat differences in this category as an algorithmic speedup.

## CCD measured directly

| Case / adjoint | Queries | CCD total | % forward | Broad phase | Narrow phase | Transfer / wrapper | Mean/query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Regularized / previous | 9 | 0.419 s | 5.7% | 0.293 s | 0.114 s | 0.013 s | 46.6 ms |
| Regularized / optimized | 9 | 0.405 s | 5.4% | 0.279 s | 0.116 s | 0.011 s | 45.0 ms |
| Zero smoothing / previous | 243 | 14.523 s | 12.0% | 8.392 s | 5.832 s | 0.299 s | 59.8 ms |
| Zero smoothing / optimized | 271 | 17.420 s | 24.2% | 8.002 s | 9.121 s | 0.297 s | 64.3 ms |

CCD total is measured around the actual contact maximum-step query and includes GPU-to-host position preparation, swept broad-phase candidate construction, narrow-phase continuous testing and wrapper overhead. Counts include the initial boundary-motion check, which earlier inner-solver counters omitted. “Mean/query” is total divided by count, not a sampled median. These replace count-times-MouthOpen estimates for these four Smile updates.

Static candidate construction during contact-state initialization or update is **not** included in the CCD category. It appears separately as contact rebuild. The narrow-phase timer wraps `Candidates.compute_collision_free_stepsize`; its duration includes the native query orchestration, not only a single geometric predicate.

## Forward: additive cost breakdown

| Exclusive cost (s) | Regularized / previous | Regularized / optimized | Zero smoothing / previous | Zero smoothing / optimized |
| --- | --- | --- | --- | --- |
| CCD broad phase | 0.293 | 0.279 | 8.392 | 8.002 |
| CCD narrow phase | 0.114 | 0.116 | 5.832 | 9.121 |
| CCD remainder | 0.013 | 0.011 | 0.299 | 0.297 |
| contact rebuild | 0.394 | 0.380 | 10.533 | 9.309 |
| contact fun / grad / diag | 0.073 | 0.068 | 1.986 | 1.906 |
| contact Hessian assembly | 0.049 | 0.054 | 0.213 | 0.118 |
| FEM operators | 3.616 | 3.700 | 47.199 | 17.243 |
| directional curvature (FEM + contact) | 0.374 | 0.370 | 13.210 | 14.890 |
| contact HVP / transfer | 1.518 | 1.522 | 21.162 | 6.199 |
| linear algebra / control remainder | 0.978 | 0.954 | 12.393 | 4.916 |

All rows above are exclusive allocations and sum to forward time. FEM operators include volumetric tissue and skin work, along with launch/synchronization overhead; these are completed-operation wall times, not isolated GPU-kernel timings. Directional curvature is the combined `model.hess_quad` call used by coarse PNCG; its FEM and IPC portions were not separately instrumented. Contact HVP/transfer includes sparse multiplication, vector movement and gather/scatter, excluding separately counted Hessian assembly. The residual category includes Krylov vector operations, diagonal preconditioning, line-search/control work, material/boundary setup, and timing overhead.

Contact state updates rebuild static candidates in `OwnedContact.update`; CCD separately builds swept candidates. This means broad-phase work occurs in both places. Candidate reuse could reduce this duplication, but it needs proof that reused candidates cover every accepted and backtracked position before it is implemented.

## Forward: nested algorithm stages

| Case / adjoint | PNCG steps / time | Newton steps / time | Newton CG incl. retries | Recorded shift retries | Newton HVPs | Adjoint HVPs |
| --- | --- | --- | --- | --- | --- | --- |
| Regularized / previous | 6 / 1.07 s | 2 / 6.18 s | 5.94 s | 0.37 s | 1360 | 2136 |
| Regularized / optimized | 6 / 1.05 s | 2 / 6.27 s | 6.05 s | 0.38 s | 1392 | 791 |
| Zero smoothing / previous | 234 / 42.69 s | 8 / 78.36 s | 77.40 s | 45.39 s | 17278 | 3863 |
| Zero smoothing / optimized | 264 / 46.50 s | 6 / 25.35 s | 24.71 s | 12.06 s | 5714 | 2546 |

This is a second view of the same forward time. Newton-CG is inside Newton, and shift retries are inside its linear work; **do not add these columns together or to the preceding table**. The remainder outside PNCG/Newton covers forward initialization and terminal checks. Successful linear solves and unsuccessful curvature/regularization attempts both cost time.

Both variants use identical forward code and bitwise-identical proposed stress/jaw tensors, yet hard-case nonlinear paths differ. The hard endpoints differ by **0.02856 mm skin RMS** and **0.29978 mm maximum-node displacement**, despite both meeting the force/contact/no-inversion conditions. The regularized difference is only 0.0000022 mm skin RMS. Previous unchanged-control repeats already showed this variability. The table describes observed paths; it is not evidence that changing an adjoint backend accelerates a forward solve. Forward repeatability deserves attention before attributing timing gains to nonlinear solver changes.

The hard historical solve spent **45.39 s** in unsuccessful Newton regularization attempts and made **17,278 Newton HVPs**. Exact contact-Hessian assembly was just **0.213 s**. This is a repeated-application/convergence bottleneck, not an assembly bottleneck. The shorter hard forward path made 5,714 Newton HVPs; its lower cost is largely explained by different linear-solve work.

## Adjoint: additive cost breakdown

| Exclusive cost (s) | Regularized / previous | Regularized / optimized | Zero smoothing / previous | Zero smoothing / optimized |
| --- | --- | --- | --- | --- |
| contact Hessian assembly | 0.022 | 0.021 | 0.029 | 0.020 |
| contact HVP / transfer | 2.455 | 0.160 | 4.410 | 0.539 |
| FEM operators | 5.457 | 1.990 | 9.854 | 6.402 |
| contact fun / grad / diag | 0.003 | 0.003 | 0.003 | 0.003 |
| directional curvature (FEM + contact) | 0.000 | 0.000 | 0.000 | 0.000 |
| adjoint linear solve control | 1.571 | 0.490 | 2.693 | 1.614 |
| adjoint control remainder | 0.056 | 0.041 | 0.053 | 0.050 |

The adjoint applies the unshifted physical Hessian. The optimized path uploads the exact IPC contact Hessian once per owned state, applies it on CUDA during CG/MINRES, then restores CPU contact for the residual and boundary derivative checks. No CCD query is part of the adjoint. A fresh discrete contact state is still built for its owned saved displacement. Remaining adjoint control includes state reconstruction, mixed derivatives and boundary gradients. Upload/assembly and sparse matvec work are charged separately where the hooks resolve them.

## Smaller costs inside the update

| Selected operation (inclusive s) | Regularized / previous | Regularized / optimized | Zero smoothing / previous | Zero smoothing / optimized |
| --- | --- | --- | --- | --- |
| Shape diagnostics | 0.691 | 0.676 | 0.732 | 0.745 |
| Adam.step (including dispatch) | 0.083 | 0.001 | 0.080 | 0.001 |
| Stress PSD projections (update + stationarity) | 0.176 | 0.141 | 0.178 | 0.145 |
| Update checkpoint read | 0.024 | 0.025 | 0.022 | 0.024 |
| Checkpoint serialization/write | 0.178 | 0.186 | 0.180 | 0.180 |
| Stress regularizer evaluation | 0.011 | 0.001 | 0.010 | 0.001 |
| Skin-position loss | 0.001 | 0.001 | 0.001 | 0.001 |

These explanatory timings lie inside the top-level “other” category. Projections occur both for the Adam proposal and the stationarity diagnostic. Checkpoint write excludes preparation copies and includes the timed serialization/write function, but not every preceding CPU-tree conversion. Their sum is not a complete decomposition of “other”; construction, state restoration, cloning, conversion and dispatch remain in its residual.

## Relation to the complete fitting runs

The prior full trajectories provide a less intrusive timing baseline:

| Run | Updates | Total | Forward | Adjoint | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original PNCG, regularized | 20 | 1455.17 s | 85.8% | 12.3% | 1.9% |
| Hybrid, regularized | 20 | 781.43 s | 71.4% | 25.6% | 3.0% |
| Hybrid, zero smoothing | 19 valid | 3013.78 s | 85.7% | 13.5% | 0.8% |

In the regularized hybrid's last five updates, the split was 32.4% forward, 59.5% adjoint and 8.1% other. In the zero-smoothing run's last five valid updates, forward remained 80.2%. The zero-smoothing trajectory has a different objective and is not a matched solver-speed comparison. Its failed proposal20 is excluded. Historical figures use authoritative `iteration-timing.jsonl` ledgers, not differences between intermediate trace timestamps.

## Optimization priorities

1. **Reduce Newton/adjoint Hessian products.** Improve conditioning/preconditioning and reduce repeated failed shift trials. Separately profile FEM kernel execution versus launch/synchronization overhead before deciding which to optimize. The previous scalar shift-reuse experiment did not establish endpoint agreement, so it remains experimental.
2. **Reduce contact rebuild work.** Profile-preserving broad-phase/candidate reuse is a concrete target; retain owned backward states and full collision coverage.
3. **Address expensive PNCG directional curvature.** It is a separate cost from CCD. Alternative curvature evaluation or switching policy needs matched convergence and gradient checks, because changing it changes the nonlinear path.
4. **Accelerate CCD where its measured share warrants it.** Broad phase and narrow phase have different costs; optimizing only the narrow phase cannot remove the full CCD total. Do not skip collision checks.
5. **Reduce diagnostic frequency if needed.** Shape statistics and checkpoint handling matter once updates become short. Adam arithmetic and skin-loss evaluation are not the main bottleneck.

These are priorities informed by the measurements, not additional implemented optimizations.

## Measurement limits and reproduction

Each scoped GPU operation is synchronized so its completed work is charged to the correct phase. These fences perturb overlap and add overhead. Inclusive parent time contains its children; only exclusive allocations are added. Every tree's accounting identity was checked. The reported update total is the `inverse_update` root, not `outer_profile_wall_seconds`, which also includes timing-hook setup and a final fence.

There is one profiled update per case/variant, no statistical confidence interval and no new 20-update optimized trajectory. Forward-path variability prevents interpreting all cross-row timing differences as optimization effects. Parameter gradients were checked in the preceding fixed-state study; these duration probes primarily establish timing and physical validity, not a new long-trajectory accuracy certificate.

Run from `exp/2026/09/22/solver-performance` on the compute host:

```sh
DEBUG=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 CHERRIES_NAME=inverse-duration CHERRIES_TAGS=smile,performance,inverse-profile \
  python src/40-profile-inverse-updates.py --output-dir data/NEW-RUN \
  --cases regularized --variants historical,candidate
# Run zero_smoothing separately after the first process exits.
```

Numerical output: `data/inverse-duration-regularized-002` and `data/inverse-duration-hard-001`. Each contains source snapshots, hashes, a common warm-start receipt, full timing trees and independent resulting checkpoints. `regularized-001` is preserved as a failed hook preflight; no inverse update completed there. Analysis: `src/41-analyze-inverse-duration.py`, `data/inverse-duration-analysis-001`. Historical audit: `data/historical-duration-audit-001`. Report: `src/42-report-inverse-duration.py` and this file.

Cherries debug runs retain local logs and receipts; no remote Comet URL is present. Validation includes timer nesting/static/slotted/exception-restoration CPU checks, compilation/Ruff, completed physical updates with no inversions, nonzero GPU-adapter use for candidate runs, and additive timing-tree checks. Profiling wrappers are restored after each update; no production physics source was changed.
