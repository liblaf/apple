# Hybrid-first cold forward profile

The instrumented forward solve took 3.167 s and did not converge; terminal force was 2.0252e-05 against 1.0e-08. This is one cold fixture profile and does not establish a general solver speedup.

PNCG recorded 27 finite accepted updates, then the forward solve failed before producing a structured handoff or convergence receipt.

Sparse Newton was not entered, so the Newton timing and CG tables are empty by design.

Endpoint validation found 0 inverted tetrahedra and contact feasibility False.

The accepted-state trace ends after PNCG step 27, but the saved endpoint is after attempted update 28, where evaluation failed. Its terminal force and collision audit describe that failed endpoint, not the last finite trace state.

The nonfinite energy is the contact-feasibility guard returning +inf at a minimum gap of 8.81887e-09 m, below its required 1e-08 m clearance; it does not by itself demonstrate mechanical-energy divergence. The one-shot PNCG update has no rollback. The 60-step earliest stall handoff could not fire before this failure.

The finite trace reached its lowest force 6.07521e-06 at PNCG step 11, then ended at 1.37865e-05 at step 27. Directional-curvature evaluation used 0.922 s and CCD used 0.816 s, together 56.7% of timed PNCG.

No sparse/reference HVP validation was recorded because sparse Newton was not entered.

The baseline shares checkpoint, seed, activation, Newton settings, GPU model, IPC version, and synchronized timing. Its PNCG curvature policy differs by design. Sequential runs may have different compiled-kernel cache state, and the timing tree does not isolate JIT; first-construction and total-wall deltas therefore are descriptive, not a clean cold-JIT speed comparison. The runs also have different termination reasons, so their wall-time ratio is not a completed-solve speed comparison.

![Residual, energy, and additive solver timing](../data/hybrid-first-profile-report-006/60-clamped-hybrid-profile.png)

## Protocol

The run starts from the recorded cold displacement with saved Smile activation and fixed prestress. It uses undamped PNCG with per-contribution-clamped curvature, followed only when needed by exact GPU free-CSR Newton with an absolute exact Jacobi diagonal. IPC is PyPI 1.6.0. Model construction and ordinary-operator prewarm are excluded; first sparse Newton construction is included. The profiler adds synchronization and changes overlap, so values are instrumented completed-operation wall times. Kernel JIT is not separately timed.

| Timing | Seconds | Interpretation |
| --- | --- | --- |
| Model construction | 8.219 | outside forward timing |
| Operator prewarm/JIT | 0.210 | excluded from forward timing |
| Forward | 3.167 | inclusive phase; do not add with children |
| PNCG | 3.065 | inclusive phase; do not add with children |

Parent phase timings are inclusive explanatory values and must not be added to their children. The phase allocations below use exclusive nodes only, so each sums to its own phase without nested double counting.

| PNCG exclusive allocation | Seconds |
| --- | --- |
| Phase controller/other | 0.118 |
| Energy evaluation | 0.302 |
| Trace/validation | 0.002 |
| PNCG diagonal | 0.257 |
| PNCG directional curvature | 0.922 |
| CCD | 0.816 |
| Contact/state update | 0.418 |
| Force evaluation | 0.229 |

| Newton exclusive allocation | Seconds |
| --- | --- |

## Comparison with baseline

| Metric | Clamped run | Baseline | Delta |
| --- | --- | --- | --- |
| Forward wall time (s) | 3.16742 | 49.9345 | -46.7671 |
| Terminal force | 2.0252e-05 | 1.75721e-06 | +1.84948e-05 |
| PNCG accepted updates | 27 | 11 | +16 |
| Newton accepted steps | 0 | 100 | -100 |
| Successful CG iterations | 0 | 3501 | -3501 |

The PNCG policy intentionally differs. The comparison is therefore an observed fixture comparison, not an attribution of every delta to clamping.

## Calls

| Scope | Calls | Inclusive s | Exclusive s |
| --- | --- | --- | --- |
| forward | 1 | 3.167 | 0.021 |
| forward/pncg | 1 | 3.065 | 0.118 |
| forward/pncg/model/hess_quad | 28 | 0.922 | 0.280 |
| forward/pncg/model/max_step_size | 28 | 0.816 | 0.001 |
| forward/pncg/model/max_step_size/collision/max_step_size | 28 | 0.815 | 0.029 |
| forward/pncg/model/hess_quad/collision/hess_quad | 28 | 0.642 | 0.010 |
| forward/pncg/model/hess_quad/collision/hess_quad/ipc/per_contact_gauss_newton_batch | 28 | 0.632 | 0.632 |
| forward/pncg/model/update | 28 | 0.418 | 0.001 |
| forward/pncg/model/update/collision/update | 28 | 0.417 | 0.012 |
| forward/pncg/model/max_step_size/collision/max_step_size/ipc/ccd | 28 | 0.401 | 0.401 |
| forward/pncg/model/max_step_size/collision/max_step_size/ipc/candidates_build | 28 | 0.385 | 0.385 |
| forward/pncg/model/update/collision/update/ipc/candidates_build | 28 | 0.373 | 0.373 |
| forward/pncg/model/fun | 29 | 0.302 | 0.001 |
| forward/pncg/model/fun/warp_model/fun | 29 | 0.279 | 0.279 |
| forward/pncg/model/hess_diag | 28 | 0.257 | 0.001 |
| forward/pncg/model/grad | 28 | 0.229 | 0.001 |
| forward/pncg/model/hess_diag/warp_model/hess_diag | 28 | 0.213 | 0.213 |
| forward/pncg/model/grad/warp_model/grad | 28 | 0.181 | 0.181 |
| forward/pncg/model/grad/collision/grad | 28 | 0.047 | 0.047 |
| forward/pncg/model/hess_diag/collision/hess_diag | 28 | 0.043 | 0.019 |
| forward/model/grad | 1 | 0.036 | 0.000 |
| forward/pncg/model/update/collision/update/ipc/normal_collisions_build | 28 | 0.033 | 0.033 |
| forward/model/grad/warp_model/grad | 1 | 0.032 | 0.032 |
| forward/collision/state_at | 2 | 0.029 | 0.000 |
| forward/collision/state_at/collision/update | 2 | 0.029 | 0.001 |
| forward/collision/state_at/collision/update/ipc/candidates_build | 2 | 0.026 | 0.026 |
| forward/pncg/model/hess_diag/collision/hess_diag/ipc/gauss_newton_hessian_diagonal | 28 | 0.025 | 0.025 |
| forward/pncg/model/fun/collision/fun | 29 | 0.022 | 0.022 |
| forward/collision/max_step_size | 1 | 0.015 | 0.001 |
| forward/collision/max_step_size/ipc/candidates_build | 1 | 0.013 | 0.013 |
| forward/model/grad/collision/grad | 1 | 0.003 | 0.003 |
| forward/collision/state_at/collision/update/ipc/normal_collisions_build | 2 | 0.002 | 0.002 |
| forward/collision/diagnostics | 1 | 0.002 | 0.001 |
| forward/pncg/trace/write | 28 | 0.002 | 0.002 |
| forward/collision/diagnostics/collision/fun | 1 | 0.001 | 0.001 |
| forward/collision/max_step_size/ipc/ccd | 1 | 0.001 | 0.001 |

## Interpretation

Residual and energy are sampled only at accepted trace states. The plotted trace shows the actual recorded behavior, including any residual decrease followed by stall. It excludes an outer inverse update and adjoint, and makes no universal optimizer or end-to-end speed claim. Startup/JIT prewarm remains outside the forward timer; sparse initial construction is charged inside its first Newton phase. See [evidence JSON](../data/hybrid-first-profile-report-006/evidence.json) for hashes, exact receipts, and call counts.

## Reproduce and audit

Exact executed sources, protocol, timing tree, accepted-state trace, endpoint, and runtime-binding receipt are in `data/hybrid-first-profile-003`. The runtime binding verifies historical input arrays and records runtime source differences. This renderer performs no forward iterations.

The measured run used:

```bash
.venv/bin/python -u src/56-profile-hybrid.py --output-dir data/hybrid-first-profile-003
```

Comet receipt: <https://www.comet.com/liblaf/apple/391adab61c844afb9ffc4e395c4f4f55>.

The clamped run terminated as `failed: nonfinite PNCG energy; endpoint feasible=False`. The baseline terminated as `failed: Newton iteration budget exhausted; endpoint feasible=True`. These are not two completed solves, so the wall-time delta is not a solver speedup.
