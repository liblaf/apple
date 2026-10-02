# Hybrid-first cold forward profile

The instrumented forward solve took 49.935 s and did not converge; terminal force was 1.75721e-06 against 1.0e-08. This is one cold fixture profile and does not establish a general solver speedup.

PNCG completed 11 updates, then handed off for nonpositive curvature. This case did not reach the first 20-step median window, so it does not validate that trigger on the full face.

Hessian construction dominates: 38.844 s for FEM/CSR plus IPC Hessian assembly, including 25.136 s in the two first-construction scopes. Later numeric refreshes remain a cost; the constructor cost can be amortized only when its structures are retained.

Newton completed 100 iterations with 3501 successful CG iterations and 100 rejected unshifted attempts. All CG attempts together cost 2.838 s; CCD wrappers across the forward solve cost 2.618 s. These are nested timings, not extra costs to add to the phase totals.

Each Newton iteration accepted the first positive shift, equal to that state's mean Hessian diagonal. All 100 accepted Newton steps had full alpha, no CCD restriction, and no backtracking. Cheap CG under this regularization did not imply fast force convergence; shift selection is the next convergence issue to investigate, alongside Hessian setup reuse.

The independent saved-endpoint check found 0 inverted tetrahedra and contact feasibility True. The sparse/reference HVP relative error at handoff was 3.08e-16; this checks that state, not every later Hessian.

![Residual, energy, and additive Newton timing](../data/hybrid-first-profile-report-004/57-hybrid-profile.png)

## Protocol

The run starts from the shared neutral displacement with saved Smile activation and fixed prestress on a local RTX 4090, with no competing compute job observed. It runs undamped PNCG with Gauss-Newton contact curvature first, then exact GPU free-CSR Newton with an absolute exact Jacobi diagonal. IPC is PyPI 1.6.0; model construction and ordinary-operator prewarm are excluded, while the first sparse Newton construction and its assembly-kernel JIT are included. The log records about 2.38 s compiling those two assembly kernels. The profiler adds synchronization and changes overlap; values are instrumented completed-operation wall times.

| Timing | Seconds | Interpretation |
| --- | --- | --- |
| Model construction | 10.276 | outside forward timing |
| Operator prewarm/JIT | 0.323 | excluded from forward timing |
| Forward | 49.934 | inclusive phase; do not add with children |
| PNCG | 1.786 | inclusive phase; do not add with children |
| Newton | 48.033 | inclusive phase; do not add with children |

Parent phase timings are inclusive explanatory values and must not be added to their children. The Newton allocation below uses exclusive nodes only, so it sums to the Newton parent without nested double counting.

| PNCG exclusive allocation | Seconds |
| --- | --- |
| Phase controller/other | 0.069 |
| Energy evaluation | 0.212 |
| Trace/validation | 0.001 |
| PNCG diagonal | 0.207 |
| PNCG directional curvature | 0.614 |
| CCD | 0.321 |
| Contact/state update | 0.171 |
| Force evaluation | 0.192 |

| Newton exclusive allocation | Seconds |
| --- | --- |
| Phase controller/other | 0.272 |
| Energy evaluation | 1.138 |
| Hessian diagonal | 0.008 |
| Cache preparation overhead | 0.136 |
| IPC Hessian assembly | 0.843 |
| Hessian assembly/cache | 38.001 |
| Trace/validation | 0.965 |
| PCG control/vector work | 0.699 |
| Sparse SpMV | 2.123 |
| CCD | 2.284 |
| Contact/state update | 1.564 |

## Calls

| Scope | Calls | Inclusive s | Exclusive s |
| --- | --- | --- | --- |
| forward | 1 | 49.934 | 0.033 |
| forward/newton | 1 | 48.033 | 0.092 |
| forward/newton/newton/step | 100 | 47.012 | 0.180 |
| forward/newton/newton/step/hessian/diagonal | 100 | 38.972 | 0.008 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare | 100 | 38.964 | 0.120 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/fem_constructor | 1 | 20.760 | 18.098 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/csr_refresh | 99 | 12.865 | 1.507 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/csr_refresh/hessian/fem_numeric | 99 | 11.358 | 11.358 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/csr_constructor | 1 | 4.376 | 0.063 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/csr_constructor/hessian/csr_refresh | 1 | 4.313 | 4.199 |
| forward/newton/newton/step/pcg | 200 | 2.838 | 0.699 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/fem_constructor/hessian/fem_numeric | 1 | 2.662 | 2.662 |
| forward/newton/newton/step/model/max_step_size | 100 | 2.284 | 0.002 |
| forward/newton/newton/step/model/max_step_size/collision/max_step_size | 100 | 2.282 | 0.068 |
| forward/newton/newton/step/pcg/hessian/spmv | 4829 | 2.139 | 2.123 |
| forward/pncg | 1 | 1.786 | 0.069 |
| forward/newton/newton/step/model/update | 100 | 1.564 | 0.005 |
| forward/newton/newton/step/model/update/collision/update | 100 | 1.560 | 0.040 |
| forward/newton/newton/step/model/max_step_size/collision/max_step_size/ipc/candidates_build | 100 | 1.452 | 1.452 |
| forward/newton/newton/step/model/update/collision/update/ipc/candidates_build | 100 | 1.391 | 1.391 |
| forward/newton/newton/step/model/fun | 200 | 1.138 | 0.011 |
| forward/newton/newton/step/model/fun/warp_model/fun | 200 | 0.984 | 0.984 |
| forward/newton/trace/newton_state | 100 | 0.930 | 0.069 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/ipc/barrier_hessian | 100 | 0.843 | 0.843 |
| forward/newton/newton/step/model/max_step_size/collision/max_step_size/ipc/ccd | 100 | 0.762 | 0.762 |
| forward/pncg/model/hess_quad | 12 | 0.614 | 0.180 |
| forward/newton/trace/newton_state/model/fun | 100 | 0.558 | 0.005 |
| forward/newton/trace/newton_state/model/fun/warp_model/fun | 100 | 0.483 | 0.483 |
| forward/pncg/model/hess_quad/collision/hess_quad | 12 | 0.434 | 0.019 |
| forward/pncg/model/hess_quad/collision/hess_quad/ipc/gauss_newton_hessian_quadratic_form | 12 | 0.415 | 0.415 |
| forward/pncg/model/max_step_size | 11 | 0.321 | 0.000 |
| forward/pncg/model/max_step_size/collision/max_step_size | 11 | 0.321 | 0.021 |
| forward/newton/trace/newton_state/model/grad | 100 | 0.288 | 0.005 |
| forward/pncg/model/fun | 12 | 0.212 | 0.001 |
| forward/pncg/model/hess_diag | 12 | 0.207 | 0.001 |
| forward/pncg/model/fun/warp_model/fun | 12 | 0.198 | 0.198 |
| forward/pncg/model/grad | 11 | 0.192 | 0.001 |
| forward/pncg/model/hess_diag/warp_model/hess_diag | 12 | 0.177 | 0.177 |
| forward/pncg/model/update | 11 | 0.171 | 0.001 |
| forward/pncg/model/update/collision/update | 11 | 0.170 | 0.006 |
| forward/pncg/model/grad/warp_model/grad | 11 | 0.167 | 0.167 |
| forward/pncg/model/max_step_size/collision/max_step_size/ipc/candidates_build | 11 | 0.157 | 0.157 |
| forward/newton/trace/newton_state/model/grad/collision/grad | 100 | 0.155 | 0.155 |
| forward/pncg/model/update/collision/update/ipc/candidates_build | 11 | 0.150 | 0.150 |
| forward/pncg/model/max_step_size/collision/max_step_size/ipc/ccd | 11 | 0.143 | 0.143 |
| forward/newton/newton/step/model/fun/collision/fun | 200 | 0.142 | 0.142 |
| forward/newton/newton/step/model/update/collision/update/ipc/normal_collisions_build | 100 | 0.129 | 0.129 |
| forward/newton/trace/newton_state/model/grad/warp_model/grad | 100 | 0.128 | 0.128 |
| forward/newton/newton/step/hessian/diagonal/hessian/cache_prepare/hessian/csr_constructor/hessian/csr_refresh/hessian/fem_numeric | 1 | 0.114 | 0.114 |
| forward/newton/trace/newton_state/model/fun/collision/fun | 100 | 0.070 | 0.070 |
| forward/model/grad | 1 | 0.036 | 0.001 |
| forward/newton/newton/step/validation/handoff_hvp | 1 | 0.035 | 0.011 |
| forward/model/grad/warp_model/grad | 1 | 0.032 | 0.032 |
| forward/pncg/model/hess_diag/collision/hess_diag | 12 | 0.029 | 0.018 |
| forward/collision/state_at | 2 | 0.029 | 0.000 |
| forward/collision/state_at/collision/update | 2 | 0.029 | 0.002 |
| forward/collision/state_at/collision/update/ipc/candidates_build | 2 | 0.025 | 0.025 |
| forward/pncg/model/grad/collision/grad | 11 | 0.024 | 0.024 |
| forward/newton/newton/step/validation/handoff_hvp/hessian/spmv | 1 | 0.021 | 0.021 |
| forward/newton/newton/step/pcg/hessian/spmv/hessian/cache_prepare | 4829 | 0.016 | 0.016 |
| forward/collision/max_step_size | 1 | 0.015 | 0.002 |
| forward/newton/trace/newton_state/trace/write | 100 | 0.014 | 0.014 |
| forward/pncg/model/update/collision/update/ipc/normal_collisions_build | 11 | 0.014 | 0.014 |
| forward/pncg/model/fun/collision/fun | 12 | 0.013 | 0.013 |
| forward/collision/max_step_size/ipc/candidates_build | 1 | 0.012 | 0.012 |
| forward/pncg/model/hess_diag/collision/hess_diag/ipc/gauss_newton_hessian_diagonal | 12 | 0.012 | 0.012 |
| forward/newton/newton/step/validation/handoff_hvp/model/hess_prod | 1 | 0.004 | 0.000 |
| forward/newton/newton/step/validation/handoff_hvp/model/hess_prod/warp_model/hess_prod | 1 | 0.003 | 0.003 |
| forward/collision/diagnostics | 1 | 0.003 | 0.001 |
| forward/model/grad/collision/grad | 1 | 0.003 | 0.003 |
| forward/collision/state_at/collision/update/ipc/normal_collisions_build | 2 | 0.002 | 0.002 |
| forward/collision/diagnostics/collision/fun | 1 | 0.002 | 0.002 |
| forward/collision/max_step_size/ipc/ccd | 1 | 0.001 | 0.001 |
| forward/newton/newton/step/validation/handoff_hvp/model/hess_prod/collision/hess_prod | 1 | 0.001 | 0.001 |
| forward/pncg/trace/write | 12 | 0.001 | 0.001 |
| forward/newton/newton/step/validation/handoff_hvp/hessian/spmv/hessian/cache_prepare | 1 | 0.000 | 0.000 |

## Interpretation

Residual and energy are sampled only at accepted trace states. The plotted trace shows the actual recorded behavior, including any residual decrease followed by stall. It excludes an outer inverse update and adjoint, and makes no universal optimizer or end-to-end speed claim. Startup/JIT prewarm remains outside the forward timer; sparse initial construction is charged inside its first Newton phase. See [evidence JSON](../data/hybrid-first-profile-report-004/evidence.json) for hashes, exact receipts, and call counts.

## Reproduce and audit

Working directory: `exp/2026/09/22/solver-performance`. The measured command was:

```bash
CHERRIES_NAME='Undamped PNCG to sparse Newton cold Smile profile' \
CHERRIES_TAGS='smile,hybrid,pncg,newton,profile,pypi' \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
.venv/bin/python -u src/56-profile-hybrid.py \
  --output-dir data/hybrid-first-profile-002
```

Use a fresh output directory for another solve. Exact executed sources, protocol, timing tree, accepted-state trace, endpoint, and runtime-binding diffs are in `data/hybrid-first-profile-002`. The runtime binding verifies every historical state array and preserves the original manifests. It records the solver import migration and official-PyPI lockfile rather than pretending the old runtime hashes still match.

The measured process preserved its complete timing and endpoint before a post-run tissue-versus-collision-node shape error. The saved endpoint was then validated using `src/56-profile-hybrid.py --output-dir data/hybrid-first-profile-002 --finalize-only true`; **no solver iterations were rerun**, and that validation is outside the reported 49.934 s. Its recomputed force matches the terminal trace. The completed Cherries report logs these recovered artifacts; original solver Comet run: <https://www.comet.com/liblaf/apple/5942077fc75343c4a2689616929fe01e>.
