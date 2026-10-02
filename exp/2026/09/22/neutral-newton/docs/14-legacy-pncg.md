# Legacy PNCG neutral forward

Ran one legacy PNCG forward from the same repaired constitutive-reference seed as the current Newton comparison. It saved **200 accepted steps** in **44.171 s**. Status: **forward_did_not_converge**.

## Result

| Solver | Accepted steps | Loop seconds | Final gradient norm | Final energy | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy PNCG | 200 | 44.171 | 4.009571e-06 | -2.659326e-08 | 24 |
| Newton-CG, GPU contact | 200 | 41.975 | 5.102715e-07 | -4.043218e-08 | 68 |

PNCG's final gradient is **73.58×** the shared stopping threshold **5.449619e-08**. Soft-rigid contact feasible: **True**. Minimum det(F): **-12.2449**. Termination: `max_steps_reached`; receipt: `{'reason': 'PNCG iteration budget exhausted', 'result': 'max_steps_reached'}`.

![Energy and gradient comparison](../data/pncg-review-002/energy-gradient-comparison.png)

Iteration budgets are equal but work per iteration differs. The time-axis curves provide additional context; these single runs do not establish a universal solver ranking or equal-accuracy speedup. Timers include trace logging/checkpoint writes and exclude model construction, preflight and terminal geometry audits. Energy uses MPa m³ and gradient norm uses MPa m²; multiply by 1e6 for J and N.

## Solver and model

This uses the old strict Dai-Kou+ PNCG implementation through `AcceptedForcePncg`, not the experimental adaptive-PNCG method. Its legacy eye-neutral settings are Armijo 0.25, 0.5 mm coordinate cap, up to 60 halvings (61 trials), initial damping 0.001 with bounds 1e-6–1, conjugacy restart interval 200, and CCD safety 0.95. PNCG computes one physical Hessian quadratic form per step and applies adaptive damping to its search model. It does not invoke inner PCG or Newton shift escalation.

Materials, zero activation, prescribed neutral jaw pose, constitutive reference, repaired initial coordinates, exact physical derivative policy, and bone/eye collision geometry match the current Newton run. The current exact diagonal and GPU contact HVP implementation are retained. Thus this is the legacy solver on the current model, not a replay of the older clamped-derivative backend. Newton and PNCG retain their respective line-search/damping/CCD settings, explicitly recorded in the protocols.

The force criterion is evaluated at accepted states, with both primary and secondary thresholds set to max(1e-8, 1e-3 times the original initial gradient). Strict line-search failure stops the solve; the runner restores the last accepted state before saving the terminal artifact. No alternative solver or recovery solve is invoked. Existing selected-neutral assets and the Newton driver are unchanged.

Accepted PNCG line-search trials: **759**, including **559** backtracks. Largest accepted coordinate displacement: **0.5000 mm**. Operation counts: `{'grad_requests': 601, 'fun': 960, 'hess_diag': 200, 'hess_quad': 200, 'hess_prod': 200, 'ccd': 200, 'update': 759, 'grad_evaluations': 200}`.

The objective recorded **559 contact-infeasible trial rejections**. Accepted energy decreases throughout, but the force norm has pronounced spikes and plateaus. Fewer inverted elements than Newton does not imply better volume geometry: PNCG's minimum det(F) is -12.245, versus -1.585 for Newton. Neither endpoint passes the force/geometry gate.

## Command and evidence

Working directory: `exp/2026/09/22/neutral-newton`.

```sh
CHERRIES_NAME='Neutral reference legacy PNCG 200' \
CHERRIES_TAGS='neutral,pncg,legacy-solver,200-iterations' \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYVISTA_OFF_SCREEN=true \
.venv/bin/python src/70-run-pncg-neutral.py
```

[Comet run](https://www.comet.com/liblaf/apple/c92d0d9dbaa1439f98004277e457f884); [terminal receipt](../data/forward-pncg-200-001/summary.json); [protocol](../data/forward-pncg-200-001/protocol.json); [trace](../data/forward-pncg-200-001/trace.jsonl); [terminal displacement](../data/forward-pncg-200-001/terminal.npz); [source/input provenance](../data/forward-pncg-200-001/provenance.json); [stdout](../data/forward-pncg-200-001.stdout.log).

The initial launch stopped at configuration validation before any model/solver work because of a dynamic-module annotation lookup. The module registration was corrected and the Config smoke check passed; that log is retained as `forward-pncg-config-failure.stdout.log`. The actual forward stores all source hashes and the unchanged runtime binding. A nonzero forward exit denotes the saved force/geometry gate failure, not missing artifacts. The report script runs no physics solves.
