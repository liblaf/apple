# Neutral face collision and skin ablations

All four arms start from the same repaired constitutive-reference neutral face, without a saved equilibrium displacement. Jaw rotation, muscle activation, and bulk additive stress are zero. The full arm contains bulk Stable Neo-Hookean tissues, heterogeneous plane-stress skin with its existing baseline tension, and IPC against the complete bones and fixed eyes. The ablations remove the stated potential or contact constraint; they do not represent the same physical forward problem.

Every arm uses the same absolute force target, anchored once from the full κ0 initial gradient: `5.7641531719690125e-07`. Fixture, material, solver, runtime-source, and input-binding provenance hashes match; only the recorded ablation fields and measured timings vary.

All arms reached the shared force threshold. full, no skin are geometrically invalid; no collision, neither are valid only under their configured reduced-PDE gates. The full arm took 66.273 s and has no valid-full-model speed comparison. Its direct exclusive bulk/skin/collision-IPC work was 12.008 / 0.194 / 28.345 s; the no-collision reduced arm took 21.548 s. Skin direct evaluation is small here, but skin removal also removes its baseline load and changed the nonlinear path.

| Arm | Configured PDE result | Wall s | PNCG / Newton | Terminal force | Inversions | Full-contact diagnostic |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| full | force converged; invalid geometry | 66.273 | 318 / 4 | 5.443e-07 | 64 | True |
| no collision | valid configured PDE | 21.548 | 160 / 1 | 2.935e-07 | 0 | False |
| no skin | force converged; invalid geometry | 23.870 | 174 / 0 | 5.424e-07 | 28 | True |
| neither | valid configured PDE | 0.493 | 18 / 0 | 5.478e-07 | 0 | False |

| Arm | Forward inclusive s | PNCG inclusive s | Newton inclusive s | Bulk exclusive s | Skin exclusive s | Collision/IPC exclusive s | Native CCD inclusive subset s | Hessian cache inclusive subset s | FEM / CSR constructor inclusive s | PCG inclusive subset s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 66.273 | 41.660 | 24.609 | 12.008 | 0.194 | 28.345 | 4.780 | 24.171 | 19.498 / 4.262 | 0.058 |
| no collision | 21.548 | 2.449 | 19.099 | 2.150 | 0.043 | 0.000 | 0.000 | 19.036 | 15.072 / 3.963 | 0.035 |
| no skin | 23.870 | 23.863 | 0.000 | 6.840 | 0.000 | 16.174 | 2.705 | 0.000 | 0.000 / 0.000 | 0.000 |
| neither | 0.493 | 0.492 | 0.000 | 0.437 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 | 0.000 |

Bulk, skin, and collision/IPC values sum exclusive scopes, avoiding nested-wrapper double counting. Collision/IPC includes broad phase, candidate preparation, and native IPC work. Native CCD is reported separately as an inclusive `ipc/ccd` subset of that pipeline; it must not be added to collision/IPC, PNCG, or Newton totals. Hessian-cache preparation, the first FEM/CSR constructors, and PCG are inclusive phase subsets. Startup/model construction and the ordinary prewarm are excluded; first sparse construction is included. Collision-disabled intersections remain a full-contact diagnostic, not a reduced-PDE convergence gate. Removing skin eliminates its prescribed baseline load, so its timing change is a physical reduction as well as removed computation.

Reproducibility note, not a timing arm: the archived prior adaptive full run `data/neutral-adaptive-ipc-adaptive-002` had matching recorded inputs and parameters. Its initial force was bit-identical; initial energy differed by 1.3e-22 and PNCG step-1 force by 5.42e-19. The first limiter decision diverged at step 6, after which κ schedule and terminal volume differed. Extra gradient/prewarm ordering and synchronized component hooks are possible perturbations; causal attribution has not been tested.

Outputs: `exp/2026/09/22/solver-performance/data/neutral-ablations-report-001/neutral-ablations.png`, `exp/2026/09/22/solver-performance/data/neutral-ablations-report-001/neutral-ablations.svg`, and `exp/2026/09/22/solver-performance/data/neutral-ablations-report-001/evidence.json`.

## Recorded commands

```bash
# neutral-ablation-full-001
.venv/bin/python -u src/65-test-neutral-ablations.py --case full --output-dir data/neutral-ablation-full-001
```

```bash
# neutral-ablation-no-collision-001
.venv/bin/python -u src/65-test-neutral-ablations.py --case no_collision --output-dir data/neutral-ablation-no-collision-001
```

```bash
# neutral-ablation-no-skin-001
.venv/bin/python -u src/65-test-neutral-ablations.py --case no_skin --output-dir data/neutral-ablation-no-skin-001
```

```bash
# neutral-ablation-neither-001
.venv/bin/python -u src/65-test-neutral-ablations.py --case neither --output-dir data/neutral-ablation-neither-001
```

## Comet

- [neutral-ablation-full-001](https://www.comet.com/liblaf/apple/c17eb73c0e0d45a0a387a39a218d3727)
- [neutral-ablation-no-collision-001](https://www.comet.com/liblaf/apple/108a473750ae4d1d8f24747358e8d843)
- [neutral-ablation-no-skin-001](https://www.comet.com/liblaf/apple/bd0859fd493f4c83baa28210e18512b9)
- [neutral-ablation-neither-001](https://www.comet.com/liblaf/apple/69fb0add465544018cb14d344a3960a3)
