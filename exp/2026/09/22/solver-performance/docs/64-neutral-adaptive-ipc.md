# Neutral adaptive IPC stiffness forward test

This is one static quasistatic neutral forward problem. It starts from a repaired geometric contact initializer in the constitutive reference, with no saved equilibrium displacement. The jaw, muscle activation, and bulk additive stress are zero. The energy contains Stable Neo-Hookean fat, muscle, and aponeurosis; heterogeneous plane-stress Stable Neo-Hookean skin with its existing prescribed baseline tension; and physical IPC contact against complete source bones and fixed eyes. It contains no inertia, target fit, regularization, inverse update, or load continuation.

All arms have matching recorded fixture/input hashes, solver/material runtime source hashes, runtime binding proof, collision coverage, and hybrid solver settings. κ0 is 0.1693 MPa (0.1 times aponeurosis E = 1.693 MPa); adaptive κ uses epsilon scale 1e-6 and a 100x cap. The force target is anchored once at κ0, before any fixed-final multiplier: `max(1e-8, 1e-3 * initial force at κ0)` = `5.7641531719690125e-07`.

adaptive kappa reached a valid equilibrium; fixed kappa0, fixed final kappa did not. On this fixture, the stiffness schedule matters: a stronger fixed barrier at the same final κ did not reproduce the valid outcome.

| Arm | Result | Wall s | PNCG / Newton updates | Terminal force | Gap nm | Inversions | Final κ MPa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed kappa0 | failed before force convergence | 15.700 | 110 / 0 | 7.592e-06 | 9.188 | 164 | 0.1693 (0 changes) |
| adaptive kappa | valid equilibrium | 75.616 | 440 / 4 | 5.446e-07 | 72971.711 | 0 | 10.8352 (6 changes) |
| fixed final kappa | force converged; invalid geometry | 68.308 | 380 / 4 | 5.640e-07 | 72859.114 | 14 | 10.8352 (0 changes) |

| Arm | PNCG s | Newton s | CCD leaf work s |
| --- | ---: | ---: | ---: |
| fixed kappa0 | 15.696 | 0.000 | 1.741 |
| adaptive kappa | 56.211 | 19.400 | 6.591 |
| fixed final kappa | 49.231 | 19.073 | 5.663 |

The timing tree synchronizes CUDA at scoped boundaries. PNCG and Newton are inclusive forward-solve phase times; CCD is a summed `ipc/ccd` leaf subset already included in those totals and must not be added again. The separately measured model setup/operator prewarm is excluded; the first sparse construction remains in Newton. Contact feasibility alone is insufficient: valid forward also requires the force gate, zero inverted tetrahedra, and solver success. A failed arm is only a time-to-failure observation, never a speedup. κ changes the physical barrier objective, so energy is connected only within fixed-κ segments. These hybrid timings are not comparable to the separate neutral-newton Newton-only driver.

Outputs: `data/neutral-adaptive-ipc-report-002/neutral-adaptive-ipc.png`, `data/neutral-adaptive-ipc-report-002/neutral-adaptive-ipc.svg`, and `data/neutral-adaptive-ipc-report-002/evidence.json`.

## Recorded commands

```bash
# neutral-adaptive-ipc-fixed-002
.venv/bin/python -u src/63-test-neutral-adaptive-ipc.py --mode fixed --output-dir data/neutral-adaptive-ipc-fixed-002
```

```bash
# neutral-adaptive-ipc-adaptive-002
.venv/bin/python -u src/63-test-neutral-adaptive-ipc.py --mode adaptive --output-dir data/neutral-adaptive-ipc-adaptive-002
```

```bash
# neutral-adaptive-ipc-final-fixed-002
.venv/bin/python -u src/63-test-neutral-adaptive-ipc.py --mode fixed --fixed-stiffness-multiplier 64 --output-dir data/neutral-adaptive-ipc-final-fixed-002
```

## Comet

- [neutral-adaptive-ipc-fixed-002](https://www.comet.com/liblaf/apple/85bed13f2fef46bd9dd3e914dd709eee)
- [neutral-adaptive-ipc-adaptive-002](https://www.comet.com/liblaf/apple/6e12322576014b74b819d791a97a85e0)
- [neutral-adaptive-ipc-final-fixed-002](https://www.comet.com/liblaf/apple/db48c4dc6c8d4a45a14aae05b40055d3)
