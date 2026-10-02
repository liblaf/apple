# Adaptive IPC stiffness cold Smile test

All arms share the saved Smile activation, jaw, neutral seed, collision coverage, IPCTK/PyTorch lock, and Newton settings. `κ0 = 0.1693 MPa = 0.1 * 1.693 MPa`, where aponeurosis is the stiffest deformable material; frozen skin spans 0.127382-0.257861 MPa. Bones and eyes are rigid obstacles. All three recorded runs retained nonempty contact sets; the fixed run preceded the later empty-contact helper correction, which was not exercised here.

This is a static, quasistatic cold forward solve at the fixed activation from the historical Smile Adam update 16 and a 0 degree jaw. Every arm starts from the same saved prestressed neutral displacement, which is not zero. The energy contains bulk Stable Neo-Hookean fat, muscle, and aponeurosis; heterogeneous plane-stress Stable Neo-Hookean skin with frozen skin tangent prestress; additive muscle active stress; and physical IPC contact against complete bones and fixed eyes. Bulk baseline stress is zero. It contains no inertia, target-fitting, or regularization terms.

| Arm | Success | Wall s | PNCG | Newton | Terminal force | Terminal gap nm | Contact feasible | Inversions | Valid forward | κ events | Final κ MPa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |
| fixed kappa0 | False | 4.465 | 44 | 0 | 0.0008169716383075964 | 9.056944067184324 | False | 6 | False | 0 | 0.1693 |
| adaptive kappa | False | 60.139 | 208 | 100 | 1.3427539644229021e-06 | 75258.4600988565 | True | 12 | False | 7 | 16.93 |
| fixed final kappa | False | 42.894 | 60 | 100 | 1.8272349321633991e-06 | 75502.74935995687 | True | 10 | False | 0 | 16.93 |

A feasible contact gap alone is not a valid forward solve: force tolerance, zero inverted tetrahedra, and the solver success receipt are also required. Failures are time-to-failure observations, not speedups. Adaptive κ changes the physical barrier objective; energy is only connected within constant-κ segments. No inverse solve or adjoint was run.

Recorded terminal physical κ: fixed kappa0: 0.1693 MPa, adaptive kappa: 16.93 MPa, fixed final kappa: 16.93 MPa. These values describe each terminal barrier objective; they do not establish a valid forward result.

Outputs: `data/adaptive-ipc-report-002/adaptive-ipc.png`, `data/adaptive-ipc-report-002/adaptive-ipc.svg`, and `data/adaptive-ipc-report-002/evidence.json`.

## Recorded commands

```bash
# Cold Smile fixed IPC stiffness 0.1 maximum E
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 .venv/bin/python -u src/61-test-adaptive-ipc.py --mode fixed --output-dir data/adaptive-ipc-maxe-fixed-001
```

```bash
# Cold Smile adaptive IPC stiffness epsilon 1e-6 and 0.1 maximum E
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 .venv/bin/python -u src/61-test-adaptive-ipc.py --mode adaptive --output-dir data/adaptive-ipc-maxe-adaptive-001
```

```bash
# Cold Smile fixed final adaptive IPC stiffness control
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 .venv/bin/python -u src/61-test-adaptive-ipc.py --mode fixed --stiffness-multiplier 100 --output-dir data/adaptive-ipc-maxe-final-fixed-001
```

## Comet

- [Cold Smile fixed IPC stiffness 0.1 maximum E](https://www.comet.com/liblaf/apple/fe90129385a1406b85f92705c72db21b)
- [Cold Smile adaptive IPC stiffness epsilon 1e-6 and 0.1 maximum E](https://www.comet.com/liblaf/apple/64bcfd753cde45bfb48146597fc415fa)
- [Cold Smile fixed final adaptive IPC stiffness control](https://www.comet.com/liblaf/apple/1f18199b86704dddb99a162b02fc92ec)
