# Fixed-reference 2 mm / 5 degree face fit

The selected dimensionless loss reached a middle tradeoff between the prior beta .25 and beta 1 fits: it improves normal agreement over beta .25 while retaining more position accuracy and less activation variation than beta 1. Both final branches contain five inverted tetrahedra and keep reducing their own objective, so neither endpoint is mechanically valid or inverse-convergence certified.

The data term is

$$
L_{\mathrm{data}}=\frac{P}{(13.236093032531715\ \mathrm{mm})^2}+N,
$$

with `eta * R` added for smoothing (`eta = 0` off and `1.8346203690062914e-05` on). It calibrates 2 mm vector position RMS and 5 degree unit-normal angular error to equal contributions. The selection is in [110-selected-shape-loss.md](110-selected-shape-loss.md) and [loss-config.json](../data/110-shape-loss-config/loss-config.json). At neutral, position and normal contributions are 0.04940857 and 0.02438146; this is the neutral mismatch, not the reference calibration point.

## Audited endpoint results

| Variant | Position RMS (mm) | Normal RMS (deg) | Surface gradient | 5 mm high-pass residual (mm) | R | Motion RMS (mm) | min det(F) | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 mm / 5 deg, smooth off | 1.89880 | 3.64540 | 0.188888 | 0.117531 | 9.00774 | 3.85073 | -0.46936 | 5 |
| 2 mm / 5 deg, smooth on | 1.90368 | 3.65139 | 0.188875 | 0.117715 | 8.37625 | 3.84374 | -0.47706 | 5 |
| L2 only, smooth off | 1.81192 | 10.19831 | 0.238393 | 0.188001 | 5.68312 | 4.15687 | -0.24879 | 2 |
| L2 only, smooth on | 1.81613 | 10.19632 | 0.238253 | 0.187927 | 5.26836 | 4.15047 | -0.25267 | 2 |
| beta .25, smooth off | 1.85738 | 4.49189 | 0.192099 | 0.122836 | 7.60244 | 3.90578 | -0.45330 | 3 |
| beta 1, smooth off | 1.94264 | 2.96979 | 0.187056 | 0.109856 | 11.15967 | 3.80311 | -0.50194 | 7 |

Relative to beta .25, the reference loss is 2.23% worse in position RMS, 18.84% better in normal RMS, 4.32% better in high-pass residual, and 18.48% higher in activation variation without smoothing. Relative to beta 1, it is 2.26% better in position RMS, 22.75% worse in normal RMS, 6.99% worse in high-pass residual, and 19.28% lower in activation variation. Smoothing reduces R by 7.01% versus recovered off, but slightly worsens position, normals, high-pass residual, and min det(F), without reducing inversions.

Against L2 only without smoothing, the selected off branch has 4.79% worse position RMS, 64.25% better normal RMS, and 37.48% lower high-pass residual, but more activation variation and five rather than two inversions.

At the shared latest inversion-free checkpoint, update 20, all rows below have zero inversions. Positive det(F) alone does not certify stability.

| Variant | Position RMS (mm) | Normal RMS (deg) | High-pass residual (mm) | R | min det(F) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2 only, off | 4.05257 | 8.93451 | 0.346292 | 0.347822 | 0.52870 |
| beta .25, off | 4.01036 | 7.00923 | 0.352453 | 0.498788 | 0.42316 |
| beta 1, off | 3.90576 | 5.29208 | 0.352719 | 1.233985 | 0.02854 |
| 2 mm / 5 deg, off | 3.97749 | 6.12349 | 0.353122 | 0.733639 | 0.27172 |
| 2 mm / 5 deg, on | 3.97766 | 6.12384 | 0.353144 | 0.730416 | 0.27189 |

Both reference branches first invert at update 28. At update 200, off/on objective changes over the final ten updates are -2.8822%/-2.7517%, and over 25 updates are -7.3569%/-7.0741%. This fixed budget did not establish inverse convergence.

Endpoint contributions are position 0.00685988 and normal 0.00401753 (off), and position 0.00689524, normal 0.00403076, and regularizer 0.00015367 (on). Do not rank objective values against old beta runs: their scalings differ. Compare the unnormalized metrics above.

## Recovery and validation

The original `130-reference-fit` smoothness-off process stopped unexpectedly after accepted update 102. It recorded no solver failure or graceful shutdown; the cause is unknown. It is preserved as incomplete, not an endpoint. The recovery protocol is [131-reference-resume-protocol.md](131-reference-resume-protocol.md); the original frozen setup remains [115-reference-fit-protocol.md](115-reference-fit-protocol.md).

`132-reference-continuation` copies the neutral-origin off history through update 102, then resumes saved q, Adam moments, and counter. Re-evaluation uses finite replay tolerances, so it is not a bitwise-continuous trajectory. The on branch is an independent fresh-neutral run. The recovery service exited 0 after Cherries shutdown; its [Comet run](https://www.comet.com/liblaf/apple/ce761830b9d14a0a9339ac23b8ef2708) and [completion receipt](../data/132-reference-continuation/service-completion.json) record the lifecycle.

The independent final audit [passed](../data/135-verification/checks.json): 100 current and archived numerical source records and three fixture receipts matched; each branch has 201 accepted solver receipts and ended at update 200 with `completed_budget_not_convergence_certified`. The recovered prefix trace and receipts 0--102 matched exactly, finite replay passed, and step-103 Adam activation maximum absolute error was 2.22e-16. CPU endpoint recomputation maximum metric errors were at most 1.78e-15 off and 4.00e-15 on. The [audit Comet record](https://www.comet.com/liblaf/apple/16bb4081d5b04cb286db63ad4c7ea64c) shut down fully. A first audit launch stopped before validation because an assertion expected the old fresh-neutral text in `protocol["start"]`; its preserved `135-verification-start-assumption-error` artifacts document the corrected audit startup and do not indicate a physics or optimization failure.

## Outputs and reproducibility

The ten-curve history comparison is [reference-loss-histories.png](../data/136-analysis/reference-loss-histories.png), with structured values in [analysis.json](../data/136-analysis/analysis.json). The audited [loss-component history](../data/138-loss-components/loss-components-full.png) and [last-50-update view](../data/138-loss-components/loss-components-last50.png) separate the dimensionless objective from its position, normal, and regularizer contributions. Their [Comet record](https://www.comet.com/liblaf/apple/223fef8e5c5144389be76019a03d3aae) and the comparison [Comet run](https://www.comet.com/liblaf/apple/2cb28a6d0735433292293ee9c5ed3b78) completed after Cherries shutdown.

The endpoint [full face](../data/140-reference-figures/full-comparison.png), [mouth](../data/140-reference-figures/mouth-comparison.png), and [position-error map](../data/140-reference-figures/position-error-maps.png) use exact geometry fixtures; the full and mouth plates are grayscale, while the position-error map uses a fixed 10 mm color scale. The matching inversion-free update-20 [full face](../data/141-reference-figures-noninverted/full-comparison.png), [mouth](../data/141-reference-figures-noninverted/mouth-comparison.png), and [error map](../data/141-reference-figures-noninverted/position-error-maps.png) follow the same rendering convention. Their [endpoint Comet run](https://www.comet.com/liblaf/apple/6f693ba4e9c74054890903f896dbd2b6) and [update-20 Comet run](https://www.comet.com/liblaf/apple/472c8cf8672f4882bf3aeed5c03f45fa) completed after shutdown.

The comparison command was:

```bash
env CHERRIES_NAME='Face reference loss: audited comparison' CHERRIES_TAGS='face,3d,raw6,reference-loss,resume,analysis' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/136-analyze-reference.py
```

The corresponding renderer commands were:

```bash
env CHERRIES_NAME='Face reference loss: endpoint comparison renders' CHERRIES_TAGS='face,3d,raw6,reference-loss,render,comparison' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/140-render-reference.py --shared-step 200 --error-limit-mm 10
env CHERRIES_NAME='Face reference loss: common inversion-free comparison renders' CHERRIES_TAGS='face,3d,raw6,reference-loss,render,noninverted' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/140-render-reference.py --output 141-reference-figures-noninverted --shared-step 20 --error-limit-mm 10
```

See [135-verification.log](../logs/135-verification.log) for the exact final-audit invocation.

The loss-component command was:

```bash
env CHERRIES_NAME='Reference-length face loss components' CHERRIES_TAGS='face,3d,raw6,reference-loss,loss-components,visualization' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/138-plot-reference-loss.py --comparison-dir 132-reference-continuation --verification 135-verification/checks.json --output 138-loss-components
```

The beta .25 and beta 1 controls are continuous fresh-neutral runs. The beta 0 and beta .05 controls were paused at update 100 then continued with saved Adam state, so they are not uninterrupted controls. Numerical consistency does not establish mechanical stability, absence of inversion, or inverse-optimization convergence.
