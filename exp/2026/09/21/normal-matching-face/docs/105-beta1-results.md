# Beta-1 normal loss on the 3D face

Beta 1 reduces normal RMS by about 34% relative to beta .25 after 200 Adam updates, while position RMS increases by about 4.6%. Both new endpoints contain seven inverted tetrahedra, versus three at beta .25, and their objectives are still decreasing. These are completed experiments, not converged or physically valid solutions.

The two new runs start independently from neutral with unrestricted Raw6 activation, with activation smoothing off and on. The objective is `L2 + beta*(L20/N0)*N + lambda*R`; beta 1 gives equal position and normal loss values at neutral, not equal gradients or equal final contributions. Lambda is zero or 0.003214147722027223. Adam uses learning rate .3, epsilon .01, and betas .9/.999. All other numerical choices are unchanged; see the [frozen protocol](80-beta1-protocol.md) and [derivative and smoke validation](88-beta1-validation.md).

Every comparison began from neutral. The beta-0/.05 controls are [audited continuations](../data/40-continuation/protocol.json), paused at update 100 and resumed with saved Adam state under finite solver tolerances. Beta .25 and 1 ran continuously from neutral through update 200. We do not assume the pause/replay is bitwise equivalent to an uninterrupted trajectory.

All values below are measured at update 200. Position RMS is the reference-area-weighted vector position error; normal RMS is the angle between corresponding oriented triangle normals. HP is the target-relative 5 mm high-pass normal residual over the primary region union. Lower values mean smaller errors.

| Smoothing | Beta | Position RMS (mm) | Normal RMS (deg) | HP RMS (mm) |
| --- | ---: | ---: | ---: | ---: |
| Off | 0 | 1.812 | 10.198 | 0.1880 |
| Off | .05 | 1.800 | 7.188 | 0.1473 |
| Off | .25 | 1.857 | 4.492 | 0.1228 |
| Off | 1 | 1.943 | 2.970 | 0.1099 |
| On | 0 | 1.816 | 10.196 | 0.1879 |
| On | .05 | 1.806 | 7.179 | 0.1470 |
| On | .25 | 1.862 | 4.498 | 0.1225 |
| On | 1 | 1.948 | 2.974 | 0.1131 |

Activation variation R measures changes in the activation tensor between adjacent tetrahedra in the same muscle; it is distinct from surface roughness. Gradient RMS measures the target-relative surface displacement gradient. Motion RMS measures displacement from neutral. Physical validity uses `J = det(F)`, not the determinant of the activation tensor.

| Smoothing | Beta | Gradient RMS | R | Motion RMS (mm) | Minimum J | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 0 | 0.2384 | 5.683 | 4.157 | -0.2488 | 2 |
| Off | .05 | 0.2097 | 6.020 | 4.045 | -0.3601 | 2 |
| Off | .25 | 0.1921 | 7.602 | 3.906 | -0.4533 | 3 |
| Off | 1 | 0.1871 | 11.160 | 3.803 | -0.5019 | 7 |
| On | 0 | 0.2383 | 5.268 | 4.150 | -0.2527 | 2 |
| On | .05 | 0.2098 | 5.587 | 4.038 | -0.3664 | 2 |
| On | .25 | 0.1922 | 7.067 | 3.899 | -0.4604 | 3 |
| On | 1 | 0.1871 | 10.371 | 3.796 | -0.5095 | 7 |

Paired against beta .25, beta 1 changes position RMS by +4.590% off / +4.627% on, normal RMS by -33.885% / -33.885%, surface-gradient RMS by -2.625% / -2.650%, and HP RMS by -10.567% / -7.656%. Activation variation increases by 46.791% / 46.749%. Against beta 0, position RMS is about 7.2–7.3% worse, normal RMS about 70.8–70.9% better, and HP RMS about 39.8–41.6% better. These endpoint tradeoffs include inverted volume elements.

Within beta 1, switching smoothing on reduces R by 7.064%, but raises position RMS by 0.288%, normal RMS by 0.132%, and HP RMS by 2.956%. The inversion count remains seven. A smoother activation field therefore does not produce a better surface residual in this test.

For beta 1, the objective falls another 2.735% off / 2.617% on over updates 190–200 and 6.985% / 6.702% over updates 175–200. Physical gradient norms remain 0.116575 / 0.140788 of their neutral values; they are 1.123529 / 1.108958 times their update-100 values. The budget ended without a convergence certificate. Loss curves compare within-run progress; objective magnitudes across different beta values are not a common quality score.

The first inversion appears at update 21 in both beta-1 branches, compared with 37 at beta .25, 60 at beta .05, and 81 for L2 alone. Beta 1 reaches seven inversions at updates 182 off / 183 on. The latest common inversion-free trace and saved checkpoint across all eight cases is update 20. At that checkpoint beta 1 has position RMS 3.905757 / 3.905932 mm, normal RMS 5.292081 / 5.292297 degrees, HP RMS .352719 / .352751 mm, and minimum J .028540 / .028796. Zero inversions at this nearly collapsed checkpoint is not evidence of mechanical stability.

The [final CPU audit](../data/95-verification/checks.json) passed for both new runs: 97 current and saved numerical source records, three fixtures, nine preflight records, and 201 accepted forward/adjoint receipts per branch. It independently checked initial conditions, Adam state, endpoint geometry, loss decomposition, and signed volume metrics. The largest reported endpoint recomputation errors are below 3.2e-15. Derivative validation passed the unchanged 2% finite-difference gate; these checks establish implementation consistency, not physical validity or optimization convergence.

- [Eight-case analysis JSON](../data/96-analysis/analysis.json) and [nine-panel loss and metric histories](../data/96-analysis/strong-normal-histories.png).
- Update-200 figures: [full face](../data/100-beta1-figures/full-comparison.png), [mouth close-up](../data/100-beta1-figures/mouth-comparison.png), and [position error](../data/100-beta1-figures/position-error-maps.png).
- Common inversion-free update-20 figures: [full face](../data/101-beta1-figures-noninverted/full-comparison.png), [mouth close-up](../data/101-beta1-figures-noninverted/mouth-comparison.png), and [position error](../data/101-beta1-figures-noninverted/position-error-maps.png).

Both figure sets use matching cameras and flat shading. All position-error maps share a 0–10 mm scale, rounded up from the pooled 99th percentile of 9.871364 mm across the eight predictions at both checkpoints, excluding the target zero-error arrays. Values above 10 mm saturate visually; the RMS table is not clipped. The summaries also record per-plate percentiles including target zeros, which are different statistics. All six comparison plates and the history figure were visually inspected for labels, clipping, checkpoint, and scale consistency.

All validation, fitting, audit, analysis, and render processes exited zero after normal shutdown. The one-step smoke and its CPU audit used DEBUG mode; the full runs used named and tagged Cherries/Comet experiments. Experiment records: [main fits](https://www.comet.com/liblaf/apple/063cf62b20a444f497ddd7ed817c9d65), [derivative validation](https://www.comet.com/liblaf/apple/65f640c85c7a4ae9a247711a00d8d997), [CPU audit](https://www.comet.com/liblaf/apple/87136fbf09e14bd69259fdea7907331e), [analysis](https://www.comet.com/liblaf/apple/1cdb955b1b8340dfb86da3fc0ca99770), [update-200 renders](https://www.comet.com/liblaf/apple/f3119bab4ad642beb41a0a24475ba313), and [update-20 renders](https://www.comet.com/liblaf/apple/9d42f269c6c9424792d5dbba37b3aeda).

Working directory: `exp/2026/09/21/normal-matching-face`. Exact derivative, smoke, and main fitting commands are in the protocol; the smoke audit command is in the validation note. The completed analysis and render commands are reproduced below with their output directories. Use distinct output directories for any new run.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
CHERRIES_NAME=beta1-final-verification CHERRIES_TAGS=normal-matching-face,beta1,verification,cpu uv run python src/85-verify-beta.py --comparison-dir 90-beta1 --output 95-verification
CHERRIES_NAME='Raw6 beta 1: audited beta analysis' CHERRIES_TAGS='face,raw6,normal-matching,beta1,analysis,cpu' .venv/bin/python src/96-analyze-beta.py
CHERRIES_NAME='3D face beta 1: matched 200-update renders' CHERRIES_TAGS='face,3d,raw6,normal-matching,beta-1,render,comparison' .venv/bin/python src/100-render-beta.py --shared-step 200 --error-limit-mm 10
CHERRIES_NAME='3D face beta 1: common inversion-free checkpoint renders' CHERRIES_TAGS='face,3d,raw6,normal-matching,beta-1,render,noninverted' .venv/bin/python src/100-render-beta.py --shared-step 20 --error-limit-mm 10 --output 101-beta1-figures-noninverted
```

Preserved logs include `logs/90-beta1.log`, `logs/85-verify-beta.log`, `logs/96-analyze-beta.log`, `logs/100-beta1-figures.log`, and `logs/101-beta1-figures-noninverted.log`. The frozen runner still carries the original beta-.05 narrative; the [beta-1 protocol copy](../data/90-beta1/beta1-protocol.md), executable config, and [preflight manifest](../data/90-beta1/beta-preflight.json) identify this extension.
