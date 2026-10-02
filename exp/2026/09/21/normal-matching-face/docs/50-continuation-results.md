# Adam continuation through update 200

The four audited Adam continuations reached update 200 with their original
saved moments and counter. Position RMS continued to decrease by about 22.5% from
update 100, while the normal-loss branches improved normal-angle RMS by about
9.2%. Every endpoint has two inverted tetrahedra, so these are recorded
numerical endpoints, not physically valid fits or convergence results.

The [continuation protocol](40-continuation-protocol.md),
[normal/adjoint validation](05-validation.md),
[resume-specific audit](45-continuation-validation.md), and final
[audit receipt](../data/45-verification/checks.json) establish provenance and
execution integrity. The audit verified 98 source snapshots, three fixtures,
the exact inherited histories, 100 new successful solve receipts per branch,
and Adam counter 200; the largest independent endpoint-metric error was
2.6646e-15. It passed with `all_completed=true`.

| Branch | Position RMS: 100 to 200 (mm) | Change | Normal-angle RMS: 100 to 200 (deg) | Change |
| --- | ---: | ---: | ---: | ---: |
| Smooth off, L2 | 2.342 to 1.812 | -0.530 (-22.624%) | 9.861 to 10.198 | +0.337 (+3.419%) |
| Smooth off, L2 + normal | 2.327 to 1.800 | -0.527 (-22.641%) | 7.918 to 7.188 | -0.730 (-9.220%) |
| Smooth on, L2 | 2.344 to 1.816 | -0.527 (-22.507%) | 9.859 to 10.196 | +0.338 (+3.424%) |
| Smooth on, L2 + normal | 2.329 to 1.806 | -0.524 (-22.483%) | 7.919 to 7.179 | -0.739 (-9.338%) |

## Endpoint measurements

All values below are shared measurements at update 200. `min J` is reported as
the raw minimum determinant, without a percent comparison because its negative
denominator would be misleading.

| Branch | Surface-gradient RMS | 5 mm high-pass residual (mm) | R | Motion RMS (mm) | min J | Inverted tets |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Smooth off, L2 | 0.238393 | 0.188001 | 5.683121 | 4.156868 | -0.248794 | 2 |
| Smooth off, L2 + normal | 0.209662 | 0.147335 | 6.019584 | 4.044753 | -0.360137 | 2 |
| Smooth on, L2 | 0.238253 | 0.187927 | 5.268361 | 4.150475 | -0.252674 | 2 |
| Smooth on, L2 + normal | 0.209753 | 0.147049 | 5.586568 | 4.038047 | -0.366359 | 2 |

At the common endpoint, adding the normal term changed the shared measurements
as follows (normal branch minus L2 branch):

| Smoothing | Position RMS | Normal angle | Surface-gradient RMS | High-pass residual | R | Motion RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | -0.011475 mm (-0.633%) | -3.010245 deg (-29.517%) | -0.028731 (-12.052%) | -0.040666 mm (-21.631%) | +0.336464 (+5.920%) | -0.112115 mm (-2.697%) |
| On | -0.010466 mm (-0.576%) | -3.017164 deg (-29.591%) | -0.028499 (-11.962%) | -0.040878 mm (-21.752%) | +0.318207 (+6.040%) | -0.112427 mm (-2.709%) |

Turning smoothing on changed the common measurements as follows (smooth-on
minus smooth-off):

| Data term | Position RMS | Normal angle | Surface-gradient RMS | High-pass residual | R | Motion RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 | +0.004214 mm (+0.233%) | -0.001984 deg (-0.019%) | -0.000140 (-0.059%) | -0.000074 mm (-0.040%) | -0.414759 (-7.298%) | -0.006393 mm (-0.154%) |
| L2 + normal | +0.005224 mm (+0.290%) | -0.008902 deg (-0.124%) | +0.000092 (+0.044%) | -0.000286 mm (-0.194%) | -0.433016 (-7.193%) | -0.006705 mm (-0.166%) |

Thus the normal term consistently improved normal, derivative, and high-pass
measurements while increasing R. Smoothing reduced R but had only sub-percent
effects on the listed data measurements.

## Tail behavior and validity

| Branch | Objective, last 10 | Objective, last 25 | Gradient / neutral | Gradient / update 100 |
| --- | ---: | ---: | ---: | ---: |
| Smooth off, L2 | -3.565% | -9.048% | 0.135283 | 0.541990 |
| Smooth off, L2 + normal | -3.220% | -8.226% | 0.225965 | 0.902598 |
| Smooth on, L2 | -3.426% | -8.729% | 0.139390 | 0.562341 |
| Smooth on, L2 + normal | -3.128% | -7.962% | 0.163307 | 0.657520 |

The objective still decreased over both tail windows, but the fixed 200-update
budget does not prove convergence. The physical-gradient ratios are lower than
neutral yet remain 0.542 to 0.903 of their update-100 values, and all endpoint
states are inverted. The first inversions occurred at update 81 for both L2
branches and update 60 for both normal branches. A second inversion occurred at
186 for smooth-off L2, 187 for smooth-on L2, and 151 for both normal branches.
The latest shared inversion-free trace state was update 59; the latest shared
saved inversion-free geometry was update 50.

## Artifacts and runs

- Original shared update-50 inversion-free figures: [full comparison](../data/31-figures-noninverted/full-comparison.png), [mouth comparison](../data/31-figures-noninverted/mouth-comparison.png), and [position-error maps](../data/31-figures-noninverted/position-error-maps.png).
- Original update-100 figures: [full comparison](../data/30-figures/full-comparison.png), [mouth comparison](../data/30-figures/mouth-comparison.png), and [position-error maps](../data/30-figures/position-error-maps.png).
- New update-200 figures: [full comparison](../data/50-figures/full-comparison.png), [mouth comparison](../data/50-figures/mouth-comparison.png), and [position-error maps](../data/50-figures/position-error-maps.png).
- Full histories with the update-100 boundary: [continuation-histories.png](../data/46-analysis/continuation-histories.png) and machine-readable [analysis.json](../data/46-analysis/analysis.json).
- [Main continuation](https://www.comet.com/liblaf/apple/ad00ed6e78c1407c8d70ee53a642ce45), [final CPU audit](https://www.comet.com/liblaf/apple/d9b8aeacd71b4d0dac3243d9212b487f), [analysis](https://www.comet.com/liblaf/apple/db909e7e21d04af78deaad22ee62e4f5), and [update-200 rendering](https://www.comet.com/liblaf/apple/72bd1ded52aa4a1ea1fede4e41686a2e).

The normal analysis command was:

```bash
CHERRIES_NAME='3D face normal matching: audited Adam continuation analysis' CHERRIES_TAGS='face,raw6,normal-matching,continuation,analysis,cpu' .venv/bin/python src/46-analyze-continuation.py
```

The main continuation and final audit used:

```bash
CHERRIES_NAME='Raw6 face Adam continuation to 200' CHERRIES_TAGS='face,raw6,normal-matching,continuation,adam' .venv/bin/python src/40-continue.py
CHERRIES_NAME='continuation-final-verification' CHERRIES_TAGS='face,raw6,continuation,cpu,audit,final' .venv/bin/python src/45-verify-continuation.py --comparison-dir 40-continuation --output 45-verification
```
