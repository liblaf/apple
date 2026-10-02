# Stronger surface-normal weight: beta 0.25

The fresh-neutral beta-0.25 Raw6 fits completed 200 updates and passed the CPU
audit. They improve normal, surface-gradient, and high-pass measurements over
beta .05, but first invert at update 37 and end with three inverted tetrahedra.
They are numerical endpoints, neither valid nor convergence-certified.

The [protocol](55-strong-normal-protocol.md) increased only the normalized normal
coefficient fivefold, from beta .05 to .25, with smoothing off and on. The
objective remains `L2 + beta*(L20/N0)*N + lambda*R`, with unchanged L2 coefficient,
unrestricted Raw6 activation, materials, target support, and solver tolerances.
Beta .25 gives the normal term 25% of the L2 term's value at neutral; it does
not prescribe their gradient ratio or their later contribution. Adam uses
learning rate .3, epsilon .01 and betas .9/.999. Lambda is zero or
0.003214147722027223. See the [derivative and smoke validation](58-strong-normal-validation.md).

The beta-0 and beta-.05 references also began neutral; they are [audited continuations](../data/40-continuation/protocol.json), paused at update 100 then resumed with saved Adam state. Their finite-tolerance forward/adjoint replay was not bitwise identical. Beta .25 is a fresh neutral start in [`60-strong-normal`](../data/60-strong-normal), running continuously through 200. The [audit receipt](../data/65-verification/checks.json) passed with `all_completed=true`, matching 97 source, three fixture, and seven preflight records with 201 accepted receipts per new branch. Endpoint metrics independently recomputed on CPU differ by at most 5.33e-15. Another experiment shared the GPU, so wall time is not compared.

All rows below are update 200. Position, normal-angle, surface-gradient and
motion measurements are RMS values. High-pass is the target-relative 5 mm
normal residual on the predefined primary region union; R measures activation
tensor variation, not surface roughness.

| Smoothing | Beta | Position RMS mm | Normal RMS deg | Surface-gradient RMS | High-pass RMS mm | R | Motion RMS mm | min det(F) | Inverted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Off | 0 | 1.811919 | 10.198309 | 0.238393 | 0.188001 | 5.683121 | 4.156868 | -0.248794 | 2 |
| Off | .05 | 1.800444 | 7.188063 | 0.209662 | 0.147335 | 6.019584 | 4.044753 | -0.360137 | 2 |
| Off | .25 | 1.857380 | 4.491892 | 0.192099 | 0.122836 | 7.602443 | 3.905776 | -0.453304 | 3 |
| On | 0 | 1.816133 | 10.196325 | 0.238253 | 0.187927 | 5.268361 | 4.150475 | -0.252674 | 2 |
| On | .05 | 1.805667 | 7.179161 | 0.209753 | 0.147049 | 5.586568 | 4.038047 | -0.366359 | 2 |
| On | .25 | 1.862078 | 4.497780 | 0.192176 | 0.122480 | 7.067394 | 3.899171 | -0.460415 | 3 |

At beta .25 versus .05, smooth-off/on respectively change position by +3.162%/+3.124%, normal angle by -37.509%/-37.350%, surface-gradient by -8.377%/-8.380%, high-pass residual by -16.628%/-16.708%, R by +26.295%/+26.507%, and motion by -3.436%/-3.439%. Raw minimum determinants fall and inversion count rises by one, so signed-det percentages are intentionally omitted.

Compared with L2 alone, beta .25 reduces normal-angle error by 55.95%/55.89%
and high-pass residual by 34.66%/34.83%, at 2.51%/2.53% higher position RMS
(off/on). Within beta .25, turning smoothing on reduces R by 7.04%, while
position RMS rises 0.253%, normal RMS rises 0.131%, and high-pass residual falls
0.290%. The weak activation regularizer has little effect on the fitted surface
and does not remove inversions.

The last-10 own-objective changes are -2.999805% off and -2.908964% on; physical-gradient ratios to neutral are 0.183169 and 0.168020. Those within-branch diagnostics do not prove convergence: the budget is fixed and every endpoint is inverted. First inversions occur at 37 for beta .25, versus 60 for beta .05 and 81 for beta 0; second inversions occur at 105/106, then third at 139 for both stronger branches. The latest common inversion-free trace is update 36 and common saved state is 30.

At update 36, stronger normals already improve normal RMS from 8.324 to
6.327 degrees and position RMS from 3.433–3.434 to 3.403–3.404 mm across the two
smoothing settings.
However, the high-pass residual is slightly worse (about .2960 versus .2939 mm),
and the stronger branches' minimum det(F) values are only .00124/.00160.
Zero inversions at that step do not establish a mechanically stable fit.
At the saved update 30, beta .05 versus .25 gives position RMS 3.637 versus
3.604–3.605 mm and normal RMS 8.364 versus 6.533–6.534 degrees, with zero
inversions in all six cases. These earlier comparisons show that normal-angle
improvement precedes inversion, while the later high-pass advantage cannot be
claimed from the common inversion-free checkpoint.

- [Analysis JSON](../data/66-analysis/analysis.json) and [nine-panel histories](../data/66-analysis/strong-normal-histories.png); objective curves are normalized within branch, not cross-beta rankings.
- Matched update-200 figures with a fixed 9 mm scale: [full](../data/70-strong-figures/full-comparison.png), [mouth](../data/70-strong-figures/mouth-comparison.png), [error maps](../data/70-strong-figures/position-error-maps.png).
- Matched inversion-free update-30 figures using the same scale: [full](../data/71-strong-figures-noninverted/full-comparison.png), [mouth](../data/71-strong-figures-noninverted/mouth-comparison.png), [error maps](../data/71-strong-figures-noninverted/position-error-maps.png).
- [Main run](https://www.comet.com/liblaf/apple/831493f7812a43c8ab727d43d0df519a), [audit](https://www.comet.com/liblaf/apple/2116ff8e876f48ceb7576730a7519021), [analysis](https://www.comet.com/liblaf/apple/37db8ef7d2ca4b798e676e71a03f08b8), [endpoint render](https://www.comet.com/liblaf/apple/7e81b47f28914eb78a36e8b5f0c2453e), and [valid render](https://www.comet.com/liblaf/apple/98096176bf0340a59c97864d6c64a434). The two renders were visually checked for readable, unclipped panels, identical cameras/flat shading, and the shared 9 mm scale.

The error-map limit is the pooled 99th percentile over the six fits at updates
30 and 200, 8.68115289542123 mm, rounded up to 9 mm. Individual values above
9 mm saturate the color scale. The table's area-weighted RMS values are computed
independently of this display limit. All experiment, audit, analysis and render
processes completed with exit code zero after Comet shutdown.

Run from `exp/2026/09/21/normal-matching-face`; the main and preflight commands
are in the linked protocol and validation documents. Analysis and rendering:

```bash
CHERRIES_NAME='Raw6 stronger normal loss: audited beta analysis' CHERRIES_TAGS='face,raw6,normal-matching,strong-normal,analysis,cpu' .venv/bin/python src/66-analyze-strong.py
CHERRIES_NAME='3D face stronger normals: matched 200-update renders' CHERRIES_TAGS='face,3d,raw6,stronger-normal,beta-025,render,comparison' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/70-render-strong.py --shared-step 200 --error-limit-mm 9
CHERRIES_NAME='3D face stronger normals: common inversion-free checkpoint renders' CHERRIES_TAGS='face,3d,raw6,stronger-normal,beta-025,render,noninverted' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/70-render-strong.py --shared-step 30 --error-limit-mm 9 --output 71-strong-figures-noninverted
```
