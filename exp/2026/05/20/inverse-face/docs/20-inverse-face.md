# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `1`
- target mean error: `0.208804 cm`
- target RMS error: `0.306574 cm`
- target max error: `0.980091 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0313292`
- optimizer steps: `20`
- series frames: `1`

## Problem

- input: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-input.vtu`
- target: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-target.vtu`
- output: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu`
- optimization series: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu.series`
- points: `225052`
- tetrahedra: `1127541`
- target `IsFace` points: `17582`
- active muscle tetrahedra: `283391`
- activation parameters: `1700346`
- target displacement max: `0.980094 cm`

## Model

- material: stable neo-Hookean, `nu = 0.49`
- SMAS stiffness ratio: `100.0`
- collisions: `off`
- optimized field: `per active muscle tetrahedron ActivationInv, 6 DoF`
- Adam: `lr=0.01`, `betas=(0.0, 0.9)`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 1.3337e-06 |
| 1 | 0.0313292 | 0.208804 | 0.980091 | 0.980091 | 1.37173e-06 |
| 2 | 0.0313291 | 0.208802 | 0.980091 | 0.980091 | 1.41832e-06 |
| 3 | 0.0313289 | 0.208801 | 0.980093 | 0.980091 | 1.4734e-06 |
| 4 | 0.0313288 | 0.208799 | 0.980097 | 0.980091 | 1.53699e-06 |
| 5 | 0.0313287 | 0.208798 | 0.980101 | 0.980091 | 1.60909e-06 |
| 6 | 0.0313285 | 0.208796 | 0.980096 | 0.980091 | 1.68967e-06 |
| 7 | 0.0313285 | 0.208795 | 0.980108 | 0.980091 | 1.77915e-06 |
| 8 | 0.0313283 | 0.208793 | 0.980105 | 0.980091 | 1.87723e-06 |
| 9 | 0.0313284 | 0.208793 | 0.980121 | 0.980091 | 1.98453e-06 |
| 10 | 0.0313281 | 0.208791 | 0.980114 | 0.980091 | 2.10072e-06 |
| 11 | 0.0313279 | 0.208789 | 0.980118 | 0.980091 | 2.22639e-06 |
| 12 | 0.0313281 | 0.208789 | 0.980132 | 0.980091 | 2.36169e-06 |
| 13 | 0.0313278 | 0.208787 | 0.980124 | 0.980091 | 2.50646e-06 |
| 14 | 0.0313276 | 0.208785 | 0.980129 | 0.980091 | 2.66137e-06 |
| 15 | 0.0313278 | 0.208785 | 0.980145 | 0.980091 | 2.82656e-06 |
| 16 | 0.0313275 | 0.208782 | 0.980136 | 0.980091 | 3.00177e-06 |
| 17 | 0.0313272 | 0.20878 | 0.980141 | 0.980091 | 3.18776e-06 |
| 18 | 0.0313276 | 0.20878 | 0.980156 | 0.980091 | 3.38466e-06 |
| 19 | 0.0313272 | 0.208777 | 0.980149 | 0.980091 | 3.59206e-06 |
| 20 | 0.0313269 | 0.208775 | 0.980154 | 0.980091 | 3.81079e-06 |
