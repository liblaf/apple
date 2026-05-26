# 20 Inverse Face

## Result

- stop reason: `stagnation`
- passed: `False`
- best step: `0`
- target mean error: `0.208805 cm`
- target RMS error: `0.306575 cm`
- target max error: `0.980094 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0313294`
- optimizer steps: `8`
- series frames: `2`

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
- Adam: `lr=3.0`, `betas=(0.0, 0.5)`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.511301 | 0.208805 | 0.980094 | 0.980094 | 3.25892e-05 |
| 1 | 0.482269 | 0.206785 | 1.01749 | 0.980094 | 0.00627529 |
| 2 | 0.509762 | 0.21878 | 1.00947 | 0.980094 | 0.023654 |
| 3 | 0.529696 | 0.227534 | 1.04741 | 0.980094 | 0.0963381 |
| 4 | 0.546939 | 0.234314 | 1.0306 | 0.980094 | 0.0490803 |
| 5 | 0.54989 | 0.238002 | 1.03109 | 0.980094 | 0.0423305 |
| 6 | 0.541504 | 0.238898 | 1.08 | 0.980094 | 0.053173 |
| 7 | 0.536252 | 0.239361 | 1.06412 | 0.980094 | 0.0485482 |
| 8 | 0.556958 | 0.244248 | 1.10288 | 0.980094 | 0.035495 |
