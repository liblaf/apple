# Inverse Face

## Result

- stop reason: `mean_point_error_tol`
- passed: `True`
- target mean error: `0.0910711 cm`
- target RMS error: `0.122756 cm`
- target max error: `0.796076 cm`
- final loss: `0.00502302`
- optimizer steps: `3`
- series frames: `4`

## Problem

- input: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-input.vtu`
- target: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-target.vtu`
- output: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu`
- optimization series: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu.series`
- points: `225052`
- tetrahedra: `1127541`
- target surface points: `17582`
- active muscle tetrahedra: `283391`
- activation values written: `1700346`

## Model

- material: stable neo-Hookean, `nu = 0.49`
- SMAS stiffness ratio: `100.0`
- collisions: `off`
- optimized field: `muscle mean plus tet residual ActivationInv`
- Adam: `lr=0.3`, `betas=(0.0, 0.9)`

## Trace

| step | loss | target mean error (cm) | best mean error (cm) | muscle grad | tet grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0132676 | 0.167013 | 0.167013 | 0.00022795 | 3.8493e-07 |
| 1 | 0.0120128 | 0.158432 | 0.158432 | 0.000935831 | 1.64341e-06 |
| 2 | 0.00827229 | 0.126432 | 0.126432 | 0.00177499 | 3.79529e-06 |
| 3 | 0.00502302 | 0.0910711 | 0.0910711 | 0.000421169 | 4.54191e-06 |
