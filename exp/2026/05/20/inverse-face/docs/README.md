# Inverse Face

## Result

- stop reason: `mean_point_error_tol`
- passed: `True`
- target mean error: `0.0766953 cm`
- target RMS error: `0.0923131 cm`
- target max error: `0.319675 cm`
- final loss: `0.00284057`
- optimizer steps: `1`
- series frames: `2`

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
- Adam: `lr=0.1`, `betas=(0.0, 0.9)`

## Trace

| step | loss | target mean error (cm) | best mean error (cm) | muscle grad | tet grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00665554 | 0.101503 | 0.101503 | 0.00256377 | 1.06505e-05 |
| 1 | 0.00284057 | 0.0766953 | 0.0766953 | 0.00527571 | 1.19705e-05 |
