# Inverse Face

## Result

- stop reason: `mean_point_error_tol`
- passed: `True`
- target mean error: `0.0700013 cm`
- target RMS error: `0.0902272 cm`
- target max error: `0.524719 cm`
- final loss: `0.00271365`
- optimizer steps: `0`
- series frames: `1`

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
- Adam: `lr=2.0`, `betas=(0.0, 0.9)`

## Trace

| step | loss | target mean error (cm) | best mean error (cm) | muscle grad | tet grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00271365 | 0.0700013 | 0.0700013 | 0.000546251 | 6.10998e-07 |
