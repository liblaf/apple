# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `20`
- target mean error: `0.208651 cm`
- target RMS error: `0.306453 cm`
- target max error: `0.980076 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0313045`
- best objective loss: `0.0313045`
- lowest objective loss: `0.0313045`
- optimizer steps: `20`
- series frames: `1`
- forward converged: `True` (0 failures)
- adjoint converged: `False` (1 failures, max relative residual `7.18244`)

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
- best metric: `loss`
- stagnation metric: `loss`
- Adam: `lr=0.02`, `betas=(0.5, 0.9)`
- forward tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`
- adjoint tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 6.42803e-06 | primary_success/1 | 1.09584 |
| 1 | 0.0313291 | 0.208803 | 0.980096 | 0.980096 | 8.38437e-07 | primary_success/4 | 7.17979 |
| 2 | 0.0313267 | 0.208789 | 0.980102 | 0.980102 | 8.69853e-07 | primary_success/6 | 7.18025 |
| 3 | 0.0313263 | 0.208787 | 0.980103 | 0.980103 | 9.19342e-07 | primary_success/2 | 7.18026 |
| 4 | 0.0313171 | 0.208741 | 0.980111 | 0.980111 | 9.83042e-07 | primary_success/10 | 7.18125 |
| 5 | 0.0313168 | 0.208739 | 0.980112 | 0.980112 | 1.06489e-06 | primary_success/2 | 7.18132 |
| 6 | 0.0313164 | 0.208737 | 0.980111 | 0.980111 | 1.16318e-06 | primary_success/2 | 7.18141 |
| 7 | 0.0313156 | 0.208732 | 0.980111 | 0.980111 | 1.27768e-06 | primary_success/3 | 7.18115 |
| 8 | 0.0313152 | 0.20873 | 0.98011 | 0.98011 | 1.41174e-06 | primary_success/2 | 7.18112 |
| 9 | 0.0313144 | 0.208725 | 0.980109 | 0.980109 | 1.56269e-06 | primary_success/3 | 7.18122 |
| 10 | 0.0313141 | 0.208722 | 0.980109 | 0.980109 | 1.73668e-06 | primary_success/2 | 7.18137 |
| 11 | 0.0313128 | 0.208714 | 0.980107 | 0.980107 | 1.92587e-06 | primary_success/4 | 7.1814 |
| 12 | 0.0313126 | 0.208712 | 0.980106 | 0.980106 | 2.14479e-06 | primary_success/2 | 7.18152 |
| 13 | 0.031312 | 0.208708 | 0.980103 | 0.980103 | 2.38269e-06 | primary_success/2 | 7.18155 |
| 14 | 0.0313097 | 0.208691 | 0.9801 | 0.9801 | 2.63671e-06 | primary_success/5 | 7.1818 |
| 15 | 0.0313094 | 0.208689 | 0.980095 | 0.980095 | 2.92827e-06 | primary_success/2 | 7.18191 |
| 16 | 0.0313088 | 0.208685 | 0.980094 | 0.980094 | 3.24166e-06 | primary_success/2 | 7.18191 |
| 17 | 0.0313069 | 0.208671 | 0.98009 | 0.98009 | 3.57041e-06 | primary_success/4 | 7.18195 |
| 18 | 0.0313067 | 0.208668 | 0.980086 | 0.980086 | 3.94633e-06 | primary_success/2 | 7.18221 |
| 19 | 0.0313051 | 0.208656 | 0.980077 | 0.980077 | 4.31914e-06 | primary_success/3 | 7.18209 |
| 20 | 0.0313045 | 0.208651 | 0.980076 | 0.980076 | 4.75092e-06 | primary_success/2 | 7.18244 |
