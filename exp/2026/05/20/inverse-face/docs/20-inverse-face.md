# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `3`
- target mean error: `0.208801 cm`
- target RMS error: `0.306573 cm`
- target max error: `0.980093 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0313289`
- best objective loss: `0.0313289`
- lowest objective loss: `0.0313289`
- optimizer steps: `3`
- series frames: `1`
- forward converged: `True` (0 failures)
- adjoint converged: `True` (0 failures, max relative residual `5.65749`)

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
- Adam: `lr=0.01`, `betas=(0.0, 0.9)`
- forward tolerance: `rtol=0.01`, `atol=0.0001`
- adjoint tolerance: `rtol=0.01`, `atol=0.0`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 1.3337e-06 | primary_success/1 | 5.65749 |
| 1 | 0.0313292 | 0.208804 | 0.980091 | 0.980091 | 1.37173e-06 | primary_success/19 | 5.65749 |
| 2 | 0.0313291 | 0.208802 | 0.980091 | 0.980091 | 1.41832e-06 | primary_success/24 | 5.65749 |
| 3 | 0.0313289 | 0.208801 | 0.980093 | 0.980093 | 1.4734e-06 | primary_success/29 | 5.65748 |
