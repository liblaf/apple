# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `3`
- target mean error: `0.207434 cm`
- target RMS error: `0.303476 cm`
- target max error: `0.971156 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0306992`
- best objective loss: `0.0306992`
- lowest objective loss: `0.0306992`
- optimizer steps: `3`
- series frames: `1`
- forward converged: `True` (0 failures, max absolute grad `0.000735445`, max relative grad `0.000498391`)
- adjoint converged: `True` (0 failures, max absolute residual `0.00554721`, max relative residual `3.6356`)

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
- loss: point-to-point mean squared displacement error on `IsFace` points
- initialization: zero `ActivationInv`
- Adam: `lr=0.02`, `betas=(0.5, 0.9)`
- forward tolerance: `rtol=0.0005`, `atol=0`
- adjoint tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd abs | fwd rel | adj abs | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 3.27307e-05 | 0 | 0 | 7.61495e-07 | 0.000494033 |
| 1 | 0.0311091 | 0.20818 | 0.977066 | 0.977066 | 3.29313e-05 | 0.000636726 | 0.000496988 | 0.00412435 | 2.6852 |
| 2 | 0.030897 | 0.207751 | 0.97406 | 0.97406 | 3.37632e-05 | 0.000689821 | 0.000498391 | 0.00418746 | 2.73563 |
| 3 | 0.0306992 | 0.207434 | 0.971156 | 0.971156 | 3.50256e-05 | 0.000735445 | 0.000490708 | 0.00554721 | 3.6356 |
