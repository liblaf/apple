# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `3`
- target mean error: `0.207053 cm`
- target RMS error: `0.302144 cm`
- target max error: `0.967076 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0304303`
- best objective loss: `0.0304303`
- lowest objective loss: `0.0304303`
- optimizer steps: `3`
- series frames: `1`
- forward converged: `True` (0 failures, max absolute grad `0.00125896`, max relative grad `0.000493867`)
- adjoint converged: `True` (0 failures, max absolute residual `0.00798412`, max relative residual `5.25579`)

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
- Adam: `lr=0.03`, `betas=(0.5, 0.9)`
- forward tolerance: `rtol=0.0005`, `atol=0`
- adjoint tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd abs | fwd rel | adj abs | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 3.27307e-05 | 0 | 0 | 7.52966e-07 | 0.000488499 |
| 1 | 0.0309992 | 0.207944 | 0.975502 | 0.975502 | 3.35943e-05 | 0.000949354 | 0.000493867 | 0.00623568 | 4.06699 |
| 2 | 0.0307027 | 0.207437 | 0.97121 | 0.97121 | 3.6178e-05 | 0.00109852 | 0.000491885 | 0.00604682 | 3.96281 |
| 3 | 0.0304303 | 0.207053 | 0.967076 | 0.967076 | 3.88128e-05 | 0.00125896 | 0.00048563 | 0.00798412 | 5.25579 |
