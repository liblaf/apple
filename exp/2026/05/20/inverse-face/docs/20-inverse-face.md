# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `10`
- target mean error: `0.207776 cm`
- target RMS error: `0.304515 cm`
- target max error: `0.974198 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0309098`
- best objective loss: `0.0309098`
- lowest objective loss: `0.0309098`
- optimizer steps: `10`
- series frames: `3`
- forward converged: `True` (0 failures, max absolute grad `0.00063588`, max relative grad `0.000499972`)
- adjoint converged: `True` (0 failures, max absolute residual `0.0110186`, max relative residual `7.17384`)

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
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 3.27307e-05 | 0 | 0 | 7.55893e-07 | 0.000490398 |
| 1 | 0.0311085 | 0.208179 | 0.977055 | 0.977055 | 8.26268e-07 | 0.00063588 | 0.000496328 | 0.0110186 | 7.17384 |
| 2 | 0.0310144 | 0.207978 | 0.975727 | 0.975727 | 8.55421e-07 | 0.000290475 | 0.000498582 | 0.0109978 | 7.17117 |
| 3 | 0.0309656 | 0.207888 | 0.97502 | 0.97502 | 9.02297e-07 | 0.000166972 | 0.00049975 | 0.0109868 | 7.16965 |
| 4 | 0.0309398 | 0.207842 | 0.974642 | 0.974642 | 9.64853e-07 | 0.000121972 | 0.000498107 | 0.010981 | 7.16884 |
| 5 | 0.0309261 | 0.207817 | 0.97444 | 0.97444 | 1.04288e-06 | 0.000112114 | 0.000491904 | 0.0109779 | 7.1684 |
| 6 | 0.0309187 | 0.207803 | 0.974328 | 0.974328 | 1.13677e-06 | 0.000117579 | 0.000489976 | 0.0109763 | 7.16817 |
| 7 | 0.0309147 | 0.207793 | 0.974272 | 0.974272 | 1.24721e-06 | 0.000129043 | 0.000489319 | 0.0109754 | 7.1681 |
| 8 | 0.0309123 | 0.207787 | 0.974238 | 0.974238 | 1.37503e-06 | 0.000147117 | 0.000499972 | 0.010975 | 7.16808 |
| 9 | 0.0309117 | 0.207782 | 0.974235 | 0.974235 | 1.52118e-06 | 0.000161217 | 0.000487628 | 0.010975 | 7.16815 |
| 10 | 0.0309098 | 0.207776 | 0.974198 | 0.974198 | 1.68674e-06 | 0.000184502 | 0.000494697 | 0.0109746 | 7.16813 |
