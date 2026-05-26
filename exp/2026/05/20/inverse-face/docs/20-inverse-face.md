# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `20`
- target mean error: `0.194159 cm`
- target RMS error: `0.279061 cm`
- target max error: `0.908447 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0259584`
- best objective loss: `0.0259584`
- lowest objective loss: `0.0259584`
- optimizer steps: `20`
- series frames: `3`
- forward converged: `True` (0 failures, max absolute grad `0.00487892`, max relative grad `0.000498756`)
- adjoint converged: `True` (0 failures, max absolute residual `7.66268e-07`, max relative residual `0.000499901`)

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
- adjoint tolerance: `rtol=0.0005`, `atol=0`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd abs | fwd rel | adj abs | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0313294 | 0.208805 | 0.980094 | 0.980094 | 3.27307e-05 | 0 | 0 | 7.52756e-07 | 0.000488363 |
| 1 | 0.031108 | 0.208178 | 0.977044 | 0.977044 | 3.21163e-05 | 0.000634813 | 0.000495494 | 7.66268e-07 | 0.000498896 |
| 2 | 0.0308989 | 0.207716 | 0.974071 | 0.974071 | 3.20487e-05 | 0.000641168 | 0.00048643 | 7.58603e-07 | 0.000495573 |
| 3 | 0.0306904 | 0.207309 | 0.970989 | 0.970989 | 3.24909e-05 | 0.00068593 | 0.000487534 | 7.42444e-07 | 0.000486662 |
| 4 | 0.0304816 | 0.206927 | 0.967803 | 0.967803 | 3.34357e-05 | 0.000750691 | 0.000484333 | 7.50923e-07 | 0.000493902 |
| 5 | 0.030282 | 0.206519 | 0.964668 | 0.964668 | 3.49558e-05 | 0.000821134 | 0.000470807 | 7.49706e-07 | 0.000494724 |
| 6 | 0.0300618 | 0.206118 | 0.961091 | 0.961091 | 3.68747e-05 | 0.000963104 | 0.000484489 | 7.46821e-07 | 0.000494622 |
| 7 | 0.0298428 | 0.205677 | 0.957459 | 0.957459 | 3.94932e-05 | 0.00111848 | 0.000491587 | 7.3318e-07 | 0.000487367 |
| 8 | 0.0296206 | 0.205178 | 0.953693 | 0.953693 | 4.25704e-05 | 0.00126315 | 0.000484154 | 7.37996e-07 | 0.000492405 |
| 9 | 0.0293747 | 0.204677 | 0.9514 | 0.9514 | 4.59224e-05 | 0.00146176 | 0.000489106 | 7.38368e-07 | 0.00049471 |
| 10 | 0.0291267 | 0.204087 | 0.949034 | 0.949034 | 5.00958e-05 | 0.0016074 | 0.000470587 | 7.39423e-07 | 0.000497522 |
| 11 | 0.0288567 | 0.203489 | 0.946365 | 0.946365 | 5.46616e-05 | 0.00193844 | 0.000498756 | 7.31824e-07 | 0.000494707 |
| 12 | 0.0285737 | 0.202771 | 0.943484 | 0.943484 | 5.85457e-05 | 0.00216121 | 0.000490733 | 7.31985e-07 | 0.00049726 |
| 13 | 0.0282813 | 0.201902 | 0.940355 | 0.940355 | 6.23054e-05 | 0.0024494 | 0.00049346 | 7.20957e-07 | 0.000492293 |
| 14 | 0.027973 | 0.20108 | 0.93693 | 0.93693 | 6.61768e-05 | 0.00276644 | 0.000496402 | 7.27524e-07 | 0.000499508 |
| 15 | 0.0276521 | 0.200049 | 0.933241 | 0.933241 | 6.98731e-05 | 0.00307314 | 0.000494402 | 7.14232e-07 | 0.00049322 |
| 16 | 0.0273247 | 0.198945 | 0.929254 | 0.929254 | 7.31674e-05 | 0.00343189 | 0.00049812 | 7.19609e-07 | 0.000499901 |
| 17 | 0.0269813 | 0.197803 | 0.924701 | 0.924701 | 7.57836e-05 | 0.0037382 | 0.000492173 | 7.07156e-07 | 0.000494366 |
| 18 | 0.0266477 | 0.196616 | 0.919921 | 0.919921 | 7.97831e-05 | 0.00405612 | 0.000487938 | 6.982e-07 | 0.000491151 |
| 19 | 0.026317 | 0.195438 | 0.914486 | 0.914486 | 8.31487e-05 | 0.00450062 | 0.000497028 | 7.05787e-07 | 0.000499597 |
| 20 | 0.0259584 | 0.194159 | 0.908447 | 0.908447 | 8.57396e-05 | 0.00487892 | 0.000497416 | 6.84737e-07 | 0.000488033 |
