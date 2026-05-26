# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `0`
- target mean error: `0.208763 cm`
- target RMS error: `0.306504 cm`
- target max error: `0.980094 cm`
- required max error: `< 0.2 cm`
- final loss: `0.031315`
- optimizer steps: `0`
- series frames: `1`

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
- Adam: `lr=0.1`, `betas=(0.0, 0.9)`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.13118 | 0.208763 | 0.980094 | 0.980094 | 0.00347541 |
