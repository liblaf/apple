# 20 Inverse Face

## Result

- stop reason: `step_safety_limit`
- passed: `False`
- best step: `30`
- target mean error: `0.20674 cm`
- target RMS error: `0.305167 cm`
- target max error: `0.967125 cm`
- required max error: `< 0.2 cm`
- final loss: `0.0310423`
- best objective loss: `3.03034`
- lowest objective loss: `3.03034`
- optimizer steps: `30`
- series frames: `4`
- forward converged: `True` (0 failures)
- adjoint converged: `True` (0 failures, max relative residual `0.510734`)

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
- best metric: `target_max_error`
- stagnation metric: `loss`
- Adam: `lr=0.02`, `betas=(0.5, 0.9)`
- forward tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`
- adjoint tolerance: `rtol=0.0005`, `atol=0.5 * first forward residual`

## Trace

| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd | adj rel |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 3.13123 | 0.208805 | 0.980094 | 0.980094 | 0.0109457 | primary_success/1 | 0.000498689 |
| 1 | 3.13122 | 0.208806 | 0.980092 | 0.980092 | 0.0115083 | primary_success/3 | 0.124541 |
| 2 | 3.13093 | 0.208809 | 0.980055 | 0.980055 | 0.0106149 | primary_success/7 | 0.0970246 |
| 3 | 3.13091 | 0.208809 | 0.980053 | 0.980053 | 0.0110832 | primary_success/3 | 0.096694 |
| 4 | 3.13017 | 0.208813 | 0.979958 | 0.979958 | 0.0118121 | primary_success/14 | 0.102897 |
| 5 | 3.13019 | 0.208814 | 0.979961 | 0.979958 | 0.0137841 | primary_success/3 | 0.0893685 |
| 6 | 3.12953 | 0.208817 | 0.979877 | 0.979877 | 0.0149428 | primary_success/6 | 0.115124 |
| 7 | 3.12953 | 0.208816 | 0.979877 | 0.979877 | 0.0162844 | primary_success/3 | 0.130589 |
| 8 | 3.1283 | 0.208813 | 0.979721 | 0.979721 | 0.0189039 | primary_success/8 | 0.152172 |
| 9 | 3.12834 | 0.208808 | 0.979725 | 0.979721 | 0.020271 | primary_success/3 | 0.142885 |
| 10 | 3.12673 | 0.208783 | 0.97952 | 0.97952 | 0.023154 | primary_success/14 | 0.154173 |
| 11 | 3.12661 | 0.208779 | 0.979505 | 0.979505 | 0.0221232 | primary_success/6 | 0.189894 |
| 12 | 3.12367 | 0.208733 | 0.97913 | 0.97913 | 0.0249944 | primary_success/13 | 0.187725 |
| 13 | 3.1232 | 0.208714 | 0.97907 | 0.97907 | 0.033266 | primary_success/5 | 0.158269 |
| 14 | 3.12301 | 0.20871 | 0.979045 | 0.979045 | 0.026614 | primary_success/3 | 0.240044 |
| 15 | 3.11814 | 0.208596 | 0.978423 | 0.978423 | 0.0383932 | primary_success/18 | 0.171706 |
| 16 | 3.1178 | 0.20859 | 0.97838 | 0.97838 | 0.0306427 | primary_success/4 | 0.254387 |
| 17 | 3.11658 | 0.208581 | 0.978224 | 0.978224 | 0.0385566 | primary_success/6 | 0.192067 |
| 18 | 3.11568 | 0.208574 | 0.978109 | 0.978109 | 0.0224963 | primary_success/7 | 0.289791 |
| 19 | 3.11374 | 0.20856 | 0.977861 | 0.977861 | 0.048726 | primary_success/9 | 0.193358 |
| 20 | 3.11303 | 0.208552 | 0.97777 | 0.97777 | 0.0370572 | primary_success/6 | 0.262345 |
| 21 | 3.11073 | 0.208531 | 0.977476 | 0.977476 | 0.0321034 | primary_success/7 | 0.307593 |
| 22 | 3.11032 | 0.208528 | 0.977423 | 0.977423 | 0.0218757 | primary_success/2 | 0.497342 |
| 23 | 3.07052 | 0.207141 | 0.97232 | 0.97232 | 0.0650242 | primary_success/51 | 0.242779 |
| 24 | 3.06946 | 0.207138 | 0.972183 | 0.972183 | 0.06573 | primary_success/3 | 0.510734 |
| 25 | 3.06311 | 0.207113 | 0.971363 | 0.971363 | 0.075558 | primary_success/7 | 0.17569 |
| 26 | 3.05016 | 0.207057 | 0.969685 | 0.969685 | 0.0614223 | primary_success/8 | 0.259225 |
| 27 | 3.04764 | 0.206997 | 0.969362 | 0.969362 | 0.0576688 | primary_success/7 | 0.285453 |
| 28 | 3.0477 | 0.206985 | 0.96937 | 0.969362 | 0.0617203 | primary_success/3 | 0.232257 |
| 29 | 3.03892 | 0.206863 | 0.968235 | 0.968235 | 0.0659262 | primary_success/10 | 0.286669 |
| 30 | 3.03034 | 0.20674 | 0.967125 | 0.967125 | 0.0593495 | primary_success/8 | 0.352207 |
