# Inverse Face

## Result

- stop reason: `stagnation`
- passed: `True`
- target mean error: `0.259231`
- target RMS error: `0.524082`
- target max error: `4.28231`
- final loss: `0.091554`
- optimizer steps: `12`
- series frames: `4`

## Problem

- input: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-input.vtu`
- target: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/10-inverse-face-target.vtu`
- output: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu`
- optimization series: `/home/liblaf/github/liblaf/apple/exp/2026/05/20/inverse-face/data/20-inverse-face.vtu.series`
- points: `225052`
- tetrahedra: `1127541`
- target surface points: `31339`
- active muscle tetrahedra: `283391`
- activation parameters: `1700346`

## Model

- material: stable neo-Hookean, `nu = 0.49`
- SMAS stiffness ratio: `100.0`
- collisions: `off`
- optimized field: `ActivationInv`
- Adam: `lr=0.5`, `betas=(0.0, 0.9)`

## Trace

| step | loss | target mean error | best mean error | grad norm |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0942007 | 0.259684 | 0.259684 | 1.73759e-05 |
| 1 | 0.0940601 | 0.259554 | 0.259554 | 0.000168635 |
| 2 | 0.0935207 | 0.259116 | 0.259116 | 0.000241895 |
| 3 | 0.092776 | 0.26081 | 0.26081 | 0.000163617 |
| 4 | 0.091554 | 0.259231 | 0.259231 | 0.000142449 |
| 5 | 0.0921181 | 0.262659 | 0.259231 | 0.000134466 |
| 6 | 0.0919054 | 0.262468 | 0.259231 | 0.000125025 |
| 7 | 0.0919642 | 0.261912 | 0.259231 | 0.000129808 |
| 8 | 0.0922861 | 0.262257 | 0.259231 | 0.000143755 |
| 9 | 0.092914 | 0.263754 | 0.259231 | 0.00013957 |
| 10 | 0.0933609 | 0.265114 | 0.259231 | 0.000138675 |
| 11 | 0.0942386 | 0.266693 | 0.259231 | 0.000138555 |
| 12 | 0.0948683 | 0.268234 | 0.259231 | 0.000153822 |
