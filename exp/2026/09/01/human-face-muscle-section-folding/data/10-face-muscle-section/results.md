# Human face muscle section folding

All determinants are recomputed from the materialized endpoints. `DoubleInverted` means `DetF < 0` and `DetAinv < 0`; it is descriptive, not a constraint or repair.

| case | best step | best RMS (mm) | active F-negative volume | zygomaticus F-negative volume | zygomaticus F&A double-inverted cells | roughness (laplacian RMS) | forward failures | inverse converged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20-human-face-smile-no-skin-lr1 | 5 | 3.63 | 0.007818 | 0.002101 | 2 | 0.0002412 | 155 | False |
| 20-human-face-smile-no-skin-lr3 | 194 | 0.6336 | 0.0004161 | 0.004392 | 4 | 0.0002135 | 6 | False |
| 20-human-face-smile-skin-estimated-plus-tightening-lr1 | 192 | 2.243 | 0.003529 | 0.01588 | 24 | 8.2e-05 | 2 | False |
| 20-human-face-smile-skin-estimated-plus-tightening-lr2-cont-lr02-warm-from-best | 9 | 2.231 | 0.003566 | 0.01538 | 24 | 8.202e-05 | 0 | False |
| 20-human-face-smile-skin-no-prestrain-lr1 | 190 | 1.915 | 0.004601 | 0.02813 | 54 | 0.0001411 | 1 | False |
| 20-human-face-smile-skin-no-prestrain-lr3-cont-lr03-from-best | 1 | 1.915 | 0.004605 | 0.02813 | 54 | 0.0001411 | 0 | False |
