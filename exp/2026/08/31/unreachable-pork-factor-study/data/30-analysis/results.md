# Unreachable pork factor-study analysis

Complete: all 16 required OFAT cases are present.

`Unreachable` is the deliberately demanding benchmark label, not a mathematical infeasibility certificate. These finite trajectories do not estimate a global or orientation-preserving reachability lower bound.

Rows labelled `valid_best` are selected only from usable inverse evaluations. Physical stationarity gates and refinement/trial failures are reported separately; nonstationary cases are deliberately retained, and the legacy 1% tail gate is not treated as convergence. Inversions are observations, not optimization constraints. Blank CSV cells mean the runner did not provide that metric.

| Dimension | Case | Physical stationarity | Valid/every frame | Final target RMS | Refinement iters | Trial failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2d | baseline | False | 1245/1245 | 0.001559 | 43 | 0 |
| 2d | energy-linear | True | 1297/1297 | 0.0001471 | 95 | 0 |
| 2d | height-high | False | 1237/1237 | 0.004714 | 35 | 0 |
| 2d | height-low | False | 1253/1253 | 0.0002799 | 51 | 0 |
| 2d | loss-l1 | False | 1406/1406 | 0.0008708 | 204 | 0 |
| 2d | loss-linf | False | 1491/1491 | 0.002763 | 289 | 0 |
| 2d | mesh-dense | False | 1259/1259 | 0.0007833 | 57 | 0 |
| 2d | mesh-medium | True | 1282/1282 | 0.0001468 | 80 | 0 |

## Trajectory determinant minima

These are finite minima across every recorded trace frame, rather than determinants at only the best or final checkpoint.

| Dimension | Case | Minimum detF | Minimum detG | Minimum detAinv |
| --- | --- | ---: | ---: | ---: |
| 2d | baseline | -26.31 | -0.1203 | -0.7569 |
| 2d | energy-linear | -0.07157 | -1.549 | -1.336 |
| 2d | height-high | -131.5 | -0.3408 | -0.553 |
| 2d | height-low | 0.4277 | 0.2117 | 0.09834 |
| 2d | loss-l1 | -14.93 | -0.02961 | -0.2729 |
| 2d | loss-linf | -18.35 | -0.1412 | -0.8863 |
| 2d | mesh-dense | -46.45 | -0.2672 | -0.6972 |
| 2d | mesh-medium | -11.5 | -0.01454 | -0.01437 |

## OFAT comparison

`is_controlled_ofat` is true only when exactly one of energy, loss, resolution, or height differs from the shared same-dimension baseline.

| Dimension | Case | Changed factor | Controlled OFAT | Delta best target RMS | Delta best highpass |
| --- | --- | --- | --- | ---: | ---: |
| 2d | baseline | NA | False | 0.0 | 0.0 |
| 2d | energy-linear | energy | True | -0.001412017950791928 | -0.0008558981401889894 |
| 2d | height-high | height | True | 0.0031552386604213325 | 0.002054194926297055 |
| 2d | height-low | height | True | -0.0012791800473096928 | -0.0009456910547844061 |
| 2d | loss-l1 | loss | True | -0.0006882819125100803 | -0.0005102482507689084 |
| 2d | loss-linf | loss | True | 0.001203789354003811 | 0.0004232856102523862 |
| 2d | mesh-dense | resolution | True | -0.0007758194495096658 | -0.0006148503764376175 |
| 2d | mesh-medium | resolution | True | -0.0014123371646268539 | -0.000873339119486014 |

## Control and observation counts

These post-hoc counts compare raw activation DoFs with the vector displacement components observed on free top vertices. They quantify the count-based aspect of potential underdetermination, but do not prove artifact causality or target reachability.

| Dimension | Case | Cells | Muscle cells | Activation DoFs | Free top vertices | Observed components | DoFs / component |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2d | baseline | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | energy-linear | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | height-high | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | height-low | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | loss-l1 | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | loss-linf | 2000 | 400 | 1200 | 99 | 198 | 6.061 |
| 2d | mesh-dense | 8000 | 1600 | 4800 | 199 | 398 | 12.06 |
| 2d | mesh-medium | 500 | 100 | 300 | 49 | 98 | 3.061 |

## Bumpiness diagnostic

`bumpiness-mechanisms.png` plots final activation neighbor-jump RMS against final top high-pass RMS in separate 2-D and 3-D panels. Circles have no recorded inversion; triangles have a recorded inversion. It is descriptive only and does not fit or assert a causal relationship.
