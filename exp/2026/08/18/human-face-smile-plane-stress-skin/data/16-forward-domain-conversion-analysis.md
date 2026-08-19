# Fixed-activation domain x conversion probe

| case | seed | domain | conversion | cut boundary | error/target | rest-area error mm | target-area error mm | dihedral deg | residual-normal Lap mm | inverted tets | folded face tris | forward |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full-3d-replay | zero | full | 3d | current | 0.727891 | 3.63173 | 3.67067 | 7.13801 | 0.243887 | 20 | 0 | primary_success |
| full-3d-replay | old | full | 3d | current | 0.602469 | 3.00959 | 3.04582 | 8.5444 | 0.227211 | 29 | 0 | primary_success |
| full-plane-stress | zero | full | plane-stress | current | 0.724917 | 3.64329 | 3.68299 | 8.86203 | 0.250681 | 21 | 0 | primary_success |
| full-plane-stress | old | full | plane-stress | current | 0.622119 | 3.12209 | 3.16264 | 10.1559 | 0.243061 | 22 | 2 | primary_success |
| isface-3d | zero | isface | 3d | current | 0.704636 | 3.56247 | 3.5955 | 7.86174 | 0.230827 | 99 | 0 | primary_success |
| isface-3d | old | isface | 3d | current | 0.600786 | 3.03063 | 3.07116 | 9.46688 | 0.215689 | 28 | 3 | primary_success |
| isface-plane-stress | zero | isface | plane-stress | current | 0.709494 | 3.57969 | 3.62051 | 9.25293 | 0.23868 | 12 | 1 | primary_success |
| isface-plane-stress | old | isface | plane-stress | current | 0.614102 | 3.09935 | 3.14896 | 10.7005 | 0.23305 | 23 | 3 | primary_success |
| isface-plane-stress-cut-fixed | zero | isface | plane-stress | hard-fixed | 0.715134 | 3.61529 | 3.65536 | 9.70684 | 0.240036 | 22 | 0 | primary_success |
| isface-plane-stress-cut-fixed | old | isface | plane-stress | hard-fixed | 0.614751 | 3.10943 | 3.15789 | 11.083 | 0.234343 | 25 | 4 | primary_success |

## Zero/old branch checks

| case | loss delta / target | IsFace delta / target | error-fraction delta | stable |
| --- | ---: | ---: | ---: | --- |
| full-3d-replay | 0.241569 | 0.241569 | 0.125423 | False |
| full-plane-stress | 0.190237 | 0.190237 | 0.102798 | False |
| isface-3d | 0.221459 | 0.221459 | 0.10385 | False |
| isface-plane-stress | 0.249106 | 0.249105 | 0.0953916 | False |
| isface-plane-stress-cut-fixed | 0.238126 | 0.238126 | 0.100383 | False |

## Current vs hard-fixed cut-boundary sensitivity

The hard-fixed variant is a sensitivity bracket, not an anatomical ground truth.

| seed | loss delta / target | IsFace delta / target | error-fraction delta | dihedral delta deg | residual-normal Lap delta mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 0.0575209 | 0.0574672 | 0.0056402 | 0.453907 | 0.00135635 |
| old | 0.0635745 | 0.0635178 | 0.000648518 | 0.382442 | 0.00129297 |

## Historical replay gate

- Reproduced within tolerance: True
- Loss-mask delta / target RMS: 7.48227e-05
- IsFace delta / target RMS: 7.48217e-05
- Fold and inversion counts are visual-review warnings, not vetoes.
