# Corrected baseline checkpoints

Primary matching tau: `0.51240621`. The historical old-skin case is excluded from tau.

| checkpoint | case | role | step | error/target | target-area error/target | dihedral deg | residual-normal Lap mm | seam residual ratio | seam normal-Lap ratio | cut max mm | cut exact zero | inverted | folded |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| terminal | isface-e0200-p000 | primary-corrected | 40 | 0.512406 | 0.517323 | 13.329 | 0.217061 | 0.724355 | 1.73626 | 0 | True | 47 | 25 |
| terminal | old-e100-p000 | old-boundary-secondary-historical-diagnostic | 40 | 0.602461 | 0.592527 | 8.54438 | 0.227208 | 0.844807 | 2.85042 | 3.29423 | False | 29 | 0 |
| terminal | no-skin | old-boundary-external-no-skin-control | 40 | 0.239161 | 0.252179 | 9.4449 | 0.155229 | 0.731484 | 0.893533 | 1.80104 | False | 69 | 5 |
| matched | isface-e0200-p000 | primary-corrected | 40 | 0.512406 | 0.517323 | 13.329 | 0.217061 | 0.724355 | 1.73626 | 0 | True | 47 | 25 |
| matched | old-e100-p000 | old-boundary-secondary-historical-diagnostic | 40 | 0.602461 | 0.592527 | 8.54438 | 0.227208 | 0.844807 | 2.85042 | 3.29423 | False | 29 | 0 |
| matched | no-skin | old-boundary-external-no-skin-control | 11 | 0.520904 | 0.553316 | 5.94089 | 0.13161 | 0.622101 | 1.72208 | 0.918168 | False | 0 | 2 |

The IsFace membrane has exactly 707 boundary edges. Seam metrics are interpretation and visual-review evidence, not automatic vetoes.

The no-skin and old-skin cases retain the historical boundary. They are controls, not boundary-matched causal material ablations.

Small inversion or fold counts are also warnings only when the artifact is visually imperceptible.
