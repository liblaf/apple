# Exaggerated material screen checkpoints

Inverted tetrahedra and folded triangles are recorded as visual-review warnings only. They do not remove a trajectory or checkpoint.

## Terminal fixed-budget checkpoint (step 40)

| candidate | origin | step | error/target | error RMS mm | contraction dihedral deg | displacement Laplacian mm | inv tets | folds | warning only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| e100-p000 | reused-2026-08-17 | 40 | 0.602461 | 3.19915 | 8.54438 | 0.119084 | 29 | 49 | 29 inverted tets; 49 folded triangles |
| e100-p200 | new-2026-08-18 | 40 | 0.583698 | 3.09952 | 4.21326 | 0.0769014 | 24 | 120 | 24 inverted tets; 120 folded triangles |
| e005-p000 | new-2026-08-18 | 40 | 0.564042 | 2.99514 | 8.84549 | 0.118017 | 51 | 58 | 51 inverted tets; 58 folded triangles |
| e005-p200 | new-2026-08-18 | 40 | 0.546565 | 2.90233 | 4.23814 | 0.074585 | 23 | 104 | 23 inverted tets; 104 folded triangles |
| e025-p100 | reused-2026-08-17 | 40 | 0.551165 | 2.92676 | 4.95345 | 0.0819422 | 33 | 80 | 33 inverted tets; 80 folded triangles |
| no-skin | reused-2026-08-17 | 40 | 0.239161 | 1.26998 | 9.4449 | 0.218168 | 69 | 349 | 69 inverted tets; 349 folded triangles |

The common-fidelity target is `0.602460923`. Each row is the closest actual saved checkpoint; no geometry is interpolated.

## Nearest discrete common-fidelity checkpoint

| candidate | origin | step | error/target | error RMS mm | contraction dihedral deg | displacement Laplacian mm | inv tets | folds | warning only |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| e100-p000 | reused-2026-08-17 | 40 | 0.602461 | 3.19915 | 8.54438 | 0.119084 | 29 | 49 | 29 inverted tets; 49 folded triangles |
| e100-p200 | new-2026-08-18 | 33 | 0.603282 | 3.20351 | 4.12568 | 0.0767238 | 16 | 103 | 16 inverted tets; 103 folded triangles |
| e005-p000 | new-2026-08-18 | 32 | 0.60404 | 3.20754 | 8.08607 | 0.112731 | 31 | 43 | 31 inverted tets; 43 folded triangles |
| e005-p200 | new-2026-08-18 | 24 | 0.602016 | 3.19679 | 3.99108 | 0.0702794 | 10 | 80 | 10 inverted tets; 80 folded triangles |
| e025-p100 | reused-2026-08-17 | 25 | 0.604714 | 3.21112 | 4.49928 | 0.0789625 | 12 | 62 | 12 inverted tets; 62 folded triangles |
| no-skin | reused-2026-08-17 | 8 | 0.607827 | 3.22765 | 5.09401 | 0.11813 | 1 | 16 | 1 inverted tets; 16 folded triangles |

## Terminal relative effects

Negative percentages are improvements because all three quantities are minimized.

| comparison | target error change | dihedral change | Laplacian change |
| --- | ---: | ---: | ---: |
| prestrain-only vs baseline | -3.114% | -50.690% | -35.422% |
| softening-only vs baseline | -6.377% | 3.524% | -0.896% |
| combined vs baseline | -9.278% | -50.398% | -37.367% |
| combined vs prestrain-only | -6.362% | 0.591% | -3.012% |
| combined vs softening-only | -3.099% | -52.087% | -36.802% |
| combined vs moderate | -0.835% | -14.441% | -8.978% |
| no-skin vs baseline | -60.303% | 10.539% | 83.206% |

## Common-fidelity relative effects

Negative percentages are improvements because all three quantities are minimized.

| comparison | target error change | dihedral change | Laplacian change |
| --- | ---: | ---: | ---: |
| prestrain-only vs baseline | 0.136% | -51.715% | -35.571% |
| softening-only vs baseline | 0.262% | -5.364% | -5.335% |
| combined vs baseline | -0.074% | -53.290% | -40.983% |
| combined vs prestrain-only | -0.210% | -3.263% | -8.399% |
| combined vs softening-only | -0.335% | -50.643% | -37.657% |
| combined vs moderate | -0.446% | -11.295% | -10.996% |
| no-skin vs baseline | 0.891% | -40.382% | -0.800% |
