# Refined-forward validation

The clean target was generated on a 48x20x48 mesh and restricted at exactly coincident nodes to the 24x10x24 inverse mesh. The inverse used 240 projected L-BFGS steps and fixed magnitude and smoothness weights of 0.01 for F-MS.

The coarse activation field was transferred to the refined mesh as `nested_parent_tet_exact`. The containment audit found 0 of 55296 fine active tetrahedra crossing their centroid-selected coarse parent.

| Method | Coarse fit RMS / D | Fine replay error RMS / D | Fine error HP / D | Fine min det(F) | Fine branch difference / D |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw6 | 0.00327641 | 0.129704 | 0.016882 | 0.751148 | 2.67631e-05 |
| F-MS | 0.0192298 | 0.0347772 | 0.0039718 | 0.978468 | 9.85533e-07 |

The fine replay uses a fresh rest-start equilibrium for the reported errors. The branch column compares that solution with a solve initialized from the refined clean target.
