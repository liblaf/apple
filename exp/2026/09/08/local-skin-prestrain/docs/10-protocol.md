# Local skin prestrain ablation

This experiment tests whether a small prescribed local skin contraction reduces the three marked cheek and mouth-corner bumps while retaining smile fit and nasolabial-fold shape.

All cases start from the same physical-volume muscle checkpoint at source update 200. The checkpoint is SHA-256 `c2f77e4bdb09e8e0be50d0f8ec9a9926ea7a4bc08db5a368ec83dcb768707b58`. The volume energy, all volume material constants, target, masks, fixed boundaries, and forward/adjoint tolerances remain the canonical physical-volume setup.

| Case | Skin | Local natural-length contraction |
| --- | --- | --- |
| no-skin | None | None |
| skin-zero | Existing Koiter membrane; E = 0.2 MPa, nu = 0.46, thickness = 1 mm | Zero |
| skin-local-1pct | Identical membrane | 0–1%, fixed on material triangles with a smooth 5 mm spatial taper |

The existing Koiter implementation contains membrane energy only, without bending. The skin's rest coordinates, reference area, thickness, and fractions are unchanged. Isotropic contraction `c` is represented by packed inverse activation `[1/(1-c)-1, 1/(1-c)-1, 0]`. The prestrain is prescribed and has no fitted parameters. The values above are controlled experiment settings, not calibrated patient-specific skin measurements.

First, equilibrate all three models at fixed canonical muscle controls with the canonical displacement as the initial guess. Then re-fit muscle controls for each of the three cases for 200 Adam updates. All three branches use newly zeroed Adam moments, freshly evaluated gradients under their own physics, lr = 0.3, eps = 0.01, and betas = (0.9, 0.999). The objective is uniform finite-IsFace coordinate MSE times 1e6. No regularizer, activation clamp, learning-rate change, or local tetrahedron repair is introduced.

Every accepted iteration records global and regional fit, motion, target projection, three marked-region 5 mm high-pass normal residuals and displacements, control change, element determinants, and activation eigenvalues. Full surface displacements are saved every iteration for actual matched-fit visual comparisons; full control/volume states are saved every ten iterations. The latest full optimizer checkpoint is saved atomically after every accepted evaluation.

Bumpiness and fitting error are the primary outcomes. Element inversion and non-SPD activation are diagnostics and do not stop the experiment. Nonfinite states or failed forward/adjoint solves stop visibly before an invalid gradient is used. Completion of the update budget is not a claim of inverse stationarity.

Final comparisons will include fixed-activation results, equal-budget best-fit results, and actual recorded states at comparable fit and motion when the trajectories overlap. Nasolabial fit and fixed cross-sections are assessed separately, because reduced motion or flattening of the real groove must not be counted as a surface-quality improvement. Actual equilibrated membrane stretch and stress signs are measured to assess whether the prescribed contraction creates local tension.

Runs use the active repository interpreter, Cherries local evidence, Comet logging, and disabled automatic Git commits. No changes are made to the shared library or earlier experiment groups.
