# Exploratory release from a nonstationary shared-muscle seed (2-D, h = 0.20)

## Status

This is an **exploratory, non-convergence-certified** continuation. The immutable canonical shared-control endpoint did not pass the strict stationarity gate (gradient infinity norm `5.66e-3`, versus `2e-8`). The requested releases were nevertheless run from that endpoint, with the label `NONSTATIONARY/EXPLORATORY` in their receipts. They must not be interpreted as a converged comparison of shared versus independent activations.

No activation bounds, determinant constraints, inversion repair, contact, barrier, or skin energy was added. The model is the 1 x 0.1 band-muscle 2-D pork: Stable Neo-Hookean material, Poisson ratio 0.49, muscle Young's modulus ten times fat, muscle triangle centroids in y = 0.04--0.06, and the parabolic h = 0.20 top target with fixed sides/bottom.

## Protocol and branch audit

The canonical shared result was read, not modified, from `data/20-canonical-h020/h020-shared/final-state.npz` (`a49a6308...7db2c4`). Its three shared activation components were tiled to the 1,200 independent muscle-triangle controls, the Adam moments were reset, and each release used 1,200 Adam updates followed by the same unbounded, strict L-BFGS refinement (forward cap 3,000; strict forward residual tolerance `1e-10`).

The strict shared-state re-solve reproduced the stored state exactly. The tiled independent-control re-solve from that stored displacement also reproduced it exactly: displacement, objective, and all three determinant minima had zero recorded delta. The zero-displacement solve instead landed on a different equilibrium branch: `||du||_inf = 0.235139`, objective lower by `3.87356e-4`, and `min det(F)` lower by `0.0923191`. Thus the two stage-2 runs differ only in their first forward-displacement initialization, not in their tiled controls or optimizer budget.

## Results

| release | first forward state | exact saved states | top RMS error | final gradient inf | minimum det(F) | final double-inverted area | forward/adjoint failures | cumulative cost* |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared-u | strict stored shared displacement | 1,226 | 0.012700 | 3.071e-4 | -363.578 | 3.8% | 0 | 3,877 |
| zero-u | zero displacement | 1,237 | 0.010671 | 3.389e-4 | -294.997 | 3.2% | 0 | 4,121 |

\*Cumulative forward/adjoint cost is canonical shared (1,414) + its nonstationary fallback attempt (1,001) + this release. It includes rejected strict-refinement trials, so it is not a cost-matched comparison to an independent cold start.

Both strict refinements ended at the configured stalled-restart limit, not at the stationarity tolerance. Their strict equilibrium residuals were valid (`5.02e-11` and `2.83e-13`), but their final gradient norms remained far above the strict gate. All forward, adjoint, inverse, nonfinite, and refinement-trial failure counters are zero.

The releases obtain close top fits only alongside severe, explicitly retained artifacts. The shared-u release reaches 4.1% peak inverted cells, 3.95% final A-inverse-negative cells, and `min det(F) = -363.578`. The zero-u release reaches 3.65% peak inverted cells, 3.5% final A-inverse-negative cells, and `min det(F) = -294.997`. In both, A-inverse and double inversions start at saved step 4; G-negative cells are already present at step 0. This is evidence that the close shape fit in these exploratory branches is coupled to folding/extreme deformation, rather than evidence of realistic muscle deformation.

## Visual assets and reproducibility

Every accepted state is a separate VTU frame and is listed consecutively in each `history.vtu.series`; this provides 1,226 and 1,237 frames respectively at one state per 30 FPS video frame. The final VTU and final-state arrays are kept alongside the traces and refinement-trial ledgers:

- `data/40-exploratory-release-h020/h020-shared-release/`
- `data/40-exploratory-release-h020/h020-shared-release_zero_u/`

The runner source is `src/40-run-exploratory-release.py` (`075812cc...f57e51f`). It was compiled and Ruff-checked before execution; the output root is separate from, and does not overwrite, the canonical or fallback receipts. A mixed-source ParaView renderer will provide the common-camera, common-scalar-range videos and final-shape PNG receipts for the four-case comparison.

## Interpretation

The branch test matters: identical tiled activations can select materially different equilibria according to the forward displacement seed. Starting from the zero branch gave a slightly lower final top RMS here, but both paths remain nonstationary and severely inverted. Therefore neither is a valid answer to whether shared-to-full continuation removes bumpiness. The supported conclusion is narrower: at h = 0.20, independent released controls can reduce surface error while driving the muscle band through widespread folding, and equilibrium-branch selection is a confound that must remain fixed and reported in any later stationary comparison.
