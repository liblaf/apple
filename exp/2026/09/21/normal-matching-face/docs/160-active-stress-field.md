# Active stress field of the reference-loss face fits

The update-200 fits with smoothness off/on show very similar activation-induced stress patterns at the shared display scales. The mouth and lower cheek contain visible concentrations and element-to-element variation. The close-up makes the principal stress axes easier to inspect. These are derived fields from the saved fits; no optimization was run for this visualization.

![Full-face stress comparison](../data/156-active-stress-figures/full-active-stress.png)

![Mouth stress comparison](../data/156-active-stress-figures/mouth-active-stress.png)

## Quantity shown

The fitted controls are unrestricted symmetric Raw6 active-strain parameters, not independent active-stress tensors. Set $B=I+\operatorname{sym}(q)$ and let $F$ be the fitted physical deformation gradient. The displayed field is the activation-dependent Cauchy stress difference at the **same** $F$:

$$
Q=f_{\mathrm{muscle}}\mu(BB^T-I),\qquad
\Delta\sigma=\sigma(F,B)-\sigma(F,I)=\frac{FQF^T}{\det F}.
$$

Here $f_{\mathrm{muscle}}$ is the cell's muscle fraction, included to match the mixed-cell energy weighting in the actual fit, and $\mu=0.0100671141$ MPa. This follows by differentiating the fitted physical-volume energy

$$
W=f_{\mathrm{muscle}}\left[\frac{\mu}{2}(\|FB\|_F^2-3)-\mu(J-1)+\frac{\lambda}{2}(J-1)^2\right],\qquad J=\det F.
$$

The volumetric terms cancel in the subtraction. This diagnostic excludes the passive stress contribution and is not the total tissue stress. It is a model prediction, not a physiological measurement. See the [extraction source](../src/150-extract-active-stress.py), [mechanics helper](../src/active_stress.py), and [extraction receipt](../data/150-active-stress/summary.json).

## Reading the figures

- **Columns:** smoothness off, smoothness on, both at update 200 of the selected 2 mm / 5 degree calibrated position-plus-normal fit.
- **Top:** $\|\Delta\sigma\|_F$ in kPa on the exposed active-mesh surface. Colors remain constant within each tetrahedral cell; no spatial smoothing or point interpolation is applied.
- **Bottom:** all three principal stress axes at sampled cells. Orange is a positive contribution and blue a negative contribution. Lines are centered, unoriented eigenvector axes, not anatomical muscle fibers or force arrows. Their length is proportional to the absolute principal value.
- **Shared scales:** magnitude colors saturate at 35.2621 kPa; axis lengths saturate at 4 mm for 24.0637 kPa. Each cap is the pooled per-cell 99th percentile across both fits (all three modes pooled for axis lengths). Raw exports retain all values.
- **Sampling:** the same 4,679 full-view cells and 3,260 close-up cells are used in both branches. One frontmost reference centroid is chosen per projected 2 mm / 0.8 mm bin. This is a sparse reference-selected illustration, not exact occlusion in the deformed geometry. Faint skin and muscle surfaces provide context.
- **Cyan:** invalid active-cell centers, excluded from Cauchy stress evaluation. Each endpoint has 2 active inverted cells and 5 inverted tetrahedra overall. Occlusion can hide or merge markers in these projections. No absolute determinant or clamping is used to make an inverted cell appear valid.

| Diagnostic | Smoothness off | Smoothness on |
| --- | ---: | ---: |
| Active cells | 288,235 | 288,235 |
| Valid positive-J active cells | 288,233 | 288,233 |
| Branch 99th-percentile magnitude, kPa | 35.3165 | 35.2166 |
| Maximum valid magnitude, kPa | 211.1840 | 203.3826 |
| Valid cells above shared color cap | 2,904 | 2,861 |
| Principal modes above shared length cap | 8,672 | 8,622 |

The visual similarity does not establish identical fields or convergence. The fitted smoothness penalty acts on activation parameters, so this visualization alone does not quantify stress smoothness. The invalid elements prevent treating either endpoint as a fully physically valid fit. The [fit report](145-reference-fit-results.md) gives optimization and geometry diagnostics.

## Full fields and verification

Open the complete active tetrahedral mesh in ParaView: [smoothness off VTU](../data/150-active-stress/smooth-off-normal/stress.vtu), [smoothness on VTU](../data/150-active-stress/smooth-on-normal/stress.vtu). These files include global cell IDs, muscle IDs/fractions, $J$, validity, $Q$, $\Delta\sigma$, principal values, and magnitude. Tensor components are in MPa; `MagnitudeKPa` is in kPa. NumPy archives additionally preserve $F$, $B$, and eigenvectors: [off NPZ](../data/150-active-stress/smooth-off-normal/stress-field.npz), [on NPZ](../data/150-active-stress/smooth-on-normal/stress-field.npz).

The frozen fitting sources, material parameters, fixture, and prior verification receipt were checked. An independent float64 autograd evaluation of the energy matched the analytic first Piola stress on 24 deterministic cells per branch: maximum absolute errors were $6.25\times10^{-17}$ and $1.15\times10^{-16}$ MPa. Identity activation gives zero contribution; changing $B$ to $-B$ leaves it unchanged. Full exported eigensystems reconstruct the valid stress tensors within $2.78\times10^{-16}$ MPa. Magnitudes, invalid masks, and extraction/render input hashes were verified. Both composite figures were visually inspected; source Ruff checks passed.

## Reproduction and run evidence

Run from `exp/2026/09/21/normal-matching-face`. Use new output names when reproducing: scripts deliberately fail if an output directory already exists.

```bash
CHERRIES_NAME='Face Raw6 active stress extraction' CHERRIES_TAGS='face,3d,raw6,active-stress,reference-loss,validation' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python src/150-extract-active-stress.py --output 150-active-stress

CHERRIES_NAME='Face reference loss: active stress field visualization final layout' CHERRIES_TAGS='face,raw6,reference-loss,active-stress,visualization' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/155-render-active-stress.py --output 156-active-stress-figures
```

Both production runs exited successfully after Cherries shutdown. Their `Comet.ml Experiment Summary` blocks and exact recorded commands are preserved in the [extraction log](../logs/150-extract-active-stress.log) and [render log](../logs/155-render-active-stress.log). Comet runs: [extraction](https://www.comet.com/liblaf/apple/be35eb66f63549eab952ab228fa15123), [final rendering](https://www.comet.com/liblaf/apple/861c53bcde2b4d169a83de9cd1c2d835). Git metadata reports `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the working tree contains pre-existing research changes, so the recorded source hashes are the relevant provenance. The extraction log has a Comet import-order warning affecting automatic Torch logging; the explicit local validation receipts are present.

The [render receipt](../data/156-active-stress-figures/summary.json) records cameras, samples, caps, hashes, and clipping counts. The first layout is retained under `data/155-active-stress-figures/` with its copied source and [original log](../logs/155-render-active-stress-first-pass.log); the final figures move labels and the colorbar outside the geometry and widen the full-face camera scale by 10%. No fitting code or saved fitted state was modified.
