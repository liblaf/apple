# Principal and residual activation in the meeting figure style

The follow-up [signed figures](76-signed-meeting-eigenmodes.md) include contraction-like and extension-like signs directly in all three panels.

The three panels below use the **corrected, unregularized full activation fit**. They reproduce the supplied meeting figure's pale deformed face, dense centered lines, camera, and common Viridis scale. The reference screenshot supplies the visual style; its smoothness-on result is not the data source for these panels.

![Principal mode and two residual components](../data/74-meeting-eigenmodes/side-context-triptych.png)

| Panel | Full-resolution image | Meaning in this fitted field |
| --- | --- | --- |
| 1 | [Principal mode](../data/74-meeting-eigenmodes/side-context-mode-1-principal.png) | Maximum contraction direction; nonnegative coefficient |
| 2 | [Residual mode 2](../data/74-meeting-eigenmodes/side-context-mode-2-residual.png) | Intermediate direction; coefficient can have either sign |
| 3 | [Residual mode 3](../data/74-meeting-eigenmodes/side-context-mode-3-residual.png) | Extension-like effective mode; nonpositive coefficient |

**The residual is modes 2 and 3 together.** These are three activation fields drawn on the same saved equilibrium shape, not three separately simulated shapes. They show the full field from which the fixed contraction directions were obtained. The subsequent scalar refit has zero residual modes by construction.

## Reading direction, magnitude, and sign

Each centered line marks an unoriented axis. Its length and color encode a common bounded display magnitude:

$$
a_k=1-\frac{1}{\sqrt{1+|z_k|}},\qquad
\ell_k=4.5\,\mathrm{mm}\;a_k.
$$

Viridis displays \(100a_k\) from 0 to 100 in every panel. Thus identical colors and unprojected lengths mean identical coefficient magnitudes across modes. Foreshortening can shorten a line on screen. No independent normalization or minimum line length is applied.

For the positive principal mode, this is the display rule used in the meeting figure. For negative modes it is a magnitude transform, **not physical extension or shortening percentage**. Colors alone therefore do not identify sign. The [full-face sign companion](../data/74-meeting-eigenmodes/side-context-mode-2-sign.png) shows mode 2 in red for positive coefficients and blue for negative coefficients, with the same line lengths.

![Mouth-corner comparison](../data/74-meeting-eigenmodes/region1-mouth-corner-triptych.png)

Individual close-ups: [principal](../data/74-meeting-eigenmodes/region1-mouth-corner-mode-1-principal.png), [mode 2](../data/74-meeting-eigenmodes/region1-mouth-corner-mode-2-residual.png), [mode 3](../data/74-meeting-eigenmodes/region1-mouth-corner-mode-3-residual.png), and [mode 2 signs](../data/74-meeting-eigenmodes/region1-mouth-corner-mode-2-sign.png).

## Decomposition and geometry

For the inverse activation tensor \(B=A^{-1}\), decompose the mechanically relevant effective tensor:

$$
Z=BB^T-I=\sum_{k=1}^3 z_k n_k n_k^T,
\quad z_1\ge z_2\ge z_3,
\quad Z_{\mathrm{res}}=z_2n_2n_2^T+z_3n_3n_3^T.
$$

Eigenvalues are ordered algebraically, not by absolute magnitude. The orthogonal reference axes are drawn at deformed tetrahedron centers with directions \(m_k=Fn_k/\|Fn_k\|\). Transported axes can cease to be orthogonal; they are not generally eigenvectors of the total spatial stress. Positive and negative coefficients correspond to contraction-like and extension-like changes in the effective activation metric.

All three modes share the geometry-based visibility mask: 125,349 candidate tetrahedra in the full-face view and 12,080 in the mouth-corner view. The mask retains tetrahedra whose projected centroid belongs to the frontmost muscle region, including interior tetrahedra of that region. There is no grid sampling. Skin opacity is 0.06 and line width is 1, matching the reference renderer.

Neutral coefficients \(|z_k|\le10^{-8}\) and nonunique axes have zero display length. An axis is considered nonunique when an adjacent eigenvalue gap is at most \(10^{-6}\max(1,\max_k|z_k|)\). The full exported data retain these cells and their flags.

## Measured differences

Statistics cover all 288,235 active tetrahedra, not just the visible subset. Squared-norm shares use weights equal to tetrahedron volume times muscle fraction.

| Mode | Positive | Negative | Neutral | Weighted squared-norm share |
| --- | ---: | ---: | ---: | ---: |
| Principal | 288,172 | 0 | 63 | 88.35% |
| Residual 2 | 161,320 | 126,834 | 81 | 2.41% |
| Residual 3 | 0 | 288,172 | 63 | 9.24% |

The residual contains **11.65% of the muscle-volume-weighted squared Frobenius norm**, or 13.76% with equal tetrahedron weights. This is not a fraction of energy, displacement, or fitting performance. Tensor components add algebraically; shapes obtained after equilibrium do not. Visual coherence alone does not establish anatomical fiber directions.

## Data and reproduction

The source is [baseline-replay.npz](../data/10-forward/baseline-replay.npz), SHA256 `07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201`. Its corrected physical-volume flag is checked before rendering; the underlying model uses physical \(J=\det F\). The modal reconstruction agrees with saved \(Z\) to a maximum absolute error of **2.66 × 10⁻¹⁴**.

The [render receipt](../data/74-meeting-eigenmodes/summary.json) includes hashes, cameras, signed statistics, display rules, and source snapshots. [All-cell eigendata](../data/74-meeting-eigenmodes/eigenmodes.npz) contain raw signed eigenvalues, reference and transported axes, deformation gradients, gaps, flags, and display lengths. Full line meshes are also available for [mode 1](../data/74-meeting-eigenmodes/all-active-mode-1-principal.vtp), [mode 2](../data/74-meeting-eigenmodes/all-active-mode-2-residual.vtp), and [mode 3](../data/74-meeting-eigenmodes/all-active-mode-3-residual.vtp).

The final render exited with code 0; both three-panel comparisons and both sign companions were visually inspected. The [Comet run](https://www.comet.com/liblaf/apple/69e5de4f9e134ae3a4f147d05ffeb88c) and [execution log](../data/74-meeting-eigenmodes/run.log) retain the run metadata. The earlier `73-meeting-eigenmodes` pass is preserved; the final pass fixes clipping in the sign legend. All magnitude PNGs are byte-identical between those passes.

An [independent CPU verification](../data/82-meeting-eigenmodes-verification/receipt.json) passed for the source hash, tensor reconstruction, saved displacement and deformed centers, transported directions, raw coefficients, display mapping, full VTP fields, common visibility masks, and all ten image hashes/dimensions. Exported line lengths agree within 4.88 × 10⁻¹⁶ m. The [verification source](../src/82-verify-meeting-eigenmodes.py) and [Comet record](https://www.comet.com/liblaf/apple/cda90550ae3846ef8f9e4d24213cc68e) retain the checks separately from the renderer.

Run [the renderer](../src/72-render-meeting-eigenmodes.py) from the experiment group, with an empty output destination:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
LIBGL_ALWAYS_SOFTWARE=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Principal and residual activation meeting style final' \
CHERRIES_TAGS='face,activation,eigenmodes,principal,residual,meeting-style' \
.venv/bin/python src/72-render-meeting-eigenmodes.py
```

No equilibrium solve or inverse optimization was rerun to create these images.
