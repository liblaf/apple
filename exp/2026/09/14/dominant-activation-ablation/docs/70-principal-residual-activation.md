# Principal and residual activation: three-mode views

Generated on 2026-09-15 as an extension of the existing dominant-activation ablation.

The three views separate the original corrected, unregularized full activation field into its strongest contraction mode and two residual components. The main finding is that the residual is not simply two weaker contractions: the middle component has both signs, while the third component is expansive wherever nonzero in this dataset.

![Three-mode face view](../data/71-eigenmodes/side-context-triptych.png)

## How to read the images

Each short line shows an **unsigned axis** at a sampled muscle tetrahedron. Red means contractile effective activation, blue means expansive effective activation, and gray means near zero. Color encodes the signed coefficient, with one shared symmetric-log scale from −40 to +40 and a linear region around zero of width ±0.01. Equal colors mean equal coefficients across all three images; the same scale is also used in the close-up. No data values are clipped.

Line length is constant within each view and is only a direction marker: it does not encode activation strength. A line pointing toward the camera appears shorter because of projection, so a center dot keeps that location visible. Nearly repeated eigenvalues retain their nonzero colored dots but omit lines whose individual axes are not uniquely determined. Neutral modes with absolute coefficient at most 10⁻⁸ are omitted.

The geometry is the common reference/rest shape. The same geometry-derived tetrahedron sample is reused in all three panels, so sampling cannot favor a particular activation mode. The face view has 3,338 sample locations; the mouth-corner view has 5,627. These are camera-facing samples, while the statistics and exported eigendata cover all 288,235 active tetrahedra.

Individual full-face images: [principal mode](../data/71-eigenmodes/side-context-mode-1-principal.png), [second mode](../data/71-eigenmodes/side-context-mode-2-residual.png), [third mode](../data/71-eigenmodes/side-context-mode-3-residual.png).

![Mouth-corner detail](../data/71-eigenmodes/region1-mouth-corner-triptych.png)

Individual close-ups: [principal mode](../data/71-eigenmodes/region1-mouth-corner-mode-1-principal.png), [second mode](../data/71-eigenmodes/region1-mouth-corner-mode-2-residual.png), [third mode](../data/71-eigenmodes/region1-mouth-corner-mode-3-residual.png).

## What is being decomposed

The corrected physical-volume model uses the effective activation tensor

$$
Z=BB^T-I=\sum_{k=1}^3 z_k n_k n_k^T,
\qquad z_1\ge z_2\ge z_3,
$$

where B is the inverse activation tensor. The reference axes are orthogonal and their signs are arbitrary. Ordering is by algebraic eigenvalue, not by absolute magnitude. Thus the first panel shows the **strongest contraction**, which need not be the largest-magnitude mode in every cell.

$$
Z_{\mathrm{principal}}=z_1n_1n_1^T,
\qquad
Z_{\mathrm{residual}}=z_2n_2n_2^T+z_3n_3n_3^T.
$$

In this specific full field, z₁ is nonnegative and z₃ is nonpositive; those signs are checked by assertions before rendering. The equivalent positive activation stretch is 1/sqrt(1+z): positive z corresponds to contraction and negative z to expansion in the effective metric. This interpretation uses BBᵀ and does not recover signs of eigenvalues of a raw indefinite B. The full field is the same source used to freeze the directions for the scalar inverse experiment; the final scalar refit would have zero residual components by construction.

These are **activation-field images**, not three separately deformed shapes. The tensor components add algebraically, but displacements after mechanical equilibrium are coupled and nonlinear. The plots do not allocate fitting error, deformation or causal work to individual modes, and they do not establish anatomical fiber directions.

## All-cell statistics

Positive/negative/neutral use a coefficient tolerance of 10⁻⁸. Norm shares below use muscle-volume weights w and the ratio sum(w zₖ²) / sum(w ||Z||²).

| Mode | Contractile cells | Expansive cells | Neutral cells | Weighted squared-norm share |
| --- | ---: | ---: | ---: | ---: |
| 1 | 288,172 | 0 | 63 | 88.35% |
| 2 | 161,320 | 126,834 | 81 | 2.41% |
| 3 | 0 | 288,172 | 63 | 9.24% |

Together, modes 2 and 3 contain **11.65% of the muscle-volume-weighted squared Frobenius norm** (13.76% with equal tetrahedron weights). This is a tensor-magnitude statistic, not a share of energy, displacement or fitting performance. The earlier forward ablation already showed that deleting these modes at fixed strengths had a substantial effect on the smile despite that norm fraction.

Near-degenerate axis counts are 90, 97 and 78 for modes 1–3. A gap is considered too small when it is at most 10⁻⁶ max(1, maxₖ|zₖ|). The display omits ambiguous lines; it does not rotate or smooth the field to make it appear more coherent.

## Evidence and reproduction

The source is [baseline-replay.npz](../data/10-forward/baseline-replay.npz), SHA256 `07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201`, with the corrected physical-volume flag verified. The maximum reconstruction error for sum(zₖnₖnₖᵀ) versus saved Z is **2.66×10⁻¹⁴**.

The [render receipt](../data/71-eigenmodes/summary.json) records source/input/image/glyph hashes, display conventions, samples and signed statistics. [eigenmodes.npz](../data/71-eigenmodes/eigenmodes.npz) and [all-active-cell-modes.vtp](../data/71-eigenmodes/all-active-cell-modes.vtp) contain the full field for further 3D inspection. The VTP stores cell-center points with all three axes and coefficients; view-specific line VTPs and sample/visibility NPZs preserve the displayed subset.

The final software render exited with code 0 and was visually checked for legibility, common scale/camera, and distinct modal patterns. Its [Comet record](https://www.comet.com/liblaf/apple/0c7184142d824b5aa7eebee32cb18d21) and [raw log](../data/71-eigenmodes/run.log) retain run metadata. The first render in `data/70-eigenmodes` remains available; the final pass only increases context opacity and glyph widths. Eigenvalues, axes, colors and sample choices are unchanged.

Run from `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
LIBGL_ALWAYS_SOFTWARE=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Principal and residual activation views with clearer context' \
CHERRIES_TAGS='face,activation,eigenmodes,principal,residual,visualization' \
.venv/bin/python src/70-render-eigenmodes.py
```

The destination must be empty; preserve existing results before reproducing. PyVista's notice about a future `extract_surface` default is retained in the log; the current renderer and copied visibility helper are snapshotted with the artifacts. No inverse or forward equilibrium was rerun for these visualizations, and no production-library files were changed.
