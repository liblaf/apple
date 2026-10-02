# Signed principal and residual activation

The [fixed-axis refit comparison](78-fixed-axis-activation.md) adds the 400-update scalar refit in the same style and scale.

The three meeting-style panels now encode sign directly: **red is contraction-like, blue is extension-like, and gray is near zero**. They use the same corrected, unregularized full fit and the same deformed line geometry as the preceding magnitude-only figures.

![Signed principal and residual modes](../data/77-signed-meeting-eigenmodes/side-context-triptych.png)

Individual full-resolution panels: [principal](../data/77-signed-meeting-eigenmodes/side-context-mode-1-principal.png), [residual mode 2](../data/77-signed-meeting-eigenmodes/side-context-mode-2-residual.png), [residual mode 3](../data/77-signed-meeting-eigenmodes/side-context-mode-3-residual.png).

The first mode is contraction-like throughout its nonzero field. The middle mode contains both signs. The third is extension-like throughout its nonzero field. **Modes 2 and 3 together form the residual tensor.**

![Signed mouth-corner detail](../data/77-signed-meeting-eigenmodes/region1-mouth-corner-triptych.png)

Individual close-ups: [principal](../data/77-signed-meeting-eigenmodes/region1-mouth-corner-mode-1-principal.png), [mode 2](../data/77-signed-meeting-eigenmodes/region1-mouth-corner-mode-2-residual.png), [mode 3](../data/77-signed-meeting-eigenmodes/region1-mouth-corner-mode-3-residual.png).

## Color and length

For the ordered eigenvalues of \(Z=BB^T-I\), the color scalar is

$$
c_k=100\,\operatorname{sign}(z_k)\left(1-\frac{1}{\sqrt{1+|z_k|}}\right).
$$

All panels use the same diverging color scale from −100 to +100. Color strength increases with coefficient magnitude; the neutral gray retains contrast against the pale face. Line length remains \(4.5\,\mathrm{mm}\,|c_k|/100\), with the previous suppression of neutral and nonunique directions. Camera, visible tetrahedra, skin opacity, and line width are unchanged. Projection can shorten lines on screen.

The sign belongs to the eigenvalue, not to an arrow direction. Flipping an eigenvector leaves both its centered axis and its color unchanged.

The percentage is a **display transform, not physical strain**. Negative coefficients approach −1, corresponding to approximately −29.3 on this display scale; that does not mean 29.3% extension. These field images do not allocate displacement, energy, or fitting performance to individual modes. The [preceding report](72-meeting-eigenmodes.md) explains tensor decomposition, reference-to-deformed axis transport, degeneracy, and all-cell statistics.

## Evidence and reproduction

The renderer reuses the previously [verified](../data/82-meeting-eigenmodes-verification/receipt.json) full line meshes, with input hashes checked before rendering. It checks eigenvalue signs against the new colors, magnitude agreement with the old scale, tetrahedron IDs, and unchanged visible line counts. No mechanical solve was rerun.

The [render receipt](../data/77-signed-meeting-eigenmodes/summary.json) records exact inputs, sources, palette, cameras, and image hashes. [Signed color values](../data/77-signed-meeting-eigenmodes/signed-color-values.npz) retain cell/control IDs, mode indices, raw eigenvalues, and the displayed values. The large source meshes remain in the previous output directory and are referenced by hash.

The final [Comet run](https://www.comet.com/liblaf/apple/026b140602d8439581c44b0ad2ac5e06) exited with code 0; its [log](../data/77-signed-meeting-eigenmodes/run.log) is retained. Both triptychs were visually checked. Input hashes, signed values, and all eight PNG dimensions passed verification. Display ranges are [0, 84.11], [−29.27, 53.91], and [−29.29, 0] for modes 1–3. The initial white-centered palette remains in `76-signed-meeting-eigenmodes`; the final pass changes only palette and caption formatting, and the numerical color values are identical.

Run [the source](../src/76-render-signed-meeting-eigenmodes.py) from `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`, with an empty output destination:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
LIBGL_ALWAYS_SOFTWARE=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Signed principal and residual meeting figures final' \
CHERRIES_TAGS='face,activation,eigenmodes,signed,meeting-style' \
.venv/bin/python src/76-render-signed-meeting-eigenmodes.py
```
