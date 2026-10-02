# Fixed-axis refit activation visualization

The saved **400-update fixed-axis refit** is now shown in the same signed meeting style as the full activation modes. The left panel below is the full fit's principal activation; the right is the scalar refit. The refit increases contraction strength in many regions. Its activation is nonnegative along the frozen axis, so its glyphs are red or near-neutral, with no blue extension-like component.

![Full principal mode versus fixed-axis refit](../data/78-fixed-axis-activation/side-context-principal-vs-refit.png)

Standalone refit images: [full face](../data/78-fixed-axis-activation/side-context-fixed-axis-refit.png) and [mouth-corner close-up](../data/78-fixed-axis-activation/region1-mouth-corner-fixed-axis-refit.png).

![Mouth-corner principal mode versus refit](../data/78-fixed-axis-activation/region1-mouth-corner-principal-vs-refit.png)

The refit is also added as the fourth column after full-field modes 1, 2 and 3: [full-face comparison](../data/78-fixed-axis-activation/side-context-four-panel.png), [mouth-corner comparison](../data/78-fixed-axis-activation/region1-mouth-corner-four-panel.png). The three full-field images are reused unchanged from the [signed-mode figures](76-signed-meeting-eigenmodes.md).

## Meaning and scale

Each active tetrahedron has one nonnegative scalar \(s\) and a frozen reference axis \(n\):

$$
B=I+snn^T,\qquad Z=(2s+s^2)nn^T,\qquad s\ge0.
$$

Both transverse activation eigenvalues are zero. This constrains activation, while passive transverse deformation remains possible.

The signed display value is

$$
c=100\frac{s}{1+s}
=100\left(1-\frac1{\sqrt{1+z}}\right),\qquad z=2s+s^2.
$$

The shared color scale remains −100 to +100 and line length is \(4.5\,\mathrm{mm}\;c/100\). In this positive scalar model, \(c/100\) equals the prescribed active shortening along \(n\); it is not the measured total deformation strain. The maximum displayed value is **93.41%**, corresponding to an active axial stretch of **0.06590**. The bounded encoding compresses large strengths, so the raw maximum \(s=14.17415\) is retained in the data.

Each result is shown on its own saved equilibrium shape with the same camera, palette, skin opacity 0.06, and line width 1. Directions are \(Fn/\|Fn\|\), using that result's deformation gradient. Thus apparent direction/location changes include differences in deformation; they do not mean the reference axes were optimized. The visibility mask is recomputed on the refit's deformed muscle regions using the same rule as before.

Of 288,235 active tetrahedra, 2,305 have exactly zero final strength. Matching the preceding figures, the display also suppresses axes whose original full-field top eigenvalue was nearly repeated and coefficients at most 10⁻⁸. This suppresses 26 additional positive-strength ambiguous axes and 2 tiny positive modes, for 2,333 zero-length glyphs in the complete export. The visible full-face view contains 123,289 lines; the mouth-corner view contains 12,529. Suppression affects display only.

## Saved fit and verification

The refit uses [best.npz](../data/42-fixed-directions-400/best.npz), SHA256 `fa592138af874695770ca82696a648db95690ac19caa9a022c8c5ad7a62fa33f`, and [initialization.npz](../data/42-fixed-directions-400/initialization.npz), SHA256 `cc98ced7bbc88a7b4ffcac2a1045708318f2f055be829472c079938c6f8bf760`. Frozen axes match their original byte hash `13ab9a0f940afa916ec16c0df95b3e15ce327556718e1b9cbc2826cc83b36cd9`.

The saved area-weighted fit RMS is **2.168863 mm**, compared with **1.835697 mm** for the full field. Update 400 was still improving and is not a convergence certificate. See the [inverse-refit report](40-fixed-direction-refit.md) for optimization, shape comparison, and mechanical diagnostics. These figures require no additional inverse or equilibrium solve.

The [render receipt](../data/78-fixed-axis-activation/summary.json) records input/output hashes, the inherited metrics, cameras, and source snapshots. The [NPZ](../data/78-fixed-axis-activation/fixed-axis-field.npz) stores all axes, strengths, three effective eigenvalues, deformation gradients, deformed centers, ambiguity flags and display lengths; the [VTP](../data/78-fixed-axis-activation/all-active-fixed-axis-refit.vtp) contains all 288,235 centered line cells.

The [Comet run](https://www.comet.com/liblaf/apple/98955431777b4204bd794a423f8c81f6) exited with code 0. Its [log](../data/78-fixed-axis-activation/run.log) is retained, and the comparisons were visually inspected. The equivalent magnitude formulas agree within 2.22 × 10⁻¹⁶.

Independent read-only verification reconstructed every NPZ/VTP field from the saved refit and rerasterized both visibility masks. All field arrays and visibility evidence agreed exactly, both transverse exported eigenvalues were zero, and line-length geometry agreed within 4.90 × 10⁻¹⁶ m. All six image hashes and dimensions passed. The NPZ and VTP hashes are `cf63e706bcf070a610fcbbba9f8459dfbccbec95376a8301d360ed9742af16be` and `fc38f10835ec4de19ab6bdcdaee4edc7dbae4d56020d3552ba070f5c645a97f9`; the executed source matches its snapshot at `24d25d01c4cf598c4235f2d59c857312de6cb58f3b874779aeac373a3749290e`.

Run [the renderer](../src/78-render-fixed-axis-activation.py) from `${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`, with an empty output destination:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
LIBGL_ALWAYS_SOFTWARE=1 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Fixed-axis refit activation meeting visualization' \
CHERRIES_TAGS='face,activation,fixed-axis,refit,signed,meeting-style' \
.venv/bin/python src/78-render-fixed-axis-activation.py
```
