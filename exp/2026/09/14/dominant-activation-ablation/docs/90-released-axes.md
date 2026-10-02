# Releasing directions from the fixed-axis refit

This experiment continues the saved fixed-axis result at update 400 with three learned activation coordinates per active tetrahedron. Its purpose is to test whether the full-tensor-derived directions provide a useful initialization when direction and strength can subsequently change together.

**The warm start improves fitting while largely retaining the initial axis field, but element distortion returns.** All 200 additional updates completed. Uniform fit RMS falls from 2.070724 to **1.092945 mm** (47.22%); area-weighted fit falls from 2.168863 to **1.136017 mm** (47.62%). The final median axis rotation is **0.56°**, the 90th percentile is **4.72°**, and the volume-weighted mean is **2.45°**. The final state has **16 inverted tetrahedra, including 9 active muscle cells**. This is a useful initialization result, not a valid converged reconstruction or anatomical validation.

## Exact conversion and constraints

The fixed model is

$$
B_i=I+s_i n_i n_i^T,\qquad s_i\geq0,\quad \|n_i\|=1.
$$

Initialize a learned vector by

$$
v_i^{(0)}=\sqrt{s_i^{(400)}}\,n_i,
$$

and optimize its three Cartesian components using

$$
B_i(v_i)=I+v_i v_i^T,\qquad
s_i=\|v_i\|^2,\qquad n_i=v_i/\|v_i\|.
$$

For every nonzero vector, this gives two direction degrees of freedom and one strength degree of freedom. The sign of the vector does not matter. The effective tensor is

$$
Z_i=B_iB_i^T-I=(2+\|v_i\|^2)v_i v_i^T.
$$

The active stretch in \(A_i=B_i^{-1}\) is \(1/(1+s_i)\) along the learned axis and exactly 1 along both transverse axes. Thus activation stays uniaxial and contraction-only throughout; the direction itself is no longer frozen. Transverse physical deformation remains possible through mechanical equilibrium.

There are 288,235 active tetrahedra and 864,705 trainable vector coordinates. The 2,305 cells that have exactly zero scalar strength at the fixed-axis endpoint map to exactly zero vectors. Because \(D(vv^T)=0\) at \(v=0\), their gradients remain zero and they cannot reactivate in this parameterization. The run preserves exact conversion instead of adding a perturbation. It asserts that these vectors and their gradients remain zero at every evaluated step.

The converted tensors match the source to floating-point precision: maximum absolute error is \(3.55\times10^{-15}\) for B and \(8.53\times10^{-14}\) for Z. The initial saved displacement is copied exactly. Replaying equilibrium changes displacement by only \(4.09\times10^{-7}\) mm RMS and changes the objective by \(-1.79\times10^{-8}\) mm²/component.

## Physics, optimizer and validation

The mesh, target, fixtures, tissue fractions, materials and solver implementation match the fixed-axis run. The muscle volume terms use physical \(J=\det F\). There is no skin energy or contact, no new regularizer, no strength cap and no inversion barrier. Inversions are measured and reported; they are not prevented by the constitutive energy.

The objective remains uniform Cartesian-component mean squared displacement error on the finite `IsFace` target vertices, multiplied by \(10^6\). Uniform vector fit RMS is \(\sqrt{3L}\) mm; rest-area-weighted fit RMS is a separate diagnostic.

Adam starts with fresh moments in vector coordinates, learning rate 0.3, epsilon 0.01, betas (0.9, 0.999) and zero weight decay. There is no post-step clamp or projection. Scalar Adam moments have no unique transport into the new vector coordinates, so this is a warm start of the physical state, with a new optimizer. Updates 128 and 200 are saved explicitly, in addition to checkpoints every 20 updates and best/last states.

The synthetic packing chain check passes to \(1.78\times10^{-15}\). Before optimization, the complete implicit gradient passes centered finite differences in both radial and tangential directions at perturbation sizes 0.005 and 0.0025. The four relative errors are 0.01185%, 0.01737%, 0.01010% and 0.01220%. Audit solves use forward relative tolerance \(5\times10^{-6}\), absolute tolerance \(10^{-12}\), and adjoint relative tolerance \(5\times10^{-6}\), with the same tight center displacement seed for every perturbation. Production forward tolerances remain \(5\times10^{-4}\) and \(10^{-10}\), with at most 5,000 steps and a 10-step line search.

All forward and adjoint solves must succeed and remain finite. `best.npz` is the lowest-objective successfully solved state, even if it has inversions. `best-noninverted.npz` is the lowest-objective state with no inverted tetrahedra. These two criteria remain separate in plots and analysis.

Direction changes use the unoriented reference-axis angle \(\arccos|n_i\cdot n_i^{(0)}|\). Cells with zero initial or current strength are excluded. The direction graph diagnostic measures differences of unit-axis projectors across shared faces within the same muscle. Neither measure is an anatomical alignment measurement.

## Results

| State | Uniform fit RMS (mm) | Area-weighted fit RMS (mm) | Median / 90th-percentile rotation | Inversions, all / active | Minimum det(F) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-400, converted and replayed | 2.070724 | 2.168863 | 0° / 0° | 0 / 0 | 0.049018 |
| Released update 37, best without inversions | 1.693337 | 1.771716 | 0.115° / 1.425° | 0 / 0 | 0.000855 |
| Released update 128 | 1.254430 | 1.305614 | 0.374° / 3.640° | 8 / 1 | -0.181579 |
| Released update 200, best and last | **1.092945** | **1.136017** | **0.560° / 4.719°** | **16 / 9** | **-0.376984** |

The step-0 angles are zero analytically; floating-point evaluation gives percentiles of order \(10^{-6}\) degrees. The final 99th-percentile rotation is 12.09°. Only 0.95% of compared active muscle volume rotates more than 15°. All 2,305 initial zero vectors remain zero.

All 201 evaluated objectives decrease strictly. Best and last both occur at update 200. The objective still falls by **16.57% over the last 50 updates**, and the vector-gradient RMS retains **25.36% of its initial value**. The run ends at the declared budget, without a stationarity claim. Audit and optimization take 25.95 minutes, excluding startup and subsequent rendering/verification.

The first inversion occurs at update 38; the first active-cell inversion occurs at update 96. Independent reconstruction identifies the first inverted tetrahedron as global cell 522639, a fat-only inactive cell. The final 16 inversions comprise 9 active mixed-tissue cells and 7 inactive fat-only cells; no pure-muscle tetrahedra are inverted. Full global/original cell IDs, tissue fractions and control labels are recorded in the verification receipt. The best state without inversions is update 37. Its minimum determinant of 0.000855 already indicates a nearly collapsed element, so absence of negative determinants should not be read as good element quality.

### Shape and activation

![Full tensor, fixed axes, and released axes](../data/92-released-axes-visualization/side-context-shape-comparison.png)

![Mouth-corner shape comparison](../data/92-released-axes-visualization/region1-mouth-corner-shape-comparison.png)

Columns show the original corrected full-tensor replay, fixed-axis update 400, and released-axis update 200. Cameras, scale, flat shading and gray material are identical. The released result shows more local surface irregularity around the cheek and mouth. These surface images do not locate interior inverted tetrahedra; inversion counts come from the full volume mesh. The original full-tensor reference also has one inverted active mixed-tissue tetrahedron (global cell 573586), with 0.09765625% muscle and 99.90234375% fat. The five-state audit on 2026-09-16 corrected the earlier "non-active" description of this cell.

![Fixed versus released activation](../data/92-released-axes-visualization/side-context-fixed-vs-released-glyphs.png)

![Fixed versus released activation around the mouth](../data/92-released-axes-visualization/region1-mouth-corner-fixed-vs-released-glyphs.png)

The left panel is fixed-axis update 400 and the right is released-axis update 200. Both use the prior meeting style and shared signed color scale from -100% to 100%. For these uniaxial states the displayed magnitude is \(100s/(1+s)\), which is nonnegative, so only the contraction side of the palette is populated. Glyph lengths are 4.5 mm times \(s/(1+s)\). Lines follow the transported axes \(\operatorname{normalize}(F n)\) on each state's own deformed geometry; differences in screen direction therefore include changes in F. Reported rotation angles compare n in the reference configuration and isolate the learned axis change.

Visibility is recomputed on the released geometry. The fixed panel reuses the saved prior visualization, including its recorded source-axis ambiguity filter; the released panel shows every visible nonzero learned control. These are spatial context views rather than an exact one-to-one line-count comparison. Saved field arrays retain all active tetrahedra.

![Fit, surface detail, inversions and direction history](../data/92-released-axes-visualization/optimization-history.png)

The same-muscle shared-face axis-projector jump RMS rises from 0.261255 to 0.275233, only **5.35%**. In contrast, the C jump RMS rises from 0.402740 to 1.097903 and the Z jump RMS from 2.888066 to 53.928653; these include activation magnitude. Stronger tensors can therefore become much less uniform even while their unit directions change relatively little.

On the predefined primary-region union, normal-displacement high-pass RMS at a 5 mm scale rises from **0.398485 to 0.453311 mm** (+13.76%). Normal target-residual high-pass RMS rises slightly, from **0.182963 to 0.185092 mm** (+1.16%). Thus the increasing surface detail does not correspond to improved target detail under this diagnostic. These local normal high-pass measures differ from the global Cartesian fitting objective.

Maximum scalar strength increases from 14.1742 to 110.6365. The minimum prescribed active axial stretch decreases from **0.06590 to 0.008958**, equivalent to 99.10% prescribed shortening at the most extreme cell. This is an eigenvalue of A, not a measurement of total physical tissue compression; no physiological amplitude range was imposed.

![Best fit versus best without inversions](../data/92-released-axes-visualization/side-context-best-vs-best-noninverted-shapes.png)

The final best-fit state remains the primary reported result. The separate update-37 panel records the best objective attained before any inverted tetrahedra, with its near-collapse limitation stated above.

### Independent verification

**Independent CPU verification passed with exit code 0.** The [receipt](../data/94-released-axes-verification/receipt.json), SHA256 `ed44c7f84ababd5f7aeb89ea1b705a9f9d1c2b80d1caa742256136f36c39bd38`, confirms exact conversion of v and u, all 201 trace rows and CSV/JSON agreement, selected checkpoint metrics and determinants, rank-one tensor relations, unchanged zero controls and moments, Adam counter 200, and matching best/last checkpoint arrays. It also checks the four saved finite-difference receipts and exact source identity against the executed fixed-axis and released-axis snapshots. This verification recomputes saved-state geometry and algebra; it does not rerun equilibrium or the gradient audit.

At update 200, the independent Z-axis action residual is \(3.64\times10^{-12}\) on a tensor scale of \(1.18\times10^4\), or \(3.08\times10^{-16}\) normalized. Unit-axis and transverse orthogonality errors stay below \(3.4\times10^{-16}\). Absolute-plus-relative tensor tolerances account for the larger activation scale. Near-zero angle comparisons explicitly allow 2 microdegrees because normalization followed by arccos is ill-conditioned at zero; materially nonzero angles retain strict comparisons.

Three preliminary verifier attempts have logs preserved under `data/94-released-axes-verification-failed-*`. Their exact executed verifier sources remain available as verified Comet source-code assets: [attempt 1](https://www.comet.com/liblaf/apple/93f70573844a48a8bf195aeb55ec6792), [attempt 2](https://www.comet.com/liblaf/apple/c7f4c4d5624c4d148cc1eb51bb286832), and [attempt 3](https://www.comet.com/liblaf/apple/864772352c7140edbafb710dc7fa3e1a). Two stopped on nanodegree discrepancies in a near-zero angle check; the third passed numerical checks but stopped on a legacy GROUP-relative snapshot path. The final pass corrects only these verifier issues, and its executed verifier source is also snapshotted locally. The optimization code and outputs were not changed.

The renderer also completed with exit code 0; all 15 PNGs reopen at their expected dimensions and were visually inspected. Its largest best-state Z reconstruction discrepancy is \(2.27\times10^{-13}\) absolute and \(8.71\times10^{-16}\) scaled. The source, input and output hashes, visible-cell masks and rendering settings are recorded in its summary. Cherries Local emitted a nonfatal internal log-copy error during rendering; the complete process log was copied into the output directory and the generated artifacts were checked directly.

## Interpretation limits

The fixed-axis update-400 state was still improving, and there is no matched additional 200-update frozen-axis control in this experiment. Any fit improvement after continuation therefore combines extra optimization, a different parameterization and optimizer initialization, and released direction variables. It cannot be assigned entirely to direction freedom.

The historical learned-axis image at update 128 used random muscle-wise starting directions, a different learning rate, and a smoothness term. A comparison to that image is descriptive; it does not isolate initialization as the sole cause. A finite trajectory that stays near its initial axes supports their usefulness as an optimization starting point, but does not establish uniqueness, convergence, or anatomical accuracy.

## Reproduction and artifacts

- Runner: [90-release-fixed-axes.py](../src/90-release-fixed-axes.py).
- Fixed input: `data/42-fixed-directions-400/best.npz`, SHA256 `fa592138af874695770ca82696a648db95690ac19caa9a022c8c5ad7a62fa33f`.
- Frozen reference-axis bytes: SHA256 `13ab9a0f940afa916ec16c0df95b3e15ce327556718e1b9cbc2826cc83b36cd9`.
- Protocol, gradient audit, source snapshots, trace and state files: [data/90-released-axes](../data/90-released-axes).
- Renderer and standalone images: [92-render-released-axes.py](../src/92-render-released-axes.py), [visualization summary](../data/92-released-axes-visualization/summary.json).
- Independent verifier and receipt: [94-verify-released-axes.py](../src/94-verify-released-axes.py), [receipt.json](../data/94-released-axes-verification/receipt.json).
- [Optimization on Comet](https://www.comet.com/liblaf/apple/064dd89a337b4e4aa2b178c71dd82c93).
- [Rendering on Comet](https://www.comet.com/liblaf/apple/f06b03876966470facbd17e391f8aace).
- [Verification on Comet](https://www.comet.com/liblaf/apple/3b3cb53ca558426390964d449cd45adc).

Run from this experiment group with the existing project virtual environment, the worktree `src` directory on `PYTHONPATH`, and four CPU threads:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
CHERRIES_NAME='Release activation directions from fixed-axis update 400' \
CHERRIES_TAGS='face,activation,learned-axis,warm-start,physical-volume,continuation' \
.venv/bin/python src/90-release-fixed-axes.py
```
