# Fixed-direction scalar-strength inverse refit

This experiment asks whether the target can be recovered after reducing every active tetrahedron to one scalar degree of freedom while keeping its strongest effective contraction direction fixed. It starts from the dominant-only forward ablation, optimizes only the nonnegative axial strengths, and leaves the mesh, target, boundary conditions, tissue fractions, materials and equilibrium implementation unchanged.

**The fixed directions retain most of the expression after scalar refitting.** With exactly 288,235 nonnegative scalar controls and all reference axes frozen, 400 Adam updates reduce area-weighted target-fit RMS from **3.351887 to 2.168863 mm**, compared with **1.835697 mm** for the full tensor field. Motion reaches **91.43%** of the full-field result, and the final mesh has zero inversions. This supports using the learned axes as a useful reduced control model for this smile; it does not establish anatomical accuracy or convergence.

## Inputs and provenance

The experiment uses the corrected physical-volume forward states from this experiment group:

- Full fitted activation replay: `data/10-forward/baseline-replay.npz`, SHA256 `07efff9f6a96ff7d4556df723f6f6386c31c111ad89ac65ebff21ded82050201`.
- Dominant-only equilibrium and displacement seed: `data/10-forward/dominant-only.npz`, SHA256 `23c4adada66e02bd7caef69114dd13f0902dee7bb775465446093456ff2985e6`.
- Frozen volume mesh: `volume.vtu`, SHA256 `238962d0d27a2d35b6211a7a60204d362187b4190dfc7591756bc73ea26ff3b6`.
- Frozen skin mesh: `skin.vtp`, SHA256 `79eed2a5e2b5f23e84287fe729989ab356cb2780e3342b722b074a9231297833`.
- Frozen fixture summary: `summary.json`, SHA256 `161daf3cd2d99d184ed7c7db2ddee18ed8f53f10bf9d9e0ef66b4c642310d081`.

The fixture files come from `exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/` in the original Apple checkout. The two saved states and all 288,235 active-cell identifiers are checked against that fixture before optimization. The frozen direction array has SHA256 `13ab9a0f940afa916ec16c0df95b3e15ce327556718e1b9cbc2826cc83b36cd9`.

The full replay is retained only as a target-fit reference and as the source of the fixed axes. Its measured reference errors are 1.763031 mm uniform vector RMS and 1.835697 mm rest-area-weighted vector RMS. The inverse state starts from the saved dominant-only equilibrium.

## Exact one-degree-of-freedom model

For each active tetrahedron $i$, define

$$
Z_i^{\mathrm{full}}=B_i^{\mathrm{full}}(B_i^{\mathrm{full}})^T-I.
$$

Let \(\lambda_{i,1}\) be its largest eigenvalue and \(n_i\) a corresponding unit eigenvector in the reference configuration. The inverse fit freezes \(n_i\) and optimizes one scalar \(s_i\):

$$
B_i(s_i)=I+s_i n_i n_i^T,\qquad s_i\ge 0.
$$

Equivalently,

$$
Z_i(s_i)=B_iB_i^T-I=(2s_i+s_i^2)n_i n_i^T.
$$

Thus the active stretch in \(A_i=B_i^{-1}\) is \(1/(1+s_i)\) along \(n_i\) and 1 in both transverse directions. The sign of \(n_i\) is immaterial because the model uses \(n_i n_i^T\). This is exactly 288,235 trainable scalars rather than six independent symmetric entries per active tetrahedron.

The initialization reproduces the dominant-only effective tensor:

$$
s_i^{(0)}=\sqrt{1+\max(\lambda_{i,1},0)}-1.
$$

There are 63 cells for which the full tensor has no positive largest mode, so their initial strength is zero. Their numerically selected largest-eigenvalue axes remain available to the optimizer, but those axes are not supported as inferred positive contraction directions by the original full fit. 90 cells meet the recorded near-repeated criterion

$$
\lambda_{i,1}-\lambda_{i,2}
\le 10^{-6}\max(1,|\lambda_{i,1}|),
$$

which includes all 63 zero-initial-strength cells, so an individual principal axis is poorly determined inside the nearly repeated eigenspace. These are interpretation limits, not extra constraints or excluded cells.

## Corrected physical-volume mechanics

The active muscle density is

$$
W(F,B)=\frac{\mu}{2}\left(\lVert FB\rVert_F^2-3\right)
-\mu(J-1)+\frac{\lambda_0}{2}(J-1)^2,
\qquad J=\det F.
$$

The correction is that both determinant terms use the physical deformation Jacobian \(\det F\), not \(\det(FB)\). Activation changes the squared-norm term through \(FB\), while the finite volume penalty measures the actual tetrahedron volume. This model is compressible and has no inversion barrier; determinant and inversion counts therefore remain diagnostics rather than enforced constraints.

The material convention is the one recorded by the source fit: stable and active stable materials receive classical Lamé \(\lambda\), without the later plus-\(\mu\) correction. Muscle uses \(E=0.03\) MPa and \(\nu=0.49\), fat uses \(E=0.003\) MPa and \(\nu=0.49\), and aponeurosis uses \(E=0.1\) MPa and \(\nu=0.35\). Skin thickness is recorded as 1 mm, but skin energy is zero; contact is disabled.

## Optimization and measurements

Adam runs for a budget of 200 updates with learning rate 0.3, \(\epsilon=0.01\), \((\beta_1,\beta_2)=(0.9,0.999)\), zero weight decay and no AMSGrad. After each proposal, strengths are projected with \(s\leftarrow\max(s,0)\). Adam moments are preserved across this projection. There is no upper bound and no loss-based proposal rejection.

The primary comparison retains the 200-update endpoint. Before that endpoint was reached, a bounded continuation policy was chosen: restore the exact saved scalar state, cached gradient, displacement and Adam moments into a new output directory, first to update 300 and then at most 400. Continue at each boundary only if the best iterate is among the last 20 evaluations, objective improvement over the last 50 evaluations is at least 1%, all forward/adjoint solves succeeded, no active or pure-muscle cells are inverted, and total inversions do not exceed the full reference count of one. This is an exploratory extra-budget check, not an equal-update comparison or a convergence guarantee. No optimizer hyperparameters change during continuation.

Each objective evaluation solves mechanical equilibrium with at most 5,000 PNCG steps, relative tolerance \(5\times10^{-4}\), absolute tolerance \(10^{-10}\), and a 10-step line search. Nonfinite values or unsuccessful forward or adjoint solves stop the run visibly. The last valid optimizer checkpoint and the best successfully solved state are retained.

The optimized objective is

$$
L=10^6\operatorname{mean}_{v\in\mathrm{finite\ IsFace},\,k\in\{x,y,z\}}
\left(u_{v,k}-u^{\star}_{v,k}\right)^2,
$$

reported in mm\(^2\) per Cartesian component. It uses uniform weights over the finite `IsFace` target vertices and contains no magnitude, smoothness, determinant or other regularizer. Its corresponding uniform vector RMS obeys

$$
\operatorname{RMS}_{\mathrm{uniform}}=\sqrt{3L}\ \mathrm{mm}.
$$

The rest-skin lumped-area-weighted fit RMS is reported separately and is not the training objective. The trace also records area-weighted motion RMS, target projection, raw and projected gradient norms, KKT residuals, strength changes, active axial stretches, equilibrium receipts, \(\det F\) statistics and fixed-boundary error. The projected-gradient diagnostic uses the unit-step mapping

$$
g_{\mathrm{map}}=s-\max(s-g,0).
$$

Exhausting 200 updates is a finite optimization budget, not a stationarity certificate. Final claims must use the saved best solved state for fit and also report the last-state projected-gradient and KKT diagnostics.

## Gradient validation

The final pre-optimization audit passed. A direct PyTorch chain check verifies the six-entry historical packing of \(s_i n_i n_i^T\) to machine precision: its maximum gradient error is \(8.88\times10^{-16}\). It also explicitly confirms nonzero derivatives at exact zero strength, which rules out a hidden zero-Jacobian problem in this scalar parameterization.

The complete implicit gradient was then compared with independent centered finite differences along two structured directions: a muscle-wise signed multiplicative direction and a gradient-sign multiplicative direction. Each direction used \(\epsilon=0.005\) and 0.0025. Every perturbed equilibrium started from the identical, tightly re-equilibrated audit-center displacement. The audit center and perturbed solves used tighter forward tolerances, relative \(5\times10^{-6}\) and absolute \(10^{-12}\), with adjoint relative tolerance \(5\times10^{-6}\).

The four adjoint-versus-centered-difference relative errors were 0.07394%, 0.14933%, 0.007843% and 0.004866%, all well within the declared 2% slope and two-epsilon consistency thresholds. Moving from the production tolerance to the tight audit center changed the full gradient by 0.04193% in relative norm and the objective by \(-7.74\times10^{-5}\) mm\(^2\).

Preliminary coarse finite-difference checks failed because the centered slope subtracts two nearby equilibrium objectives while the inner equilibrium and adjoint were still solved at production tolerance. Solver error at that accuracy was large enough relative to the small loss difference to spoil a slope comparison. Tightening the audit solves, fixing every perturbation to the same displacement seed, and checking two step sizes removed that numerical ambiguity. The multiplicative directions deliberately leave initially zero cells at zero so a centered difference does not cross the constraint boundary; the separate analytic chain check covers their exact-zero derivative. The passing audit therefore validates the implemented gradient on the tested directions and branch, rather than every coordinate or every possible equilibrium branch.

## Results

| State | Objective (mm²/component) | Uniform fit RMS (mm) | Area-weighted fit RMS (mm) | Area-weighted motion RMS (mm) | Inversions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full fitted tensor replay | 1.036093 | 1.763031 | 1.835697 | 4.150593 | 1 |
| Dominant-only start, re-equilibrated | 3.671581 | 3.318847 | 3.351887 | 2.101848 | 0 |
| Fixed axes: 200 updates | 1.903618 | 2.389739 | 2.484848 | 3.292439 | 1 |
| Fixed axes: 300 updates | 1.607785 | 2.196214 | 2.294994 | 3.589897 | 0 |
| Fixed axes: 400 updates | 1.429300 | 2.070724 | 2.168863 | 3.794709 | 0 |

The start is a fresh equilibrium evaluation of the saved dominant-only state. Its area-weighted fit differs by 0.000679 mm from the previous ablation’s saved 3.351208 mm because the equilibrium was replayed. The shape panel uses that original saved ablation geometry; the optimization curve uses its re-equilibrated step 0. All comparisons retain the same target, mesh, material and boundary conditions.

The final refit recovers **85.08% of the observed objective gap** and **78.03% of the observed area-RMS gap** introduced by removing the other effective activation modes. These are gap-recovery fractions relative to the existing full-field result, not fractions of explained shape variance. The final area-weighted error remains 0.333166 mm (18.15%) above that reference.

![Final shape comparison](../data/51-fixed-directions-400/composites/side-context.png)

![Final mouth-corner comparison](../data/51-fixed-directions-400/composites/region1-mouth-corner.png)

The four columns are target / full fitted tensor / dominant-only start / fixed-axis scalar refit at update 400. They use identical frozen cameras, flat shading and physical geometry at displacement scale 1. Mouth opening and mouth-corner pull recover substantially. Remaining mismatch is visible in the mouth-corner, lip and adjacent-cheek contours. These figures do not imply that the raw unregularized surface is smooth.

![Loss and fit history](../data/51-fixed-directions-400/history/loss-and-fit.png)

All **401 evaluated objectives decrease strictly**; best and last are both update 400. The unit-step projected-gradient RMS falls from 4.55852e-05 to 1.27966e-05, retaining **28.07%** of its initial value. The last 50 updates reduce the objective by **5.26%**. Therefore the run stopped at its declared 400-update cap, not at stationarity, and the remaining fit gap cannot be assigned entirely to the fixed axes. The saved Adam counter and moments support further continuation if desired.

The 200-update row matches the source fit’s update count; it does not match physical step length, computational cost, or optimizer conditioning across the six-parameter and scalar spaces. The 300/400 rows are extra-budget continuations with the same learning rate, epsilon, moments, constraints and fixed axes. The recorded cumulative audit/optimization time is 37.24 minutes, excluding startup and final rendering.

### Contraction strengths and physical diagnostics

The refit uses stronger axial actuation than its initialization. These stretches belong to the prescribed active tensor A, not the measured total deformation F. Their values are descriptive; no physiological range or cap was imposed.

| Natural axial stretch statistic | Initial | Update 400 |
| --- | ---: | ---: |
| Minimum | 0.158907 | 0.065902 |
| Cell-wise 1st percentile | 0.442398 | 0.299868 |
| Cell-wise median | 0.962216 | 0.929141 |
| Muscle-volume-weighted 1st percentile | 0.392838 | 0.256792 |
| Muscle-volume-weighted median | 0.932124 | 0.874742 |
| Active-muscle volume with stretch < 0.5 | 3.631% | 12.383% |
| Active-muscle volume with stretch < 0.65 | 11.353% | 24.107% |

The strongest final prescribed shortening is 93.41%. This large amplitude tail limits a physiological interpretation of a good shape fit. Final strengths range from 0 to 14.174152; 2,305 cells end at zero, including all 63 cells with no positive initial mode.

The final physical determinant range is **0.049016–2.573123**, with zero total, active and pure-muscle inversions. At most one non-active element was inverted during optimization (first seen at update 21, last seen at 286); the full-field reference also contains one inversion. The final active-muscle-volume-weighted RMS of det(F)−1 is 0.019320, and fixed-boundary error is exactly zero. Absence of final inversions does not mean exact incompressibility or contact validity.

### Independent verification and evidence

CPU-only checks passed for [update 200](../data/60-verification/receipt.json), [update 300](../data/61-verification-300/receipt.json), and [update 400](../data/62-verification-400/receipt.json). They independently reconstruct B and Z, confirm one scalar per active tetrahedron, verify frozen axes against the original largest-eigenvalue projectors, recompute fit/motion/determinants, and authenticate optimizer counters and continuation ancestry. Each child’s inherited trace prefix matches its parent exactly. Final recomputed core metrics differ by at most 2.22×10⁻¹⁶; gradient diagnostics agree exactly.

The [final protocol](../data/42-fixed-directions-400/protocol.json), [trace](../data/42-fixed-directions-400/trace.csv), [best state](../data/42-fixed-directions-400/best.npz), and [optimizer checkpoint](../data/42-fixed-directions-400/optimizer-latest.pt) retain the result and exact executed source snapshots. The NPZ stores scalars and displacements; reconstruct the tensor using the axes in [initialization.npz](../data/42-fixed-directions-400/initialization.npz). The [render receipt](../data/51-fixed-directions-400/summary.json) records input/output hashes and exported skin meshes.

Comet runs: [200 updates](https://www.comet.com/liblaf/apple/bd4b8601bd4f4aee9b5795f17c0ba666), [continuation to 300](https://www.comet.com/liblaf/apple/b3b0c803ffe74e37b29356dd9141a100), [continuation to 400](https://www.comet.com/liblaf/apple/8ce8060f959e41079f2295ab20ff044e), [final verification](https://www.comet.com/liblaf/apple/43eff2f7177d4aa887e204f92828dd6d). The two unsuccessful pre-optimization gradient audits are retained in `data/39-coarse-gradient-audit` and `data/39b-refined-gradient-audit`; neither performed inverse updates.

The final side, mouth-corner and history figures were visually inspected. [Final rendering](https://www.comet.com/liblaf/apple/d9fb78a94c1e4847ba6d3a356c66de5f) completed with exit code 0. A subsequent renderer-source change only removed an unused lint suppression; the executed source snapshot recorded in the render receipt remains authoritative.

The verification process completed despite a nonfatal Cherries local-log asset-copy warning. The renderer’s CLI output-directory override also left the default output path registered for Comet upload; therefore the local figure directory was not uploaded, but all local assets and hash receipts are complete. All numerical receipts, source snapshots and Comet records are available. No production-library files or original-checkout files were changed; no commit or push was made.

## Reproduction

Run from:

`${APPLE_HISTORICAL_WORKTREE}/exp/2026/09/14/dominant-activation-ablation`

Use an empty output directory because the entrypoint rejects a nonempty destination:

```bash
env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
  CHERRIES_NAME='Refit fixed activation directions' \
  CHERRIES_TAGS='face,activation,fixed-direction,inverse,physical-volume' \
  .venv/bin/python \
    src/40-inverse-fixed-directions.py \
    --output-dir data/40-fixed-directions-reproduction
```

To continue from a checkpoint, use a new empty destination and a total step budget larger than the saved step. The checkpoint restores strengths, cached gradient, displacement seed, Adam moments and counter, best state, trace and elapsed-time offset:

```bash
env \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=${APPLE_HISTORICAL_WORKTREE}/src \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  COMET_AUTO_LOG_ENV_DETAILS=false COMET_AUTO_LOG_GIT_PATCH=false \
  CHERRIES_NAME='Resume fixed activation direction refit' \
  CHERRIES_TAGS='face,activation,fixed-direction,inverse,resume,physical-volume' \
  .venv/bin/python \
    src/40-inverse-fixed-directions.py \
    --resume data/40-fixed-directions/optimizer-latest.pt \
    --output-dir data/40-fixed-directions-resumed \
    --steps 300
```

For the final continuation, the actual inverse arguments were `--resume data/41-fixed-directions-300/optimizer-latest.pt --output-dir data/42-fixed-directions-400 --steps 400`. Use a different empty output directory when reproducing it. Final verification used `src/60-verify-fixed-directions.py --inverse-dir data/42-fixed-directions-400 --expected-steps 400 --output-dir data/62-verification-400`. Final rendering used `src/50-render-fixed-directions.py` with `--initialization`, `--best`, `--last`, `--trace` and `--refit-summary` pointing into `data/42-fixed-directions-400`, and `--output-dir data/51-fixed-directions-400`; `LIBGL_ALWAYS_SOFTWARE=1` retained software rendering.

The recorded production environment is Python 3.14.6, Torch 2.12.0+cu130 and an NVIDIA GeForce RTX 4090 at Git commit `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. `data/40-fixed-directions/protocol.json` records input and source hashes, while `gradient-validation.json` retains the independent solve receipts and finite-difference values.

## Scope and limitations

This is an in-sample refit to the same smile target that supplied the original full activation. It tests representational and finite-optimization performance for one target, one mesh, one material law, one set of fixed axes, one initialization and one equilibrium branch. It does not test held-out expressions, subjects, anatomy, noise robustness or transfer to another discretization.

The directions were extracted from the original fitted tensor field, so a successful refit would show that one scalar per frozen effective axis is sufficient for this target under this setup; it would not independently identify anatomical fiber directions. A worse finite-budget result would not prove that the constrained model's global optimum is worse, because Adam convergence, equilibrium branch choice and the nonconvex parameter-to-shape map remain possible limitations. Conversely, matching the target would not establish uniqueness, physiological validity, spatial smoothness, inversion freedom or out-of-sample predictive value.
