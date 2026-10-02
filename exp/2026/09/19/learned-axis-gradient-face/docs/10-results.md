# Learned-axis contraction-only fitting of the 3D face

Both branches completed **200 updates from neutral** using learned-axis
contraction-only activation and the same gradient-only data loss. Smoothness
reduced squared activation-tensor roughness by **98.2%**, but increased surface-
gradient RMS by **13.3%**, increased positional RMS by **7.8%**, and retained
**31.3% less motion**. Both stayed inversion-free throughout the run.

Neither branch met the declared inverse-convergence criteria. These are matched
finite-budget results, not converged optima or an intrinsic capacity comparison.

## Matched endpoint results

Errors and motion use fixed reference-surface area weights. Surface-gradient
RMS is dimensionless; position and motion are vector RMS in millimeters.

| Metric | Neutral | Smoothness off | Smoothness on |
| --- | ---: | ---: | ---: |
| Surface-gradient residual RMS | 0.272271 | **0.194387** | 0.220259 |
| Positional RMS, mm | 5.095908 | **4.171781** | 4.496773 |
| Centered positional RMS, mm | 4.693870 | **3.663917** | 4.026916 |
| Mean residual norm, mm | 1.983900 | 1.994860 | 2.001229 |
| Motion RMS, mm | 0 | 1.439004 | 0.989105 |
| Projection onto full target displacement | 0 | 0.204774 | 0.129497 |
| Activation tensor roughness R | 0 | 3.214770 | **0.058137** |
| Minimum physical det(F) | 1 | 0.563383 | 0.743923 |
| Inverted tetrahedra | 0 | **0** | **0** |
| Maximum strength s | 0 | 3.819677 | 1.524078 |
| Minimum active axial stretch parameter 1/(1+s) | 1 | 0.207483 | 0.396184 |
| Axis rotation from initialization, volume-weighted RMS degrees | 0 | 3.080392 | 2.976516 |

The active stretch parameter describes the imposed actuation tensor; it is not
the measured physical stretch. The learned axes moved about three degrees RMS
from the target-gradient-derived initial field. This single initialization and
optimizer budget do not establish a direction-independent optimum.

![Matched final surfaces](../data/30-figures/full-face-overview.png)

All states use the same camera, material, lighting, flat shading, and deformation
scale one. Check the [optimization curves](../data/30-figures/optimization-progress.png)
and the [saved corresponding skins](../data/30-figures/skins/).
The unsmoothed endpoint shows stronger local creasing at the mouth corner and
larger overall movement. The smoothed endpoint stays visibly closer to neutral.
Both still have substantially less mouth opening than the target.

## Does smoother activation give a closer surface shape?

Not in this comparison. The frozen 5 mm cotangent high-pass diagnostic projects
onto reference normals. The target-relative high-pass residual increases in all
three measured regions when smoothness is enabled:

| Normal high-pass residual RMS, mm | Neutral | Off | On |
| --- | ---: | ---: | ---: |
| Right lateral cheek | 0.080613 | **0.068702** | 0.072827 |
| Right lower cheek / jaw | 0.118481 | **0.101881** | 0.107917 |
| Right mouth corner | 0.996162 | **0.759334** | 0.878442 |
| Area-weighted three-region union | 0.438733 | **0.336288** | 0.387339 |

The union's high-pass displacement RMS drops from 0.142350 to 0.099667 mm
(-30.0%), while its target-relative high-pass residual rises by 15.2%. Thus the
smoothed branch generates less fine-scale motion and a weaker expression; it
does not match the target's fine-scale shape more closely. The global derivative
error and positional error also worsen. There remains substantial underfitting
in both branches, so this is not an optimal regularization-weight claim.

Regional views: [mouth](../data/30-figures/details/region1-mouth-corner-overview.png),
[cheek](../data/30-figures/details/region2-lateral-cheek-overview.png), and
[jaw](../data/30-figures/details/region3-lower-cheek-overview.png).

## Convergence and independent verification

Both statuses are `completed_budget_not_convergence_certified`. Over the final
25 updates, each branch's own objective decreased by **3.23% off** and **1.43%
on**, compared with the 0.1% plateau threshold. Final projected-gradient RMS
ratios were **37.29% off** and **58.07% on**, compared with the 1% threshold.
The forward equilibrium solver's success does not establish inverse convergence
or a stable mechanical minimum.

The independent CPU [endpoint audit](../data/25-verification/checks.json) passed:

- All **402 forward and 402 adjoint receipts** succeeded (201 evaluations per
  branch, including neutral). All trace entries report zero inverted cells.
- Shared initial activation and displacement differences were exactly zero;
  the recorded learning-rate histories matched exactly.
- Reconstructing packed activation from saved strengths and axes agreed within
  `8.89e-16`. Nonnegative strengths, unit axes, rank-one PSD increments, and
  contraction-only tensors were verified. Both minimum B eigenvalues are one
  to floating-point precision.
- Independently recomputed graph regularization and objective accounting agreed
  within `4.45e-16`; direct signed tetrahedron-volume ratios reproduced physical
  determinant extrema within `3.45e-15`.
- Fixture metadata and archived numerical sources matched the accepted gates.

The successful CPU activation and full implicit derivative gates are described
below and in the [validation report](05-validation.md).

## What is being compared

Both branches optimize the same gradient-only surface loss from the same neutral
face. The activation is `B=I+s n n^T`, where `s>=0` and `||n||=1`. Both strength
and axis receive gradients; after each Adam step, strength is clamped and the
axis normalized. This is **three physical degrees of freedom per active cell**,
one strength plus two axis coordinates, stored as four constrained scalars.
It replaces the previous unrestricted six-component symmetric activation.

There are 288,235 active cells: 864,705 generic physical activation degrees of
freedom. The active natural stretch is `1/(1+s)` along the axis and one in the
transverse plane. There is no active extension, strength upper bound, or
magnitude penalty. Positive activation tensors do not by themselves ensure
positive physical element volumes.

Initialization is exactly neutral (`s=0`, `B=I`, `u=0`). Both branches share axes
chosen as minimum-eigenvalue eigenvectors of the data objective's symmetric
tensor gradient at neutral. They are effective target-derived directions, not
anatomical fibers. No previous fitted face is used. The explicit strength/axis
parameterization allows nonzero strength derivatives at neutral; axis
derivatives become available as strength grows.

The data objective is unchanged from the preceding gradient-only face study:
area-normalized squared rest-surface tangential derivative error, multiplied by
`99.56749008299767`. **Positional L2 is evaluation-only.** The on branch adds
`lambda * R`, where

```text
R = ell^2 / V_active * sum_same-muscle_edges w_ij ||B_i - B_j||_F^2
ell = 0.005 m
lambda = 5.6274264974806405
```

The conductance uses shared-face area, center distance, and muscle fraction.
Smoothing the tensor handles the `n` versus `-n` equivalence without a false
axis-sign discontinuity. This regularizer encourages agreement across
face-sharing cells within each muscle; it does not couple different labels.

The fixture has 228,660 volume vertices and 1,146,517 tetrahedra. The target
uses 15,299 skin vertices and 29,899 triangles. The same corrected physical
volume energy `J=det(F)`, materials, skull fixation, and solver are retained.
There is no added contact, Koiter skin, positional anchor, or interior target.
See the [pre-run protocol](00-protocol.md) and saved
[runtime protocol](../data/20-comparison/protocol.json).

## Smoothness weight selection

A fresh eight-update data-only probe equalized the Euclidean norms of strength
and axis gradients to establish a coefficient scale. This is a coordinate-
dependent optimization heuristic, not a measured material or anatomical value.
Each of the following 20-update pilots then restarted at neutral.

| Coefficient | Tensor roughness R | Roughness reduction | Gradient RMS | Positional RMS, mm | Data-loss increase |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.110560 | — | 0.250083 | 4.878981 | — |
| 0.562743 | 0.083456 | 24.5% | 0.250633 | 4.884528 | 0.44% |
| **5.627426** | **0.026235** | **76.3%** | **0.253711** | **4.914831** | **2.92%** |
| 56.274265 | 0.002886 | 97.4% | 0.261438 | 4.989150 | 9.29% |

All pilot endpoints had zero inverted tetrahedra. The selected coefficient was
the smallest tested value achieving the predeclared 75% reduction. Its
gradient RMS increased 1.45%; the squared data loss increased 2.92%. The pilot
is a weight-selection diagnostic, not a converged fit. The final comparison
restarts both branches at neutral with fresh Adam states.

The [selection receipt](../data/15-selection/selection.json) includes every
candidate, exact coefficients, motion, determinant minima, and source hashes.
The [pilot Comet run](https://www.comet.com/liblaf/apple/d10419c83f464f1f8f23e8b9525b4a6d)
records 21 evaluations per branch (updates 0–20). Its initial Comet parameter
`steps=500` is the driver default before pilot-mode override; the saved protocol,
traces, endpoint status, and `pilot_steps=20` record the effective budget.

## Validation and interpretation limits

The CPU activation algebra, feasibility, sign-invariance, and smoothness
derivative gates passed. The full implicit face derivative gate passed in
strength and tangent-axis directions, with worst relative error **0.03053%**.
See [the validation report](05-validation.md), including the preserved failed
looser-tolerance check and the tighter equilibrium settings used for finite
differences.

The final optimizer uses Adam `lr=0.3`, `eps=0.01`, `betas=(0.9,0.999)`, with
the same step-based halving every 100 updates in both branches. Forward PNCG
uses maximum 5,000 iterations, relative tolerance `5e-4`, absolute tolerance
`1e-10`, and the inherited 10-trial line search. Adjoint tolerance is `5e-4`.
All branch-specific forward and adjoint receipts are saved.

The stopping diagnostic requires two consecutive 25-update checks, starting
at update 100, with objective relative span below 0.001 and physical rank-one
projected-gradient RMS below 1% of its initial value. This is a numerical
diagnostic on a nonconvex feasible set, not a proof of global optimality or
mechanical stability. A finite budget is reported as such.
The trace's `recent_objective_relative_span` is measured only at the scheduled
checks (updates 100, 125, 150, 175, and 200); other rows contain the driver's
placeholder value 1.0 and must not be interpreted as measured spans.

## Reproduction and evidence

Working directory:
`exp/2026/09/19/learned-axis-gradient-face`.
The runtime uses the project interpreter (Python 3.14.6, Torch 2.12.0+cu130,
Warp 1.14.0), float64, four CPU threads, and an RTX 4090. Git base is
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the pre-existing working tree was
dirty. The 96 source records (95 unique snapshots) and fixture hashes define the
actual run. Changes for this study are scoped to this experiment group.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

CHERRIES_NAME='Learned-axis face: smoothness weight selection' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,gradient-only,smoothness,pilot' \
.venv/bin/python src/10-run.py \
  --phase pilot --output 10-pilot --preparation data/06-preparation-tight

DEBUG=1 CHERRIES_NAME='Learned-axis face: choose smoothness weight' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,smoothness,selection' \
.venv/bin/python src/15-select-pilot.py

CHERRIES_NAME='Neutral learned-axis face: smoothness off versus on' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,gradient-only,smoothness,matched-comparison' \
.venv/bin/python src/10-run.py \
  --phase compare --output 20-comparison \
  --preparation data/06-preparation-tight --steps 200 \
  --coefficient 5.6274264974806405

DEBUG=1 CHERRIES_NAME='Learned-axis face: independent endpoint verification' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,verification' \
.venv/bin/python src/25-verify-results.py \
  --preparation-dir data/06-preparation-tight

CHERRIES_NAME='Learned-axis face: final smoothness comparison figures' \
CHERRIES_TAGS='learned-axis,contraction-only,3d,smoothness,figures' \
.venv/bin/python src/30-render.py
```

The [comparison Comet run](https://www.comet.com/liblaf/apple/e3fde5ffc88f4d38a82bd8790fd6308e)
records scalar metrics, source, and metadata. The installed Comet asset hook is
a no-op, so local NPZ, VTP, and figure files—not remote asset uploads—are the
artifact evidence. Cherries preserves local logs and output snapshots under
the repository's `.cherries/runs/2026/09/19/learned-axis-gradient-face/`.

The main comparison, independent audit, and final renderer all exited with
code zero, including Cherries shutdown. The comparison ran from 16:11:53 to
17:14:27 on 2026-09-19 (Asia/Shanghai). The
[renderer Comet run](https://www.comet.com/liblaf/apple/474f843a0ebc481dbd926aeae58167ac)
finished at 17:15:28. The full-face image, all three regional views, and the
optimization plot were visually inspected; labels and geometry are readable
without clipping. Ruff passed on all seven scripts in this group.

The saved outputs include per-branch traces, summaries, solver receipts, full
activation/displacement NPZ checkpoints, the latest optimizer state, independent
verification JSON, four corresponding skin VTP files, and five comparison plots.
The [render receipt](../data/30-figures/summary.json) records the shared cameras,
endpoint labels, source snapshot, and runtime. Preparation commands and gate
artifacts are documented in [05-validation.md](05-validation.md).

Extract from `logs/10-run.log` (logging prefixes and unrelated fields omitted):

```text
Comet.ml Experiment Summary
  Data:
    name : Neutral learned-axis face: smoothness off versus on
    url  : https://www.comet.com/liblaf/apple/e3fde5ffc88f4d38a82bd8790fd6308e
  Metrics [count] (min, max):
    off/activation_smoothness [201]    : (0.0, 3.214769691629389)
    off/fit_rms_mm [201]               : (4.171780532556763, 5.095908027590781)
    off/inverted_all_cells             : 0.0
    off/surface_gradient_rms [201]     : (0.19438658785695315, 0.2722714131419773)
    on/activation_smoothness [201]     : (0.0, 0.05813672423761841)
    on/fit_rms_mm [201]                : (4.496773317549205, 5.095908027590781)
    on/inverted_all_cells              : 0.0
    on/surface_gradient_rms [201]      : (0.22025887386925286, 0.2722714131419773)
  Others:
    cherries/entrypoint : exp/2026/09/19/learned-axis-gradient-face/src/10-run.py
    cherries/exp_dir    : exp/2026/09/19/learned-axis-gradient-face
    cherries/git/sha    : d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
    cherries/start_time : 2026-09-19 16:11:53.626340+08:00
    cherries/end_time   : 2026-09-19 17:14:27.229140+08:00
```
