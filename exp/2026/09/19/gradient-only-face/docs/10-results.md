# Gradient-only fitting of a 3D human face from neutral

Both branches completed 100 updates from the **same neutral face**, zero
activation, and fresh Adam state. Gradient-only fitting reduced global
surface-gradient RMS by **31.1% relative to L2**, but increased positional RMS
by **51.0%** and produced **36.7% less motion**. Local improvement was mixed:
cheek and jaw high-frequency residuals decreased, while the mouth-corner
residual increased. Both final volume meshes contain one inverted tetrahedron.
This pilot therefore supports a shape-versus-position tradeoff, not a complete
replacement for positional fitting or an inversion-free solution.

## Matched endpoint results

Errors and motion use the same fixed reference-surface area weights. Gradient
RMS is dimensionless; position and motion are vector RMS in millimeters.

| Metric | Neutral | L2, update 100 | Gradient only, update 100 |
| --- | ---: | ---: | ---: |
| Surface-gradient residual RMS | 0.272271 | 0.232366 | **0.160205** |
| Positional RMS, mm | 5.095908 | **2.341699** | 3.535964 |
| Motion RMS, mm | 0 | 3.480295 | 2.204222 |
| Mean residual norm, mm | 1.983900 | 0.773135 | 1.943826 |
| Centered positional RMS, mm | 4.693870 | 2.210389 | 2.953740 |
| Projection onto full target displacement | 0 | 0.627635 | 0.352812 |
| Minimum physical `det(F)` | 1 | -0.118597 | -0.163034 |
| Inverted tetrahedra | 0 | 1 | 1 |
| Active tensors with nonpositive minimum eigenvalue | 0 | 819 | 203 |
| Minimum activation eigenvalue | 1 | -0.305231 | -0.169922 |

![Matched final surfaces](../data/30-figures/full-face-overview.png)

The renders use identical cameras, gray material, lighting, flat triangle
shading, and deformation scale 1. Final labels include the step and inversion
count. See the [common-scale position errors](../data/30-figures/position-error-overview.png)
and [optimization trajectories](../data/30-figures/optimization-progress.png).

## Are the bumps actually better?

The frozen diagnostic projects displacement onto reference normals and applies
the existing 5 mm cotangent high-pass filter in three target-defined regions.
The residual compares the filtered predicted motion with the filtered target;
the displacement measure alone describes how much fine-scale motion was made.

| Normal high-pass residual RMS, mm | L2 | Gradient only | Change |
| --- | ---: | ---: | ---: |
| Right lateral cheek | 0.134950 | **0.064067** | -52.5% |
| Right lower cheek / jaw | 0.132313 | **0.089215** | -32.6% |
| Right mouth corner | **0.411005** | 0.560223 | +36.3% |
| Area-weighted three-region union | **0.215986** | 0.250875 | +16.2% |

The union's high-pass **displacement** RMS falls from 0.325518 to 0.219266 mm
(-32.6%), but its target-relative high-pass **residual** increases. Thus less
fine-scale deformation does not mean uniformly closer target shape. The
global gradient objective and the regional, normal-only 5 mm diagnostic
measure different aspects of the result.

The [cheek close-up](../data/30-figures/details/region2-lateral-cheek-overview.png)
shows visibly reduced irregularity for gradient-only fitting. The
[mouth close-up](../data/30-figures/details/region1-mouth-corner-overview.png)
still shows local artifacts and incomplete target opening. All mesh facets
remain visible because the shading is flat; no display smoothing was applied.

Gradient-only fitting leaves a much larger mean positional residual. This is
consistent with its constant-translation nullspace, although the skull
constraints prevent treating translation as a freely realizable volume mode.
Removing the mean residual still leaves a higher centered error, so an offset
alone does not explain the positional gap. The weaker recovered expression
and finite optimization budget also matter.

## Comparison before any inversion

At the shared saved update **70**, both volume meshes have zero inverted
tetrahedra, and the same tradeoff is already present:

| Metric | L2, update 70 | Gradient only, update 70 |
| --- | ---: | ---: |
| Gradient RMS | 0.229690 | **0.171981** |
| Positional RMS, mm | **2.697024** | 3.740868 |
| Motion RMS, mm | 3.008853 | 1.931451 |
| Minimum physical `det(F)` | 0.076050 | 0.080891 |
| Inverted tetrahedra | 0 | 0 |

L2 first inverts at update 81; gradient-only first inverts at update 79.
The separately saved best noninverted states are L2 update 80
(`min J=0.000330`) and gradient-only update 78 (`min J=0.007789`). These are
severely compressed states, not evidence of well-conditioned or mechanically
stable solutions. Their different update counts are not substituted into the
primary matched comparison.

## Objective and physical setup

The [pre-run protocol](00-protocol.md) fixes the comparison. There are 228,660
volume vertices, 1,146,517 tetrahedra, and 288,235 active cells. Both objectives
use the same 15,299 observed skin vertices and 29,899 triangles. The three
isolated historical L2 vertices absent from the skin are omitted from both
branches. The target is the supplied Smile correspondence field; no interior
target deformation is invented.

For `e=u-u_target`, the optimized objectives are

```text
L2    = (10^6 / 3) sum_i w_i ||e_i||^2
Lgrad = c / sum_t A_t * sum_t A_t ||sum_i e_i tensor grad(phi_i)||_F^2
```

All weights, areas and triangle gradients are fixed in the reference shape.
`Lgrad` contains **no positional or activation-regularization term**. It matches
the full vector-valued tangential displacement derivative in world coordinates;
it is not normal-only or rotation-invariant shape matching.

The fixed positive multiplier `c=99.567490083` matches the initial
active-muscle-volume-weighted Frobenius RMS of Adam's proposed change in `B`
to L2: **0.007781754846** in each branch. The unscaled gradient branch would
propose only 0.000081908376 because Adam epsilon is 0.01. Off-diagonal entries
count twice in the Frobenius norm. This matches initial step magnitude only,
not direction or later trajectory, and leaves the gradient-only minimizers
unchanged. Loss values from the two objectives are not compared directly.

Both branches optimize unrestricted symmetric `B=I+sym(q)` with Adam
`lr=0.3`, `eps=0.01`, `betas=(0.9,0.999)`, no decay and no projection. Physics
uses the experiment-local corrected active norm `||F B||^2` and physical
volume `J=det(F)`, with the historical passive materials and classical Lamé
convention. Fat is 0.003 MPa / nu 0.49, muscle 0.03 MPa / nu 0.49, and passive
aponeurosis 0.1 MPa / nu 0.35. Skull fixation is unchanged; there is no Koiter
skin energy or contact. This experiment does not modify the production law.

## Verification and limits

- Surface zero/translation/affine/refinement and autograd checks passed.
- The complete implicit gradient at neutral passed central finite differences in two
  directions and at epsilon 0.01 and 0.005; maximum relative error was 0.6897%
  against the predeclared 2% threshold.
- All 202 recorded forward/adjoint evaluations succeeded. Forward settings
  were PNCG max 5,000, rtol 5e-4, atol 1e-10, with the existing 10-trial line
  search. No final line-search receipt was exhausted. Adjoint rtol was 5e-4.
- Independent CPU verification confirmed exact shared zero initialization,
  identical calibration/fixture receipts and 94 archived numerical source
  hashes, endpoint metrics, physical determinants, activation eigenvalues and
  fixed constraints. Cotangent sparse energy agreed with direct surface
  gradient energy to at most 6.25e-17 absolute error.
- A separate direct tetrahedron-volume-ratio recomputation of the shared
  update-70 checkpoints confirmed zero inversions and their tabulated
  positional errors and minimum determinants.
- Both own objectives were still improving: the last ten updates reduced L2
  by 7.95% and the gradient objective by 4.08%. Their final activation-gradient
  RMS values were 25.4% and 20.8% of their initial values, respectively.
  Neither inverse convergence nor a stable equilibrium minimum is certified.
- Raw6 permits indefinite activation tensors and active extension. Neither
  gradient matching nor the constitutive volume penalty is an inversion
  barrier. These are numerical comparison results, not anatomical validation.

## Reproduction and recorded outputs

Working directory:
`exp/2026/09/19/gradient-only-face`.
Interpreter: project `.venv/bin/python`, Python 3.14, Torch 2.12.0+cu130,
Warp 1.14.0, RTX 4090, float64, four CPU threads. Git base:
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the existing working tree was dirty.
Archived loaded sources, rather than that commit alone, define the runtime.
The new work is scoped to this experiment group.

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DEBUG=1 CHERRIES_NAME='3D face surface-gradient derivative and step-scale validation' \
CHERRIES_TAGS='gradient-only,3d,physical-J,validation' \
.venv/bin/python src/10-run.py \
  --output 06-validation --validation-only true

CHERRIES_NAME='Neutral face: L2 versus gradient-only surface fitting' \
CHERRIES_TAGS='gradient-only,l2,3d,neutral-start,physical-J,matched-comparison' \
.venv/bin/python src/10-run.py \
  --output 10-comparison \
  --calibration exp/2026/09/19/gradient-only-face/data/06-validation/calibration.json

DEBUG=1 CHERRIES_NAME='Independent checks of neutral-start 3D face endpoints' \
CHERRIES_TAGS='gradient-only,3d,verification' \
.venv/bin/python src/20-verify.py

CHERRIES_NAME='3D face gradient-only versus surface L2 figures' \
CHERRIES_TAGS='gradient-only,3d,figures' \
.venv/bin/python src/30-render.py
```

The runner refuses to overwrite an existing comparison directory. L2 took
1,184.1 seconds and gradient-only took 976.7 seconds. The normal Cherries/Comet
run and shutdown completed successfully, with 101 metric samples per branch:
[Comet experiment](https://www.comet.com/liblaf/apple/04d65402fa1e4703b22811aacae11522).
Its summary records metrics and source upload; the checkpoint files and source
archive are verified locally. The complete `Comet.ml Experiment Summary` block
is preserved in [the run log](../logs/10-run.log).

The installed Cherries Comet asset hook is a documented no-op, so remote
metric/source recording does not imply that meshes or figures were uploaded.
The files linked here are the local artifacts. Logging plugin initialization
was reordered after fitting to preserve the local snapshot log handler; this
does not change the archived numerical implementation used for the results.
The [final rendering run](https://www.comet.com/liblaf/apple/d8953393e59f4b13a0349cd52892ab04)
then completed with no Local-plugin error. Its 20 PNGs are byte-identical to
the preserved initial rendering. Source, camera and runtime receipts are in
[the rendering summary](../data/30-figures/summary.json); the first renderer's
nonfatal snapshot-copy error remains recorded in `logs/30-render-initial.log`.

Key local evidence:

- [Protocol and archived-source hashes](../data/10-comparison/protocol.json),
  [scale calibration](../data/10-comparison/calibration.json), and
  [full-chain derivative checks](../data/10-comparison/gradient-validation.json).
- [Independent endpoint verification](../data/20-verification/checks.json).
- `data/10-comparison/{l2,gradient}/`: `trace.csv`, `summary.json`,
  `solver-receipts.jsonl`, `last.npz`, `best.npz`, `best-noninverted.npz`,
  `step-0070.npz`, all other ten-update checkpoints, and `optimizer-latest.pt`.
- `data/30-figures/`: full-face comparison, common error maps, mouth/cheek
  detail comparisons, optimization plots, rendering receipts and `.vtp`
  surfaces with `GlobalPointId`, reference, target, displacement and error.
  The target export is surface-only.
