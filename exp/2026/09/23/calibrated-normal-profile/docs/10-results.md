# Calibrated target-normal comparison at h = 0.20

[Slide-ready normal-only figure (PNG)](../data/32-final-slide/l2-normal-shape-activation-h200-16x9.png) · [vector PDF](../data/32-final-slide/l2-normal-shape-activation-h200-16x9.pdf) · [vector SVG](../data/32-final-slide/l2-normal-shape-activation-h200-16x9.svg)

## Design and calibration

The eight new fits cover four activation models with activation smoothness off or on. All use the same 100 × 10 mesh on a 1 × 0.1 strip, 400 independently controlled muscle triangles, fixed bottom and sides, fixed material properties, corrected physical `J = det(F)` energy, and parabolic top-displacement target `u_y = 4 h x(1−x)` with `h = 0.20`. They start independently at `q = u = m = v = 0`, `B = I`, and use 1,200 Adam updates with initial learning rate 0.03 and decay 0.99 per update. The four models are unrestricted symmetric `B`, positive-semidefinite contraction-only `B−I`, learned-axis rank-one contraction, and fixed reference x-direction contraction.

The positional loss is the mean squared *vector* displacement error over free top vertices. The normal loss compares corresponding oriented top-edge normals: `N = Σ wᵢ(1−nᵢ·nᵢ*)`, with fixed reference-edge-length weights summing to one. Activation smoothness is `R = mean ||Bᵢ−Bⱼ||²_F` over 498 neighboring muscle pairs. The fitted objective is

```text
J = L2 + c N + α h² R,       α ∈ {0, 1}
c = 0.02² / (1 − cos 5°) = 0.10511649525950077.
```

Thus position RMS 0.02 and a uniform 5° normal error each contribute 0.0004. Position uses model length units; it is not a millimeter value. This is a fixed scale chosen before these fits, not an optimized weight. The `beta` field in the saved normal-run histories stores the direct coefficient `c`.

## Results

The normal-only slide uses the eight new normal-loss endpoints in the original four-row, two-smoothness-column shape-and-activation layout. Every new normal run reached 1,200 accepted updates. The 16:9 PNG is 7680 × 4320 pixels; its PDF and SVG use vector geometry. Its mesh panels use one set of full-domain physical axes and an equal x/y scale. The dashed parabola is the target; activation color shows signed eigenvalues of `B−I`, with equal-length transported axes showing direction. [Figure manifest](../data/32-final-slide/manifest.json).

The table gives the new normal-fit endpoints. RMS is the square root of the original positional `L2`; normal RMS is the reference-weighted angular error. `D2` is a second reference-coordinate derivative of *displacement error*, not geometric curvature. `min J` is the minimum physical `det(F)`.

| Activation model | Smoothness α | Step | Position RMS | Normal RMS | D2 error | min J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Free symmetric | 0 | 1200 | 0.138478 | 19.730° | 103.302 | 0.3369 |
| Free symmetric | 1 | 1200 | 0.138225 | 20.545° | 89.208 | 0.3186 |
| Contraction, free directions | 0 | 1200 | 0.140449 | 21.537° | 61.751 | 0.4267 |
| Contraction, free directions | 1 | 1200 | 0.140317 | 22.601° | 36.859 | 0.5688 |
| Contraction, learned direction | 0 | 1200 | 0.140053 | 22.866° | 30.862 | 0.6068 |
| Contraction, learned direction | 1 | 1200 | 0.140089 | 23.148° | 26.076 | 0.7265 |
| Contraction, fixed x-direction | 0 | 1200 | 0.140152 | 23.071° | 28.219 | 0.7082 |
| Contraction, fixed x-direction | 1 | 1200 | 0.140167 | 23.199° | 25.514 | 0.7284 |

The original L2 controls are retained in the report data for matched quantitative comparisons. The unrestricted, unsmoothed L2 control failed at proposal 263, leaving accepted update 262; the other L2 controls reached 1,200. Therefore full endpoints in that first cell have unequal budgets. Matched comparisons use the latest saved step shared within each model: 260 for free activation and 1,200 for the other three. The normal-only slide does not depict L2 control shapes. [Independent comparison data](../data/20-verification/comparison.json) include the L2 controls, accepted steps, physical checks, and Hessian measurements.

The following changes compare each normal fit with the corresponding L2 control at its row's shared update. A positive position or D2 percentage means worse fit on that measure; a negative normal percentage means closer target normals. Projection is the fitted displacement projected onto the full prescribed displacement pattern, divided by the target's squared norm.

| Activation model | Smoothness α | Shared update | Position RMS | Normal RMS | D2 error | Target projection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Free symmetric | 0 | 260 | +4.18% | −44.56% | +37.2% | −27.8% |
| Free symmetric | 1 | 260 | +0.90% | −24.67% | +412.5% | −5.6% |
| Contraction, free directions | 0 | 1200 | +0.54% | −23.50% | +83.4% | +14.0% |
| Contraction, free directions | 1 | 1200 | +0.30% | −5.98% | +212.8% | +5.3% |
| Contraction, learned direction | 0 | 1200 | +0.15% | −7.88% | +69.6% | +11.9% |
| Contraction, learned direction | 1 | 1200 | +0.07% | −3.60% | +121.4% | +10.9% |
| Contraction, fixed x-direction | 0 | 1200 | +0.03% | −5.13% | +143.7% | +5.7% |
| Contraction, fixed x-direction | 1 | 1200 | +0.003% | −3.41% | +125.6% | +6.3% |

Normal matching reduces angular error in all eight matched comparisons, but raises the D2 displacement-error residual in all eight. Position RMS also rises in all eight, though by less than 1% in seven. The strongest angular gain, free activation without smoothness, comes with 4.18% higher position RMS and 27.8% less target-directed motion at update 260. Its L2 control has four backtracking top edges and later fails, so that cell is especially unsuitable for a simple quality ranking. The large D2 increases show that matching target normals does not by itself suppress short-scale profile oscillation.

## Provenance and verification

The preflight checks reproduced the chosen calibration, checked direct and implicit loss derivatives for all four activation models with smoothness off and on, and checked the L2 objective and gradients against the historical implementation. The maximum direct curve-normal derivative error was 3.42e−10. [Preflight receipt](../data/05-verification/checks.json).

The independent [endpoint and matched-step verification](../data/20-verification/comparison.json) passed. It recomputed position, normal, slope, D2, roughness, objective, activation constraints, physical `det(F)`, fixed displacements, and equilibrium forces. The largest force residual was 9.02e−11. All eight normal endpoints have positive physical `det(F)` (minimum 0.3186) and positive smallest displacement-Hessian eigenvalues (minimum 1.64e−5). Those Hessians support local discrete forward stability at the saved states; they do not prove inverse convergence. The eight saved L2 controls reproduce the historical checkpoints within maximum control difference 4.75e−11 and displacement difference 5.09e−13. The independent verification [terminal log](../logs/20-terminal.log) and [Comet record](https://www.comet.com/liblaf/apple/5a807b6e06fb4f2186fba93b4b93f214) record its completed run.

The first execution began as a 16-run factorial. After the scope was clarified to eight *new* normal fits, the remaining duplicate L2 work was stopped. Four L2 reproductions had finished; the four remaining L2 controls were copied from the earlier neutral-start study only after matching all numerical source hashes. An interrupted partial learned-direction L2 run is retained separately under `data/09-interrupted-duplicate-l2`. The revised [protocol receipt](../data/10-comparison/protocol.json) records each condition's origin, source hashes, input gates, initial state, and copied file hashes. No historical fit initialized a new normal optimization.

The eight new normal fits were saved in [case data](../data/10-comparison/). The first, interrupted run has [terminal log](../logs/10-terminal.log) and [Comet record](https://www.comet.com/liblaf/apple/6e1208325718438c8d4c681a40168ad7). The completion run has [terminal log](../logs/11-terminal.log) and [Comet record](https://www.comet.com/liblaf/apple/56fa76c584bf444d928eedbddf20d4c2). The final slide has [render log](../logs/32-terminal.log) and [Comet record](https://www.comet.com/liblaf/apple/93ceac03bedb47d2b59d4c8c3f835afe). Working directory: `exp/2026/09/23/calibrated-normal-profile`. The recorded Python entrypoints, with the one-thread numerical environment, were:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  DEBUG=1 CHERRIES_NAME="Calibrated normal 0.02 RMS versus 5 degrees preflight" \
  CHERRIES_TAGS="2d,normal,calibration,validation" \
  .venv/bin/python src/05-verify.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CHERRIES_NAME="0.02 RMS equals 5 degrees normal: neutral 4x2x2 comparison" \
  CHERRIES_TAGS="2d,normal,calibration,neutral,activation,smoothness" \
  .venv/bin/python src/10-run.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CHERRIES_NAME="Finish eight calibrated normal fits and reuse L2 controls" \
  CHERRIES_TAGS="2d,normal,calibration,reuse-controls" \
  .venv/bin/python src/11-finish-normal-only.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CHERRIES_NAME="Calibrated normal independent endpoint verification" \
  CHERRIES_TAGS="2d,normal,calibration,verification" \
  .venv/bin/python src/20-verify.py

env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CHERRIES_NAME="Final eight calibrated normal fits slide, 7680x4320" \
  CHERRIES_TAGS="2d,normal,calibration,slide,8k" \
  .venv/bin/python \
  src/31-render-normal-only.py --output 32-final-slide
```

These are finite Adam trajectories. The improved normal agreement does not establish inverse convergence or anatomical validity. In this experiment it comes with higher positional and D2 errors at the matched updates. Physical `det(F)` and forward-Hessian checks are separate from the data-loss comparison.
