# Physical-volume continuation: stopped on a new inversion

The exact corrected baseline continuation and the weak activation-variation branch both stopped at evaluated step 205 because a second tetrahedron inverted. Both retained step 204 as their last accepted diagnostic state. Neither reached the first 100-update review, so these runs cannot establish longer-run nasolabial-fold recovery or the effectiveness of activation smoothing.

## What was tested

The [frozen continuation protocol](10-protocol.md) preserves the corrected physical-volume material, phase constants, constraints, uniform face-coordinate MSE, Raw6 controls, and Adam settings. It resumes the canonical evaluated step-200 checkpoint, including its cached exact gradient and both optimizer moments. The first new evaluated state is step 201. The [CPU resume check](../data/10-resume-validation.json) agrees with the explicit Adam update formula to a maximum absolute control difference of 2.22e-16. The canonical checkpoint, fixture, local mechanics helpers, and loaded runtime sources were independently checked against their recorded hashes.

The independent [regularized branch](12-regularized-branch-protocol.md) starts from that same step-200 state and retains the same Adam moments. Its sole objective change is a weak mean squared Ainv difference across 501,409 same-muscle shared-face neighbors. Its fixed weight is 0.07049801689660026, calibrated to give the penalty 1% of the initial data-gradient RMS. The first update adds its analytic penalty gradient to the cached data gradient; subsequent states differentiate the combined objective. No projection or activation clamp is added.

The initial state already contains inverted cell 573586. It is accepted only for this diagnostic comparison; it is not physically valid. The stop rule permits no new inverted cell and no worsening of the known cell below its fixed determinant floor. A rejected geometry receives no adjoint evaluation and is never selected as best.

## Accepted-state measurements

All errors and motions below are vertex-vector RMS in millimetres on fixed supports. The regional boxes are the same overlapping target-space boxes used in the preceding closeup report.

| Measurement | Origin 200 | Baseline 204 | Weak penalty 204 |
| --- | ---: | ---: | ---: |
| Global fit error | 1.763031 | 1.751249 | 1.751259 |
| Nasolabial-region error | 2.536021 | 2.516629 | 2.516655 |
| Mouth-corner error | 1.658858 | 1.642046 | 1.642067 |
| Lateral-cheek error | 1.749542 | 1.737071 | 1.737089 |
| Lower-cheek / jaw error | 2.152271 | 2.140817 | 2.140826 |
| Face motion | 4.485370 | 4.498238 | 4.498210 |
| Target projection | 0.801626 | 0.804410 | 0.804405 |

Over four accepted updates, the baseline improves global error by 0.668% and nasolabial-region error by 0.765%. The two branches have nearly equal endpoint fit and motion: their global-fit difference is 0.00000919 mm. That makes the available endpoints comparable, but four updates are too few to assess a useful smoothing tradeoff.

The [accepted-state comparison](../data/40-comparison/summary.json) independently recomputes these measurements and checks the regularized optimizer checkpoint against its saved geometry. It also measures only 0.00007366 mm face RMS difference between branches. The mean squared neighboring Ainv jump is 0.05693948 without the penalty and 0.05693514 with it; both remain above the origin value 0.05616921. The complete scalar table is available as [CSV](../data/40-comparison/comparison.csv).

The 5 mm rest-normal displacement high-pass RMS changes slightly upward in all three marked regions for both branches. Its target-residual counterpart decreases slightly near the mouth corner and lateral cheek, and increases near the lower cheek. A lower displacement high-pass value alone is not a fold-quality criterion. The fixed images and exact cross-sections show no visible recovery of the target groove over this short interval.

![Canonical step 200, accepted baseline step 204, and target with identical camera and lighting](../data/30-step204/nasolabial-region.png)

The figure uses the actual saved coordinates, fixed topology, flat shading, and the same camera and lights as the previous report. Reference-to-step-204 displacement on the displayed face has RMS 0.01825 mm and maximum 0.05627 mm. The differences are correspondingly small.

Other fixed views: [mouth corner](../data/30-step204/region1-mouth-corner.png), [lateral cheek](../data/30-step204/region2-lateral-cheek.png), [lower cheek](../data/30-step204/region3-lower-cheek.png), and [face context](../data/30-step204/side-context.png). The [horizontal sections](../data/30-step204/nasolabial-horizontal-sections.png) use exact native triangle intersections at y = 2.170, 2.180, and 2.190 m, with the same clipping window. They are not a scalar fold-depth measurement.

## Why the continuation stopped

Newly inverted global cell 620845 is an inactive, pure-fat tetrahedron near the pre-existing inversion. It is not incident to the fitted face surface; its accepted step-204 centroid is about 12.14 mm from the closest deformed skin triangle. This location does not establish a cause for the visible bumps. The independent [inversion audit](../data/11-inversion-audit/summary.json) checks all tetrahedra by both signed-volume ratios and deformation-gradient determinants, which agree to floating-point precision.

Three of this cell's vertices are fixed, while vertex 100320 is free. The free vertex crosses the plane of the fixed triangle between steps 204 and 205. Its height relative to that plane falls from 0.424481 mm at rest to about 0.0003105 mm on the original side at accepted step 204, then crosses about 0.0004325 mm beyond the plane at the rejected state. The rest tetrahedron has mean-ratio quality 0.56921, where 1 denotes a regular tetrahedron: this is severe deformation of a nondegenerate rest element. These geometric facts locate the failure; they do not by themselves establish which boundary-condition or material change would resolve it.

For the unregularized path its determinant changes from +0.00784765 at step 200 to +0.000731454 at accepted step 204, then -0.00101894 at rejected step 205. The regularized path reaches +0.000759173 at step 204 and -0.00101198 at step 205. It was already nearly collapsed at the origin. The pre-existing inverted cell improves slightly during these updates. The stop is therefore specifically caused by the newly inverted element, not worsening of the known cell.

Step-205 forward solves succeed with residual norms about 1e-10, but both evaluated geometries fail the independent orientation check. Solver convergence and geometric validity are separate conditions. The actual [passive stable material](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean.py) uses a finite polynomial term in J, and the [corrected active material](../src/volume_preserving_active.py) also has no inversion barrier. Penalizing physical volume change does not impose J > 0.

The frozen baseline rejected NPZ mistakenly retains its generic `solver_valid=true` flag from the initial driver. Its rejection receipt explicitly records that no gradient was evaluated. The separate inversion audit supplies an explicit [correction receipt](../data/11-inversion-audit/correction-receipt.json); this rejected file must not be interpreted as accepted or physically valid. The regularized driver writes `solver_valid=false` for rejected geometry and retains an exact optimizer checkpoint after every accepted state. The baseline stopped before its first ten-step checkpoint, so its accepted step-204 geometry is preserved, but no step-204 Adam checkpoint was saved.

## Consequences for the staged plan

The requested continuation was attempted with the prescribed settings and stopped under the predeclared guard. There is no step-300 or step-400 result. The nasolabial-weighted test is not triggered: the required 100-update regional comparison was never reached, and both global and local error were still decreasing during the four accepted updates. No conclusion about mechanical impossibility follows.

The weak smoothness comparison was also attempted from the common checkpoint and hit the same new inversion. It does not show that regularization cannot improve the surface; it shows that this particular weak penalty does not prevent the immediate failure. Skin and attachment changes were not introduced. The next controlled question is how to preserve element orientation through equilibrium and inverse updates before investing in longer anatomical fitting or interpreting fold recovery.

## Commands and recorded runs

Working directory for both runs:

```text
exp/2026/09/08/physical-volume-continuation
```

```bash
CHERRIES_NAME='Physical volume continuation 200 to 300' CHERRIES_TAGS='physical-volume,continuation,exact-adam,nasolabial,diagnostic' .venv/bin/python src/20-continue.py > logs/20-fit300-terminal.log 2>&1
CHERRIES_NAME='Physical volume weak activation variation from step 200' CHERRIES_TAGS='physical-volume,regularization,same-muscle,paired,diagnostic' .venv/bin/python src/22-regularized-branch.py > logs/22-reg300-terminal.log 2>&1
```

Both processes and Cherries shutdown hooks finished, with exit code 1 from the intentional geometry rejection. Their Comet summaries record four accepted metric steps and the exception. The active interpreter was used directly to preserve the baseline environment: Python 3.14.6, Torch 2.12.0+cu130, CUDA build 13.0, on the RTX 4090. Git HEAD was d56fa1b553b287b22b2cf7bb82d46117e34ed6bb. No commit or push was performed.

| Run | Observed Comet summary | Local evidence |
| --- | --- | --- |
| Exact continuation | [Comet run](https://www.comet.com/liblaf/apple/c43b4242f6694279ad767ce786471d46): fit RMS [4] = (1.7512493879795168, 1.7600475028977682) | [summary](../data/20-fit300/summary.json), [trace](../data/20-fit300/trace.csv), [terminal](../logs/20-fit300-terminal.log), [provenance](../data/20-fit300/provenance.json) |
| Weak activation variation | [Comet run](https://www.comet.com/liblaf/apple/c7abc806b82744bb9de6384a0b67823c): fit RMS [4] = (1.751258579664865, 1.7600485899259448) | [summary](../data/22-reg300/summary.json), [trace](../data/22-reg300/trace.csv), [terminal](../logs/22-reg300-terminal.log), [provenance](../data/22-reg300/provenance.json) |

The full Comet summary blocks are preserved in the linked terminal logs. Environment-detail and Git-patch auto-collection were disabled in the experiment-local profile; numerical runtime sources are archived locally. Comet scalar/source logging completed, but its Cherries asset hook does not upload these numerical artifacts. The render was a separate local DEBUG postprocess; its [terminal log](../logs/30-step204-render.log) records a post-output snapshot-copy warning, while its direct [output receipt](../data/30-step204/summary.json), image dimensions, native geometry, and file hashes passed verification.

The independent inversion audit is CPU-only and records its separate analysis runtime in its receipt (Python 3.14.7, NumPy 2.5.2, PyVista 0.46.5, Torch 2.13.0). It loads the saved arrays and does not rerun or alter the forward physics. Its determinant values agree with the original run's diagnostics.

## Comet summary excerpts

These are the numerical summary blocks emitted by each completed shutdown hook. The preceding table links their complete terminal records.

20-fit300-terminal.log

```text
Comet.ml Experiment Summary
---------------------------------------------------------------------------------------
  Data:
    display_summary_level : 1
    name                  : Physical volume continuation 200 to 300
    url                   : https://www.comet.com/liblaf/apple/c43b4242f6694279ad767ce786471d46
  Metrics [count] (min, max):
    fit_rms_mm [4]    : (1.7512493879795168, 1.7600475028977682)
    gradient_rms [4]  : (7.94557233451903e-06, 8.04888082228973e-06)
    known_cell_J [4]  : (-0.30295910593365444, -0.30067756218732067)
    objective_mm2 [4] : (1.0222914729662107, 1.0325890708188896)
```

22-reg300-terminal.log

```text
Comet.ml Experiment Summary
---------------------------------------------------------------------------------------
  Data:
    display_summary_level : 1
    name                  : Physical volume weak activation variation from step 200
    url                   : https://www.comet.com/liblaf/apple/c7abc806b82744bb9de6384a0b67823c
  Metrics [count] (min, max):
    fit_rms_mm [4]    : (1.751258579664865, 1.7600485899259448)
    gradient_rms [4]  : (7.914555259242178e-06, 8.018031861251136e-06)
    known_cell_J [4]  : (-0.3029594763876514, -0.30071523035683706)
    objective_mm2 [4] : (1.0263160188335783, 1.0365637596704689)
```
