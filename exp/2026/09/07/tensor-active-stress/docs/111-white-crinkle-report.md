# White skin and fixed-cohort muscle section

The current result has visibly smoother skin and much less severe local muscle distortion around the mouth. Surface fit RMS is higher: **1.610 mm**, compared with **0.654 mm** for the bumpy baseline. Some skin waviness and target-shape mismatch remain.

The panels compare **June saved no-skin baseline · global step 194** and **Current PSD active stress · global step 1024** using actual saved geometry. They do not have equal optimizer budgets or the same activation formulation. This is therefore a visual record, not a matched causal comparison.

Within each comparison, panels share their recorded camera and true geometric scale. White skin gives exterior context. The target is observed `IsFace` skin only; it has no invented interior tissue state.

## Saved-state diagnostics

| State | Fit RMS (mm) | Motion RMS (mm) | detF min / max | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| June saved no-skin baseline · step 194 | 0.654339 | 4.974140 | -7.394505 / 16.514292 | 142 |
| Current PSD active stress · step 1024 | 1.610246 | 4.125968 | 0.135720 / 2.691734 | 0 |

## White exterior skin

![White skin front](../data/110-white-crinkle/skin-front.png)

![White skin mouth closeup](../data/110-white-crinkle/skin-mouth.png)

[Front PNG](../data/110-white-crinkle/skin-front.png) · [Front PDF](../data/110-white-crinkle/skin-front.pdf) · [Mouth PNG](../data/110-white-crinkle/skin-mouth.png) · [Mouth PDF](../data/110-white-crinkle/skin-mouth.pdf) · [Mouth-with-edges PNG](../data/110-white-crinkle/skin-mouth-edges.png) · [PDF](../data/110-white-crinkle/skin-mouth-edges.pdf)

## Fixed-rest-cohort interior muscle

The retained half-space is **x <= 1.4070235649558214 m**. The cohort contains **145,929** muscle-bearing tetrahedra (`MuscleFraction > 0`). Its rule is: `min signed distance across the four rest vertices <= 0; retain entire cell`. The same saved source-cell IDs are extracted in each state; deformed states are never reclipped. Muscle selection is `MuscleId > 0 and MuscleFraction > 0; equals saved ActivationMask`. The retained cohort has **55** inverted muscle-bearing tetrahedra in the June state and **0** in the current state; no inversion rejection or repair was applied.

![Fixed-cohort muscle crinkle](../data/110-white-crinkle/muscle-crinkle.png)

![Interior muscle mouth closeup](../data/110-white-crinkle/muscle-mouth.png)

[Crinkle PNG](../data/110-white-crinkle/muscle-crinkle.png) · [Crinkle PDF](../data/110-white-crinkle/muscle-crinkle.pdf) · [Mouth PNG](../data/110-white-crinkle/muscle-mouth.png) · [Mouth PDF](../data/110-white-crinkle/muscle-mouth.pdf)

![Rest-space plane context](../data/110-white-crinkle/plane-context.png)

[Plane context PNG](../data/110-white-crinkle/plane-context.png) · [Plane context PDF](../data/110-white-crinkle/plane-context.pdf)

## Visual observations

The current white skin is visibly smoother at the cheek, under-eye, mouth border, and lower jaw, while residual waviness remains around the lip corners and lower cheek. The current mouth contour and cheek folds still differ from the target; its fit RMS is higher and its smaller motion RMS does not establish improved matched motion or a regularization effect. The sagittal crinkle and oblique mouth views show severe localized baseline perioral and cheek stretching/folding, much less severe local distortion in the current state, and no inversion repair or rejection. The edged rest panel separates irregularities inherent to the fixed whole-tetrahedron cut from deformation; the plane locator is clear, and mouth closeups are crops of the same fixed cohort.

Six PNGs and six rendered PDF pages were visually checked; no browser interaction test was performed.

The jagged cut boundary is formed by exposed faces of complete tetrahedra, so part of its outline reflects mesh discretization.

## Reproduction

Run from `exp/2026/09/07/tensor-active-stress` with the saved input files present. Choose a new output directory when rerunning so the completed figures remain intact:

```bash
CHERRIES_NAME=110-white-crinkle-v2 \
CHERRIES_TAGS=tensor,optimizer,learning-rate,full-field \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python src/110-render-white-crinkle.py
```

[Source110 terminal capture](../logs/110-white-crinkle-v2-terminal.log) records the exact command `.venv/bin/python src/110-render-white-crinkle.py`, completed in **0:00:09.904680**, and its [Comet run](https://www.comet.com/liblaf/apple/5c73d8d4e6164319b30fc26da6108eb5). Downloads contain this report, source110 receipt, manifest, renderer source copies, terminal capture, and the figure archive. They omit the large VTU/NPZ states; rerendering requires the manifest-pinned saved files already in this experiment checkout. Publication verification is recorded separately.
