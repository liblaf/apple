# Forward Face Worklog

## 2026-06-03

- Started a new Cherries experiment group at `exp/2026/05/27/forward-face/`.
- Checked the nearby inverse-face setup and the prior zygomaticus-major notes before editing.
- Verified that `42-expression-muscle-orientation-3152k.vtu` is the right source for this run because it contains both `Expression000` and `MuscleOrientation`.
- Verified that `Zygomaticus_major001_00` and `Zygomaticus_major001_01` correspond to `MuscleId` 46 and 47. Inside the `InFaceConvex` tetra subset these select 2522 active tetrahedra with about 0.842 cm^3 of muscle-fraction volume.
- Added `src/10-prepare-forward-face.py` to extract the `InFaceConvex` tetra subset, keep cranium and mandible fixed, and write the union-rule material fractions.
- Added `src/20-forward-face.py` to apply a local Zygomaticus-major activation, rotate it into world coordinates, solve with the new forward library and PNCG, and save the result as rest coordinates plus `Displacement`.
- Ran the prep through Cherries as `forward-face 3152k prep`. It produced `data/10-forward-face-3152k-input.vtu` and `data/10-forward-face-3152k-target.vtu`.
- Prep metrics: 225052 points, 1127541 tetrahedra, 2522 Zygomaticus-major activation tetrahedra, 17582 face target points, 26189 fixed cranium/mandible points, and `Expression000` face target RMS 0.306575 cm.
- Started the first forward solve with activation local delta `(-0.5, 0.2, 0.1, 0.0, 0.0, 0.0)`.
- First forward solve completed physically, but Cherries marked it failed because the script sent the string metric `forward/result = primary_success` to `cherries.log_metrics()`. Patched the script to log only numeric/bool metrics while keeping strings in the JSON summary.
- First solve metrics from `data/20-forward-face-3152k-summary.json`: PNCG `primary_success` in 210 steps, face RMS 0.005038 cm, lip-top RMS 0.003319 cm, and face RMS ratio to `Expression000` 0.0164. This is much too small.
- Ran a stronger probe with activation local delta `(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)` and output stem `20-forward-face-3152k-a087`.
- The stronger probe converged with PNCG `primary_success` in 735 steps. It produced face RMS 0.125168 cm, face max 0.818799 cm, lip-top RMS 0.226329 cm, lip-top max 0.447926 cm, and lip-bottom RMS 0.123693 cm.
- Visual snapshot `data/20-forward-face-3152k-a087.png` shows localized cheek/lip lift. It is visibly smaller than the full `Expression000` expression but no longer tiny, so selected this activation as the proper forward deformation.
- Updated `src/20-forward-face.py` so the selected activation is the default for the canonical `20-forward-face-3152k.*` run.
