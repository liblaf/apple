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
