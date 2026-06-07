# Forward Face Worklog

## 2026-06-08

- Resumed the `forward-face` experiment against the current request rather than trusting the older June 3 outputs.
- Found that the existing scripts and reports used `Expression000` and `smas_stiffness_ratio = 1.0`; the current target contract is `Expression001` with SMAS and muscle stiffness ratio `1e2`.
- Verified that `/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu` contains `Expression001`, `MuscleOrientation`, and `InFaceConvex`.
- Updated `src/10-prepare-forward-face.py` defaults to write `10-forward-face-3152k-expr001-smas100-*` artifacts from `Expression001` and `smas_stiffness_ratio = 1e2`.
- Updated `src/20-forward-face.py` defaults to read the new prep artifacts, keep the selected Zygomaticus-major activation family, and solve with muscle plus SMAS stiffness ratio `1e2`.

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
- Ran the final canonical Cherries command without CLI overrides. It converged with PNCG `primary_success` in 619 steps and wrote `data/20-forward-face-3152k.vtu`, `data/20-forward-face-3152k.png`, and `data/20-forward-face-3152k-summary.json`.
- Final metrics: face RMS 0.126191 cm, face max 0.826099 cm, lip-top RMS 0.229352 cm, lip-top max 0.456870 cm, lip-bottom RMS 0.123553 cm, lip-bottom max 0.394535 cm.
- Comet printed final URL `https://www.comet.com/liblaf/apple/9e4c4457d5004d40b4b497940bd9496e`, but shutdown warned that online logging failed. Local artifacts are the trusted evidence.
- Wrote final report `docs/10-forward-face-3152k.md`.
- Follow-up request: run the same selected activation on the 515k mesh without SMAS.
- Added script options `--output-stem` and `--use-smas false`. In prep, no-SMAS mode sets `BackgroundFraction = 1 - MuscleFraction` and `SmasStiffnessFraction = 0`; in forward, no-SMAS mode skips the SMAS potential.
- Ran 515k no-SMAS prep from `42-expression-muscle-orientation-515k.vtu` with output stem `10-forward-face-515k-nosmas`. It produced 58651 points, 253876 tetrahedra, 420 Zygomaticus-major activation tetrahedra, 6787 face target points, and zero SMAS stiffness volume.
- Ran 515k no-SMAS forward with the same activation `(-0.87, 0.65, 0.65, 0.0, 0.0, 0.0)` and output stem `20-forward-face-515k-nosmas`.
- 515k no-SMAS forward metrics: PNCG `primary_success` in 322 steps, face RMS 0.071006 cm, face max 0.599423 cm, lip-top RMS 0.096671 cm, lip-top max 0.257022 cm, lip-bottom RMS 0.052231 cm, lip-bottom max 0.184981 cm.
- Verified `data/20-forward-face-515k-nosmas.vtu` keeps rest coordinates, has required displacement/activation arrays, has 420 active tets, has zero inactive activation, and has zero SMAS stiffness fraction.
- Wrote follow-up report `docs/20-forward-face-515k-nosmas.md`.
- Follow-up request: use the 515k mesh and the previous forward solution as the target deformation, then run inverse physics to recover the muscle activation.
- Added `src/30-inverse-face-515k.py`. This recovery script reads `data/10-forward-face-515k-nosmas-input.vtu`, uses `data/20-forward-face-515k-nosmas.vtu` as the target, disables SMAS, and optimizes one six-component local `ActivationInv` delta for the active Zygomaticus-major tetrahedra.
- The first inverse attempt wrote the input/target meshes and a best checkpoint but did not yet write a `.vtu.series`; stopped that attempt and patched `src/30-inverse-face-515k.py` so every evaluated inverse step is appended to `data/30-inverse-face-515k-nosmas.vtu.series` for ParaView timeline inspection.
- Ran the patched inverse command as `forward-face 515k inverse nosmas activation-inv full6 series` with `--max-point-error-cm 0.08`. Comet URL: `https://www.comet.com/liblaf/apple/2a616c45e54b4557ac0f7a2c21f959da`.
- The inverse solve stopped at step 93 with `max_point_error_tol`, `passed = true`, face RMS error 0.016721 cm, face max error 0.076498 cm, all-point max error 0.085630 cm, and recovered local `ActivationInv` delta `(5.966956, -0.149711, -0.488843, -0.060450, 0.097881, 0.028487)`.
- Verified `data/30-inverse-face-515k-nosmas.vtu.series` contains 94 frames for times 0 through 93, backed by 94 VTU files under `data/30-inverse-face-515k-nosmas.vtu.d/` totaling about 1.9G.
- Wrote inverse report `docs/30-inverse-face-515k-nosmas.md`.
