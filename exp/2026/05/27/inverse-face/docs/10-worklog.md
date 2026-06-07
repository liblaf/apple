# Inverse Face Expression001 Worklog

## 2026-06-08

- Started the fresh `exp/2026/05/27/inverse-face` run for the 3152k human-face inverse physics target.
- Confirmed that the requested source family should use `/home/liblaf/github/liblaf/melon/exp/2025/04/30/human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu` because it has `Expression001`, `MuscleOrientation`, and `InFaceConvex`.
- Confirmed that `InFaceConvex` is available as cell data on the source mesh. It selects 1,127,541 tetrahedra before extraction.
- Added `src/10-prepare-inverse-face.py` to extract the `InFaceConvex` tetra subset, target `Expression001`, fix both `IsCranium` and `IsMandible`, and write the three requested fractions:
  - fat/background: `1 - max(SmasFraction, MuscleFraction)`, `E = 1.0`, `nu = 0.49`
  - muscle: `MuscleFraction`, active, `E = 100.0`, `nu = 0.49`
  - SMAS-only: `max(SmasFraction - MuscleFraction, 0)`, passive, `E = 100.0`, `nu = 0.49`
- The existing `src/20-inverse-face.py` already uses the new `Forward` and `DifferentiableForward` library path, stable neo-Hookean materials, no collision potentials, PNCG for each forward solve with `rtol = 5e-4`, `atol = 0`, and `max_steps = 10000`, and a report-worthy `cherries.main(main)` entrypoint.
