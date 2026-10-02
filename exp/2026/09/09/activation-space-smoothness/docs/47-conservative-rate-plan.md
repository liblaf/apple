# Conservative learning-rate follow-up

This follow-up tests whether reducing the learned-axis Adam rate changes the distorted optimization trajectory. It is separate from the completed original study and preserves its results.

Use rate **1.9644964486386853**, exactly one quarter of 7.857985794554741. This was already tested in the 16-update fit-only calibration: it attained 0.155 mm motion with no inverted tetrahedra. That short, low-motion pilot does not establish inversion-free behavior at larger deformation. The half-rate pilot already inverted at a similar attained motion to the original rate, so the quarter-rate candidate offers a more conservative diagnostic.

## Fixed protocol

- Reuse the unchanged `20-run-case.py`, shared model/metric code, fixture, materials, target, equilibrium and adjoint settings.
- Start each arm from the original seed-20260909 controls with initial strength 0.001, zero displacement seed, and fresh Adam state. Keep ε = 0.01 and β = (0.9, 0.999).
- Run smoothing off with λ_C = 0 and on with the original λ_C = 0.0004450069704277614. The rate is the only changed numerical setting relative to the corresponding original arm.
- Retain the original failure policy. Inversions remain recorded diagnostics; no new projection, magnitude penalty, line search, or inversion rejection is introduced in this test.
- First complete 64 updates per arm, one GPU process at a time. If both complete and measured runtime supports it, continue both to 128 using the exact saved controls, displacement, gradients, Adam moments, and counter. Do not reset Adam during continuation.
- Preserve the original 13:21 Asia/Shanghai deadline and final 90-minute reporting reserve: stop fitting by **11:51 Asia/Shanghai on September 9, 2026**. An administrative time stop is reported separately from solver failure. Any incomplete phase retains its exact last evaluated checkpoint.

## Evaluation

Compare each smaller-rate arm with its original-rate arm at actual saved states matched within 0.05 mm in both fit RMS and motion RMS. Also compare the smaller-rate off/on pair using the same rule. Report when no such match exists rather than interpolate or silently relax tolerances.

Report the first inversion and its attained fit/motion, fit and motion trajectories, inversion count and minimum det(F), per-step physical geometry/activation changes, and the existing frozen local surface score. Compare matched states, best-fit states, and attained endpoints separately. A smaller inversion count caused only by much smaller motion is not sufficient evidence of an improvement.

Success here means obtaining evidence about rate sensitivity. Solver completion does not establish stationarity, physiological validity, or an inversion-free final state. The original 10% matched surface threshold continues to govern the smoothness comparison.

The frozen settings and provenance will be stored under `data/47-conservative-rate-settings/`; new fits use `data/48-axis-off-lr-quarter-64/` and `data/49-axis-on-lr-quarter-64/`. Any 128-update continuations use new directories. No existing fit output is overwritten.
