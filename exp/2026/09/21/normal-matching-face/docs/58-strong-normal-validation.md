# Strong-normal preflight

The [stronger-normal protocol](55-strong-normal-protocol.md) changes only beta
from 0.05 to 0.25 in the frozen numerical runner. The full-face derivative
validation completed and exited 0 after Cherries/Comet shutdown. Its largest
relative finite-difference error was 0.00018195139912727322 (0.018195%) for the
combined positional, normal and activation-smoothing objective. Normal-only
validation had a maximum relative error of 0.01891370519875695 (1.891371%).
Both satisfy the existing 2% gate; no tolerance was changed.

The exact finite-difference cases were normal-only and combined-with-smoothness.
Coverage of the unsmoothed combined objective follows by subtracting the
independently validated direct activation penalty, not by claiming a separate
full-face finite-difference run. `Study.regularizer(q)` depends only on q and
does not enter equilibrium or the displacement derivative. Thus, at fixed q,
the two combined objectives have the same implicit adjoint contribution, with
the smoothed case adding `lambda * grad(R)`. The existing CPU check of this
same regularizer has relative derivative error 9.639826555009356e-11.

Both one-update smoke branches start with zero controls, displacement, Adam
moments and adjoint initial guess. Each completed update 1 and exited 0, with
zero inversions. At neutral, L2 is 8.656092875221388 and the normal contribution
is 2.164023218805347, exactly 25% of that L2 value. The smoke position RMS is
approximately 5.032574 mm and normal-angle RMS 8.87390 degrees.

The frozen runner enforces equality of source hashes, fixtures, normalization,
beta, smoothing scale/coefficient and Adam learning rate/epsilon with the
derivative gate. The independent CPU audit additionally binds the baseline
protocol and verification receipts, initialization, Adam updates, solver
receipts and recomputed surface/volume metrics. Its result must pass before
the main fits launch. Record final audit receipts in the results report.

The independent CPU smoke audit passed and exited 0 before the main fits
launched. Both branches completed their one-update budget with no failure.
The saved first controls match the initial-gradient Adam update to a maximum
absolute difference of 1.3877787807814457e-17. Source, fixture, derivative-gate
and all four inherited baseline links passed. The main run's
`strong-preflight.json` records the paths and SHA256 hashes of these receipts.

- [Derivative receipt](../data/56-strong-validation/gradient-validation.json)
- [Original CPU algebra checks](../data/05-normal-verification/checks.json)
- [Independent CPU smoke audit](../data/59-strong-verification/checks.json)
- [Main preflight provenance](../data/60-strong-normal/strong-preflight.json)
- [Derivative validation run](https://www.comet.com/liblaf/apple/b25dfa039d9849f693ac4f06343f5ae5)

Working directory: `exp/2026/09/21/normal-matching-face`; Python is the repository
`.venv/bin/python`.

```bash
CHERRIES_NAME='3D face stronger normals: beta 0.25 derivative validation' CHERRIES_TAGS='face,3d,raw6,normal-matching,stronger-normal,beta-025,gradient-validation' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python src/10-run.py --output 56-strong-validation --beta 0.25 --branches smooth-off-normal,smooth-on-normal --validation-only true
DEBUG=1 CHERRIES_NAME='3D face stronger normals: neutral one-update smoke' CHERRIES_TAGS='face,raw6,stronger-normal,beta-025,neutral,smoke' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python src/10-run.py --output 59-strong-smoke --steps 1 --beta 0.25 --branches smooth-off-normal,smooth-on-normal --gate 56-strong-validation
```
