# Stronger normal matching from neutral

Test a fivefold increase of the normalized normal weight, beta 0.05 to 0.25,
with unrestricted Raw6 activation. Run two fresh neutral fits, smoothing off
and on, for 200 Adam updates. Compare against the existing beta 0 and 0.05
results at matching update counts. Do not transfer fitted activation, forward
displacement, Adam moments, or an adjoint warm start into either new fit.

The objective is `L2 + beta*(L20/N0)*N + lambda*R`. L2 retains coefficient 1;
the normal term has 25% of the positional term's value at neutral, versus 5%
previously. This is a loss-value normalization, not a prescribed gradient or
Adam-step ratio. Fixed reference-area support, target normals, normalization,
materials, constraints, and physical-volume energy remain unchanged.
Lambda is 0 or 0.003214147722027223. Adam retains learning rate 0.3,
epsilon 0.01, betas 0.9/0.999, and the existing forward/adjoint tolerances.

Reuse the frozen `src/10-run.py` through CLI configuration. Run its
validation-only mode at beta 0.25 into `data/56-strong-validation`, requiring
the existing 2% implicit finite-difference gates. Follow with a one-update
two-branch smoke into `data/59-strong-smoke`, independently audited on CPU.
The main fits are independent fresh starts in `data/60-strong-normal`.
Record source and fixture hashes, successful solve receipts, initialization,
optimizer counters, metrics and geometry. Preserve failed proposals if any
solve fails; do not silently change the physics, weight, or tolerances.

Controls are the verified `data/40-continuation` results, which began neutral
and paused/resumed at update 100 with preserved Adam state. Their forward and
adjoint replay was finite-tolerance, not bitwise identical; the new fits do not
pause at 100. The previous replay audit quantified this small procedural
difference. Distinguish this from an objective switch or fitted warm start.

The frozen runner copies the original study's `docs/00-protocol.md` into each
output as `protocol.md`. That inherited narrative describes the original
beta 0.05, four-branch, 100-update experiment. For this extension, the executable
`config.json` and `protocol.json`, together with this stronger-normal protocol,
define the weight, branch selection and update budget. Preserve the inherited
document and add this file as `strong-normal-protocol.md` beside it.

Report position RMS, normal-angle RMS, surface-gradient RMS, target-relative
5 mm high-pass residual, activation variation R, motion, determinant minima,
inversion counts, recent objective changes and physical gradients separately.
Compare at matched evaluated/saved updates, including the latest common
inversion-free saved geometry. Completing the budget does not certify
convergence, stability, or positive physical determinants. No contact,
determinant penalty, projection, or coefficient search is added.

At setup, another GPU experiment was active on the shared RTX 4090. Leave it
untouched and do not use wall-clock time as a comparison metric. The current
97 baseline numerical sources and three fixtures hash-match the original run.

Working directory: `exp/2026/09/21/normal-matching-face`.
Use the repository `.venv/bin/python`. Main numerical command:

```bash
CHERRIES_NAME='3D face stronger normals: beta 0.25 from neutral' CHERRIES_TAGS='face,3d,raw6,normal-matching,stronger-normal,beta-025,neutral,smoothness' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python src/10-run.py --output 60-strong-normal --steps 200 --beta 0.25 --branches smooth-off-normal,smooth-on-normal --gate 56-strong-validation
```
