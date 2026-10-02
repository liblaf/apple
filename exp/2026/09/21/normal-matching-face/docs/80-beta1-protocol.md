# Beta 1 normal matching from neutral

Test beta 1, four times the previous beta .25 and twenty times beta .05,
with the same unrestricted Raw6 face activation. Run two fresh neutral fits,
activation smoothing off and on, for 200 Adam updates. Each starts with zero
controls and displacement, identity activation, zero Adam moments, and a zero
adjoint initial guess. No fitted or optimizer state is transferred.

The objective is `L2 + beta*(L20/N0)*N + lambda*R`. At beta 1, the normal and
position terms have equal loss values at neutral; their gradient magnitudes
and their contributions during fitting need not be equal. The position
coefficient remains one. Reference-area support, oriented target normals,
normalization, physical-volume energy, constraints, passive materials, and
solver tolerances remain unchanged. Lambda is zero or
0.003214147722027223, with a 5 mm activation-variation length scale. Adam uses
learning rate .3, epsilon .01 and betas .9/.999.

Reuse frozen `src/10-run.py` via CLI, preserving all original numerical sources.
First run the full implicit derivative validation at beta 1 into
`data/81-beta1-validation`, with the existing 2% acceptance threshold. The
normal-only and combined-with-smoothness cases are checked; the unsmoothed
combined derivative follows by removing the independently validated direct
q-only regularizer derivative, as documented in the previous validation.
Then run one-update smoke fits into `data/83-beta1-smoke` and independently
audit them on CPU into `data/84-beta1-verification`. Launch the two main fits
into `data/90-beta1` only after the smoke audit passes.

The final CPU audit (`src/85-verify-beta.py`, output `data/95-verification`)
checks source and fixture hashes, derivative/smoke lineage, initial state,
Adam state/counter, accepted solve receipts, and independently recomputed
endpoint geometry/objective. A main-only `beta-preflight.json` binds protocol,
derivative, smoke and previous comparison audit receipts by path and SHA256.
Keep failed proposals and solver evidence if a branch fails. Do not silently
relax tolerances, change weights, or replace failed branches with restarts.

Compare beta 1 with the existing beta 0/.05 results in `data/40-continuation`
and beta .25 in `data/60-strong-normal`. Every fit began neutral. The beta 0/.05
controls paused at update 100 and resumed with preserved Adam state under an
audited finite-tolerance forward/adjoint replay; beta .25 and the new beta 1
fits run continuously through 200. This procedural difference is retained
in comparisons, rather than described as a bitwise-equivalent trajectory.

Report position and normal RMS, surface-gradient RMS, target-relative 5 mm
high-pass normal residual, activation variation, motion, physical det(F),
inversions, own-objective tail changes and physical gradients separately.
Use matching update counts, including the latest common saved inversion-free
geometry. If a branch fails before 200, report its actual successful endpoint
and compare all eight cases at their latest common evaluated/saved update.
Completing the update budget is not convergence or mechanical-stability proof.
No contact, determinant barrier, activation projection or new coefficient
search is introduced.

The frozen runner copies the original study's `docs/00-protocol.md` to
`protocol.md`; that narrative describes beta .05/four branches/100 updates.
For this extension, executable `config.json` and `protocol.json`, together
with this file copied as `beta1-protocol.md`, define the selected beta,
branches and update budget. Preserve the inherited narrative.

At setup the RTX 4090 had about 21 GiB free and no other numerical compute
process was listed. Leave other applications untouched. Do not compare wall
time against earlier shared-GPU runs. All 97 numerical source files and three
fixture files currently hash-match the beta .25 experiment.

Working directory: `exp/2026/09/21/normal-matching-face`. Commands use the
repository `.venv/bin/python`:

```bash
CHERRIES_NAME='3D face beta 1: derivative validation' CHERRIES_TAGS='face,3d,raw6,normal-matching,beta-1,gradient-validation' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py --output 81-beta1-validation --beta 1 --branches smooth-off-normal,smooth-on-normal --validation-only true
DEBUG=1 CHERRIES_NAME='3D face beta 1: neutral one-update smoke' CHERRIES_TAGS='face,raw6,normal-matching,beta-1,neutral,smoke' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py --output 83-beta1-smoke --steps 1 --beta 1 --branches smooth-off-normal,smooth-on-normal --gate 81-beta1-validation
CHERRIES_NAME='3D face beta 1: fresh neutral fits' CHERRIES_TAGS='face,3d,raw6,normal-matching,beta-1,neutral,smoothness' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/10-run.py --output 90-beta1 --steps 200 --beta 1 --branches smooth-off-normal,smooth-on-normal --gate 81-beta1-validation
```
