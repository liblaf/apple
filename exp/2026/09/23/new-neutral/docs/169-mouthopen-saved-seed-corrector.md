# Physical validation of a saved defect-corrected seed

Frozen diagnostic168 output002 reached seeds with100 inversions and no newly
inverted cells. Its additional observed determinant model target1e-6 was
missed by about1e-8, comparable with known iterative predictor variation. No
nonlinear corrector was run. Failure of that numerical model check does not
establish failure of the physical force/contact/inversion policy.

This experiment freezes audited run012 controls/displacement/moments/counters268
and binds output002's final attempt11 seed, controls, QP receipt and source
cache. It independently rechecks fixed values, target controls, old-to-seed
rigid-arc CCD, retained inversion count/volume, all previously positive cells,
and positivity of the observed seed-plus-residual forecast. It records the
stricter1e-6 model-margin failure explicitly; it does not claim that check passed.

Three independent predictor reconstructions at the exact saved target controls
measure per-cell determinant repeatability from the same old source state.
They are diagnostics only. One collision-on nonlinear corrector then starts
from the original hash-bound saved seed. The corrected state must meet internal
force1e-9, original physical force1e-8, collision, Armijo, maximum100 retained
inversions and maximum1e-4 inverted rest-volume fraction. Raw failed/corrected
states and determinants are saved before reporting acceptance.

No diagnostic physical state or optimizer history is adopted. Exact saved skin
pre-strain, IsFixed-only constraints and all-four-fixed-tet exclusion remain.
The single diagnostic has600 seconds after rebuilding physics. The full audited
preview remains run012. No commits or pushes are made.

## Launch

Launched from this experiment directory with:

```bash
TMPDIR="$PWD/tmp/saved-defect-seed-001-runtime" \
CHERRIES_NAME='MouthOpen saved seed physical corrector test' \
CHERRIES_TAGS='mouthopen,diagnostic,seed-repeatability,corrector' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u \
src/169-test-mouthopen-saved-defect-seed.py \
> tmp/169-mouthopen-saved-defect-seed-001.log 2>&1
```

The exact process identity is saved in the output's `job.json` (PID 884596,
start ticks 4227140, tool session 98725). The live pointer and four tailnet
assets were verified after startup. Ruff and compilation passed before launch.
Source snapshots and all input hashes are recorded in `protocol.json`.

## Result

Numerical work and Cherries shutdown completed with exit 0. The candidate was
rejected. Fresh force was 7.592564089054216e-10 (0.0007592564 N), below
the internal target. The corrected state had 103 retained inversions, including
new cells 18514, 155249 and 598977. RMS increased from 1.9802796612 mm to
1.981149507901 mm; Armijo also failed. This is neither an accepted
inverse update nor convergence. Source hashes and optimizer moments are intact.

All three same-control predictor repeats retained positive signs at the
previously positive cells. Their largest displacement difference from the saved
seed was about 2.85 nm. The model margin failure was near iterative variation,
but passing seed signs still did not predict the actual corrected geometry or
objective. Repeating pose-only margin adjustments is therefore unsupported.

The next frozen diagnostic will test a joint strain-and-pose proposal, using
reduced determinant gradients rather than freezing the Adam strain direction.
The audited preview remains run012. Detailed raw repeated/corrected states,
per-cell variability, forward receipts and completion verification are retained
in `data/mouthopen-saved-defect-seed-001/`.

Comet: <https://www.comet.com/liblaf/apple/2fac056c81af455ca5f3974edeff2c6b>

The final determinant changes show why seed sign alone was insufficient:

| Original tet | Saved seed J | Seed + residual forecast | Corrected J |
| --- | ---: | ---: | ---: |
| 18514 | 0.0016090749 | 9.9979741e-07 | -5.2911121e-05 |
| 155249 | 1.0112383e-06 | 2.7772714e-06 | -4.7190192e-05 |
| 598977 | 2.5958086e-06 | 9.9922403e-07 | -2.8174173e-05 |
| 656201 | 1.0023719e-06 | 1.5436592e-06 | 5.021144e-05 |
| 688235 | 9.8874559e-07 | 1.5766501e-06 | 8.7336186e-05 |

The rejected state used 160 PNCG and 16 Newton steps.
Contact remained valid; inversion count and objective decrease failed.
