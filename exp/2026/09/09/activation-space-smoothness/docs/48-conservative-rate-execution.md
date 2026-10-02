# Conservative-rate execution record

This record supplements the [frozen follow-up protocol](47-conservative-rate-plan.md). All times below use Asia/Shanghai on September 9, 2026. This pair uses an 11:51 fitting cutoff; the overall deadline remains 13:21. The later user-requested rate-0.3 follow-up has its own [revised time budget](54-rate-03-plan.md).

## Frozen settings

The settings extraction completed at 11:09 and verified all 95 source modules from the original Axis-off provenance against the current files. No numerical source was changed. [Comet record](https://www.comet.com/liblaf/apple/009d65ef220d45cab4345d72e71b8146).

The settings SHA-256 is `780f2bd88d1a7001c5ae98f5f07476377d3e0cd51dc3e28cf592fe2802630268`. The rate is 1.9644964486386853; the original smoothing coefficient is retained without retuning. A separate minimal settings file avoids retaining the original calibration's rate-selection metadata beside the new rate.

## Smoothing off, first 64-update phase

Started at approximately 11:10. [Comet record](https://www.comet.com/liblaf/apple/a2b034d30b1d41fab464bd1ce80d6dff).

Working directory: `exp/2026/09/09/activation-space-smoothness`.

```bash
run_window_s=$(.venv/bin/python -c 'import datetime,time; print(max(1,int(datetime.datetime(2026,9,9,3,51,tzinfo=datetime.timezone.utc).timestamp()-time.time())))')
timeout --signal=INT --kill-after=90 "$run_window_s" env \
  PYTHONUNBUFFERED=1 \
  CHERRIES_NAME='Learned axis smoothing off quarter learning rate 64 updates' \
  CHERRIES_TAGS='activation-space,learned-axis,learning-rate,quarter-rate,smoothness-off' \
  .venv/bin/python src/20-run-case.py \
  --case learned-axis \
  --settings data/47-conservative-rate-settings/settings.json \
  --steps 64 --checkpoint-interval 16 \
  --output-dir data/48-axis-off-lr-quarter-64
```

The independent initial audit confirms bitwise equality of q, C, and Z against original Axis-off. Initial fit is 5.309929996 mm, with no inverted tetrahedra. The equilibrated displacement differs only at numerical roundoff. The original and new optimizer provenance differ in rate and requested phase budget; fixture, objective, materials, solvers, and initialization agree.

At the first inversion, update 25, fit is 4.713890 mm and motion is 1.143371 mm. Original Axis-off first inverts at update 8 with 1.166806 mm motion. This intermediate observation shows delayed onset in update count, with a similar attained amount of motion. It does not yet establish the final trajectory or convergence behavior.

The first phase completed normally at 11:20:29, including Cherries shutdown. All 64 updates were evaluated. Fitting/output time was 615.54 seconds. The endpoint is also the best-fit state: fit RMS 2.587802 mm, motion RMS 4.358874 mm, 409 inverted tetrahedra, minimum det(F) −1.946184, S(C) 3,002.4325, and primary residual HP 0.218834 mm. These are endpoint diagnostics, not a matched comparison with the original run.

## Smoothing on, first 64-update phase

Started after the unsmoothed process completed, at approximately 11:20. [Comet record](https://www.comet.com/liblaf/apple/e78cd7f3028443b4a9a089112db383e1). It uses the same frozen settings and time cutoff. The command is identical except for:

```bash
CHERRIES_NAME='Learned axis smoothing on quarter learning rate 64 updates'
CHERRIES_TAGS='activation-space,learned-axis,learning-rate,quarter-rate,smoothness-on'
# Runner arguments:
--case learned-axis-smooth
--output-dir data/49-axis-on-lr-quarter-64
```

The first phase completed normally at 11:45:23, including Cherries shutdown; no administrative cutoff occurred. All 64 updates were evaluated. Fitting/output time was 1,457.35 seconds. The endpoint has fit RMS 5.152102 mm, motion RMS 7.308119 mm, 2,189 inverted tetrahedra, minimum det(F) −4.186610, S(C) 38,451.1786, and primary residual HP 0.609685 mm. The best fit occurred at update 45, with fit RMS 3.378381 mm. The most negative observed det(F) was −27.583275 at update 63. Successful inner solves do not imply a physically valid state or a decreasing outer objective.

Both arms first inverted at update 25: motion was 1.143371 mm off and 1.142112 mm on, compared with approximately 1.17 mm at the original rate. This is evidence that reducing the rate delayed initial inversion in update count but did not prevent inversion at comparable attained motion.

No extension of this pair was started. The smoothed trajectory had become expensive (roughly 35–45 seconds per recent update), leaving insufficient time for both 128-update continuations under the frozen 11:51 cutoff. The user then requested the separate rate-0.3 diagnostic. All original, pilot, and quarter-rate artifacts remain preserved.

Saved-state verification and comparison records are generated with `src/53-verify-conservative-rate.py` and `src/52-compare-conservative-rate.py`, respectively; they do not rerun the inverse fits. Their results are recorded separately after completion.

## Independent verification

[Verification receipt](../data/53-conservative-rate-verification/summary.json), [Comet record](https://www.comet.com/liblaf/apple/afced4248c3f44cdb967ba2e1af961d3). Completed at 11:46:53, with 130 evaluated states, 12 full checkpoints, and the original first-step equivalence thresholds passing. Both arms reached update 64. The receipt separately reports evidence integrity, completed budgets, numerical equivalence, and physical diagnostics; the evidence pass does not certify inversion-free geometry.

Receipt SHA-256: `def8fad04b8f57a36ca966e97fdc1d05b597884813843c11854443a13623ff38`.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CHERRIES_NAME='Verify quarter learning rate pair at 64 updates' \
  CHERRIES_TAGS='activation-space,learning-rate,quarter-rate,verification' \
  .venv/bin/python src/53-verify-conservative-rate.py \
  --off-dir data/48-axis-off-lr-quarter-64 \
  --on-dir data/49-axis-on-lr-quarter-64 \
  --settings data/47-conservative-rate-settings/settings.json \
  --output-dir data/53-conservative-rate-verification
```
