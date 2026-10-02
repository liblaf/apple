# Rate-0.3 execution record

This targeted test follows the [frozen protocol](54-rate-03-plan.md). All times use Asia/Shanghai on September 9, 2026. The fitting cutoff is 12:40; the overall deadline is 13:21.

## Frozen settings

The final settings freeze completed at 11:43 and verified all 95 numerical/runtime source hashes from the original smoothed arm. [Comet record](https://www.comet.com/liblaf/apple/9d44f4eec4f443a0acf8b2c8a410cdab).

Settings SHA-256: `146730c651d23abcd871648950847ced9a850eba1ed9843f4381ad5ca1a65f55`. The rate is exactly 0.3, approximately 26.19 times smaller than the original 7.857985794554741. Initialization, epsilon, betas, smoothness coefficient, fixture, and numerical implementation are unchanged. An earlier unused freeze cited the off-arm source receipt; it was preserved under `data/54-rate-03-settings-off-reference-draft/` and replaced before fitting with a freeze citing the original smoothed arm. No fit used the earlier draft.

## First 128-update phase

Working directory: `exp/2026/09/09/activation-space-smoothness`.

```bash
run_window_s=$(.venv/bin/python -c 'import datetime,time; print(max(1,int(datetime.datetime(2026,9,9,4,40,tzinfo=datetime.timezone.utc).timestamp()-time.time())))')
timeout --signal=INT --kill-after=90 "$run_window_s" env \
  PYTHONUNBUFFERED=1 \
  CHERRIES_NAME='Learned axis smoothing on learning rate 0.3 first 128 updates' \
  CHERRIES_TAGS='activation-space,learned-axis,learning-rate,rate-03,smoothness-on' \
  .venv/bin/python src/20-run-case.py \
  --case learned-axis-smooth \
  --settings data/54-rate-03-settings/settings.json \
  --steps 128 --checkpoint-interval 16 \
  --output-dir data/55-axis-on-lr03-128
```

The first phase started at 11:46:16 and completed at 11:59:05, including normal Cherries shutdown. [Comet record](https://www.comet.com/liblaf/apple/94d2add068934433ab16cc9460be8a2c). All 128 updates were evaluated; the endpoint is also the best-fit state. Fit RMS is 4.382659 mm, motion RMS 1.745661 mm, with one inverted tetrahedron, minimum det(F) −0.370467, S(C) 73.939348, and primary residual HP 0.307523 mm. Fitting/output time is 763.78 seconds.

Initial q, C, and Z agree bitwise with the original smoothed run. Equilibrated initial displacement differs by 2.39e−14 mm RMS. The first inversion occurs at update 111, with fit 4.793328 mm, motion 1.000948 mm, and minimum det(F) −0.038787. Thus the low-rate trajectory still inverts; the later update count is not proof of improved physical validity.

## First-phase saved-state verification

The new rate-0.3 verifier was exercised on the completed immutable first phase while the continuation ran. [Receipt](../data/57-rate-03-phase128-verification/summary.json), [Comet record](https://www.comet.com/liblaf/apple/a468208140d9406c8eb37ca8e8ba32cd). All 129 evaluated states and 10 full checkpoints passed evidence checks, with Adam step 128, exact original-on q/C/Z initialization, and initial displacement difference 2.39e−14 mm RMS. This verifies the saved evidence; the recorded inversion is not an integrity failure. The receipt preserves 94 run source files, three settings-freeze files, and three verifier files.

Receipt SHA-256: `f6a1a4c95142cf8383dee53b150c766d6c6236f86563848617606c4f5cf1529d`.

The CPU-only command uses `src/57-verify-rate-03.py --run-dir data/55-axis-on-lr03-128 --settings data/54-rate-03-settings/settings.json --output-dir data/57-rate-03-phase128-verification`, with normal Cherries/Comet recording and one OpenMP/BLAS thread. Final verification will also check the new continuation states and saved-Adam lineage.

## Continuation to 256 updates

The continuation started at approximately 11:59. The final 16 first-phase updates averaged 10.09 seconds; 128 more updates projected to 1,291 seconds, with 2,425 seconds remaining until the fitting cutoff. This met a 25% runtime margin plus 90 seconds before launch.

The continuation retains exactly the same settings and restores the last evaluated controls, displacement seed, saved gradient, Adam moments, and counter. Parent states are copied, not re-evaluated. It uses a separate output directory.

```bash
run_window_s=$(.venv/bin/python -c 'import datetime,time; print(max(1,int(datetime.datetime(2026,9,9,4,40,tzinfo=datetime.timezone.utc).timestamp()-time.time())))')
timeout --signal=INT --kill-after=90 "$run_window_s" env \
  PYTHONUNBUFFERED=1 \
  CHERRIES_NAME='Continue learned axis smoothing on rate 0.3 to 256 updates' \
  CHERRIES_TAGS='activation-space,learned-axis,learning-rate,rate-03,smoothness-on,continuation' \
  .venv/bin/python src/20-run-case.py \
  --case learned-axis-smooth \
  --settings data/54-rate-03-settings/settings.json \
  --steps 256 --checkpoint-interval 16 \
  --resume data/55-axis-on-lr03-128/optimizer-latest.pt \
  --output-dir data/56-axis-on-lr03-256
```

The continuation completed normally at 12:23:12, including Cherries shutdown. [Comet record](https://www.comet.com/liblaf/apple/cd8bcfc50be744faa1a271ea1f9b4e9c). All global updates through 256 were evaluated; the endpoint is also the best-fit state. Fit RMS is 2.876797 mm, motion RMS 4.047514 mm, with 73 inverted tetrahedra, minimum det(F) −1.026526, S(C) 462.513823, and primary residual HP 0.204550 mm. This phase used 1,406.91 seconds of fitting/output time. Its accepted trace includes the original 0–128 prefix and new 129–256 states.

No extension to 384 was launched. The last 16 updates averaged 14.44 seconds, projecting 1,848 seconds for another 128 updates, while less than 17 minutes remained before the frozen 12:40 fitting cutoff. The test ended at a completed declared phase without an administrative stop. Completion is not a stationarity or physical-validity certificate.

## Final saved-state verification

The full continuation passed independent CPU-only verification, [Comet record](https://www.comet.com/liblaf/apple/56d9466adb454e0480d33702b51e54ec), completed at 12:24:44. The [final receipt](../data/57-rate-03-verification/summary.json) has SHA-256 `c94df13cc7fabdf4f6d105e451630dabec5baf4bc8a2aca551769b1ec81d079c`.

The checks cover 257 trace states, 257 solver receipts, 257 surface states, and all 18 scheduled full checkpoints. The final declared, evaluated, and Adam step is 256. The two-node 128→256 lineage passes, with the original parent checkpoint hash, copied prefix artifacts, saved gradient, moments, and counter retained. No Adam reset, administrative cutoff, or solver failure is recorded. Initial q/C/Z remain bitwise equal to the original smoothed arm; displacement differs by 2.39e−14 mm RMS. All 95 numerical source entries match the original run and current files, with no source drift. The receipt also preserves and hashes 94 distinct run source files, three settings-freeze files, and three verifier files.

The independent physical check confirms 73 inverted tetrahedra at the endpoint, including 45 active cells. Evidence integrity and physical validity are separate: the saved evidence passes, while the mesh is not inversion-free.

The command uses `src/57-verify-rate-03.py --run-dir data/56-axis-on-lr03-256 --settings data/54-rate-03-settings/settings.json --output-dir data/57-rate-03-verification`, with normal Cherries/Comet recording and one OpenMP/BLAS thread. The final three-rate comparison is generated separately from saved states by `src/58-compare-rate-03.py`.
