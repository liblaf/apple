# Local skin prestrain ablation results

The prescribed local 1% contraction gives a modest cheek improvement at nearly matched global fit and motion: lateral-cheek residual roughness falls by 19.54% (0.031900 mm), and lower-cheek/jaw roughness by 10.86% (0.015943 mm). Mouth-corner roughness and nasolabial fit do not improve. After the same 200-update budget, global fit RMS is 1.397767 mm without skin, 2.608769 mm with identity skin, and 2.625398 mm with local contraction. The tested membrane configuration therefore retains a substantial fitting penalty despite the local gains.

## Purpose and frozen protocol

This experiment separates the immediate effect of adding skin and local one-percent skin contraction from the effect of re-optimizing the muscle controls. The three cases are `no-skin`, `skin-zero` (elastic skin with identity activation), and `skin-local-1pct` (the same skin with a material-fixed local contraction field).

The local field is constructed once from the frozen target-space mouth-corner, lateral-cheek, and lower-cheek boxes. It is carried to rest skin through `GlobalPointId`, uses a 5 mm rest-geodesic interior taper, and has zero activation on all 1,315 triangles incident to the protected right nose-to-mouth vertices. The field receipt and validation are [summary.json](../data/10-prestrain-field/summary.json) and [validation.json](../data/10-prestrain-field/validation.json). The validation status is `passed_cpu_prestrain_field_validation`.

Every run begins from the canonical muscle-control checkpoint at step 200. When skin is enabled, its membrane constants are E = 0.2 MPa, nu = 0.46, and thickness = 1 mm. Forward and adjoint relative tolerances are identically 5e-4; the absolute tolerance is 1e-10. Each re-fit uses that unchanged control at its evaluated step 0 and fresh zero Adam moments; step 0 is a re-equilibrated forward state, **not** an optimized state. Warm-starting changes the reached gradient norm, not these settings. Adam uses `lr=0.3`, `eps=0.01`, and `betas=(0.9, 0.999)` for exactly 200 updates. The objective is uniform finite-`IsFace` Cartesian coordinate MSE times 1e6, without regularization. Completion at step 200 is evidence of the specified budget only; it is not a stationarity claim.

The primary local bumpiness measurement is the area-weighted 5 mm high-pass RMS of the **normal residual**: it measures remaining small-scale normal error against the target. Its companion is the 5 mm high-pass RMS of **normal displacement**: it measures small-scale surface motion itself. A change in the latter alone does not establish improved target agreement.

## Commands and experiment receipts

All six executions used `.venv/bin/python src/20-run-case.py` from `exp/2026/09/08/local-skin-prestrain`, with the frozen historical fixture and `data/10-prestrain-field/skin-prestrain.npz`.

| Case and stage | Command suffix | Comet receipt | Current state |
| --- | --- | --- | --- |
| no skin, initial forward | `--case no-skin --stage forward` | [3a96fba8](https://www.comet.com/liblaf/apple/3a96fba829a64778b41e263e6e834efd) | completed |
| skin zero, initial forward | `--case skin-zero --stage forward` | [4b9b2182](https://www.comet.com/liblaf/apple/4b9b2182fbf749b2aa44d516eaab537e) | completed |
| local 1%, initial forward | `--case skin-local-1pct --stage forward` | [51a73575](https://www.comet.com/liblaf/apple/51a735755373499190c69a1931d620cc) | completed |
| no skin, 200-update re-fit | `--case no-skin --stage refit` | [b7970958](https://www.comet.com/liblaf/apple/b7970958635e479e87307ecff8e26dc7) | completed |
| skin zero, 200-update re-fit | `--case skin-zero --stage refit` | [3bf0a06f](https://www.comet.com/liblaf/apple/3bf0a06f4e2447d4ab562651c59a9513) | completed, best = step 200 |
| local 1%, 200-update re-fit | `--case skin-local-1pct --stage refit` | [54cf8541](https://www.comet.com/liblaf/apple/54cf85413dcd4dc78505a6b910366bee) | completed, best = step 200 |

The terminal receipts are under [logs](../logs/). Their provenance, per-step CSV, solver receipts, every-step skin surface, and every-tenth full state are saved under the corresponding `data/20-forward-*` and `data/30-refit-*` directories.

## Initial fixed-control forward states

These are the original completed forward runs at the canonical control. The forward-only four-panel renders and exact three-plane NLF sections are saved in [data/21-forward-comparison](../data/21-forward-comparison/summary.json).

| Case | Fit RMS (mm) | Motion RMS (mm) | Right NLF fit RMS (mm) | Mouth residual HP 5 mm (mm) | Mouth displacement HP 5 mm (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| no skin | 1.763031 | 4.485370 | 2.536021 | 0.288089 | 1.026365 |
| skin, identity | 3.346025 | 3.390272 | 6.237368 | 0.661486 | 0.510581 |
| skin, local 1% | 3.369840 | 3.403631 | 6.274468 | 0.667006 | 0.506955 |

This is a fixed-control receipt, not an equal-fit comparison. It shows that introducing skin changes equilibrium strongly; the local field’s change relative to identity skin is small at this fixed control.

## Re-equilibrated fixed-q step 0

The re-fit runs evaluate the unchanged canonical control before any Adam update, with identical rtol=5e-4 and atol=1e-10 settings. This is the primary fixed-q comparison because the skin forward residuals differ from the earlier initial-forward residuals by an amount comparable to the local-field effect.

| Case | Fit RMS (mm) | Motion RMS (mm) | Right NLF fit RMS (mm) |
| --- | ---: | ---: | ---: |
| no skin | 1.763031 | 4.485370 | 2.536021 |
| skin, identity | 3.354705 | 3.414252 | 6.247633 |
| skin, local 1% | 3.373091 | 3.412276 | 6.277777 |

| Case | Mouth residual HP (mm) | Lateral residual HP (mm) | Lower-jaw residual HP (mm) |
| --- | ---: | ---: | ---: |
| no skin | 0.288089 | 0.140279 | 0.149427 |
| skin, identity | 0.663503 | 0.146761 | 0.131663 |
| skin, local 1% | 0.667747 | 0.126563 | 0.125159 |

The solver receipts establish successful re-equilibration with identical tolerances. Forward gradient norms and nonzero pre-update Adam gradients remain convergence receipts, not primary outcome measures.

## Completed no-skin 200-update reference

The no-skin re-fit completed its fixed 200-update budget with best data fit at step 200: fit RMS **1.397767 mm**, motion RMS **4.818188 mm**, and right NLF fit RMS **1.934593 mm**. Its mouth normal-residual high-pass RMS was **0.263264 mm**, while mouth normal-displacement high-pass RMS was **1.028850 mm**. This is the no-skin reference used in the completed comparison below.

## Completed 200-update comparisons

All three re-fits completed their specified 200-update budget, and the best saved data fit for each was its step-200 state. The final saved-state comparison receipt is [data/40-comparison/summary.json](../data/40-comparison/summary.json); its compact numeric table is [summary-table.csv](../data/40-comparison/summary-table.csv). Independent artifact verification passed in [data/50-verification/summary.json](../data/50-verification/summary.json): it checked all 606 evaluated states and 606 saved skin-surface states, 552 source-snapshot records across the six runs, exact canonical control at step 0, fresh optimizer budgets, and final checkpoint states.

| Case at best step 200 | Fit RMS (mm) | Motion RMS (mm) | Right NLF fit RMS (mm) |
| --- | ---: | ---: | ---: |
| no skin | 1.397767 | 4.818188 | 1.934593 |
| skin, identity | 2.608769 | 3.590157 | 4.994049 |
| skin, local 1% | 2.625398 | 3.573577 | 5.035153 |

The identity membrane has a 1.211002 mm higher global fit RMS and a 3.059456 mm higher right-NLF fit RMS than the no-skin re-fit at their separate best states. The local field has a 1.227631 mm higher global fit RMS and a 3.100560 mm higher NLF fit RMS than no skin. Those best-state differences describe the whole-membrane comparison under this fixed protocol; they are not matched-motion local-field estimates.

### Local 1% versus identity skin at matched fit and motion

The saved step-200 states provide a comparison within tolerance for both skin cases: identity skin has fit/motion RMS 2.608769/3.590157 mm and local 1% has 2.625398/3.573577 mm. Their absolute differences are 0.016629 mm in fit and 0.016580 mm in motion, each within the predeclared 0.05 mm tolerance. The three-panel frozen NLF view is [common-overlap-skin-zero-skin-local-1pct-nasolabial-region.png](../data/40-comparison/common-overlap-skin-zero-skin-local-1pct-nasolabial-region.png); the five matched views and matching receipt are recorded in the [summary](../data/40-comparison/summary.json).

![Lateral cheek at step 200: identity skin, local 1% contraction, and target smile](../data/40-comparison/common-overlap-skin-zero-skin-local-1pct-region2-lateral-cheek.png)

The same camera and lighting show shallower lateral-cheek grooves with local contraction; substantial unevenness remains relative to the target. These are the saved triangle surfaces, without geometric smoothing or displacement scaling.

| 5 mm normal high-pass RMS (mm) | Identity skin | Local 1% | Local minus identity |
| --- | ---: | ---: | ---: |
| Mouth residual | 0.556690 | 0.559249 | +0.002559 |
| Lateral-cheek residual | 0.163277 | 0.131377 | -0.031900 |
| Lower-cheek/jaw residual | 0.146830 | 0.130887 | -0.015943 |
| Mouth displacement | 0.537599 | 0.535893 | -0.001706 |
| Lateral-cheek displacement | 0.158522 | 0.125284 | -0.033238 |
| Lower-cheek/jaw displacement | 0.168518 | 0.156769 | -0.011750 |

At matched global fit and motion, local 1% reduces the residual high-pass measurement in the lateral cheek by 0.031900 mm (19.54%) and lower cheek/jaw by 0.015943 mm (10.86%), while increasing mouth-corner residual high-pass by 0.002559 mm (0.46%). The companion displacement high-pass falls in all three regions. These are modest absolute, regional gains rather than a uniform reduction of local residual bumpiness; the protected mouth-adjacent fold is not improved by this field in this run. The matched NLF fit itself is 0.041104 mm worse for local 1% (5.035153 versus 4.994049 mm).

### No common overlap with no skin

The comparator exhaustively tested actual evaluated skin-zero/no-skin state pairs for both `|Δfit| ≤ 0.05 mm` and `|Δmotion| ≤ 0.05 mm`. None exists. Its nearest-to-identity-best saved no-skin state is step 0, still 0.845738 mm apart in fit and 0.895213 mm apart in motion, so it was rejected rather than rendered as a matched comparison. The no-skin and skin curves occupy separate fit-motion ranges in [metrics-vs-fit.png](../data/40-comparison/metrics-vs-fit.png).

The completed best-state NLF sections are [equal-budget-best-fit-nasolabial-horizontal-sections.png](../data/40-comparison/equal-budget-best-fit-nasolabial-horizontal-sections.png). They are exact affine intersections of the saved skin triangles at y = 2.170, 2.180, and 2.190 m; lines are neither joined nor smoothed. The all-region residual and displacement curves are [roughness-vs-fit.png](../data/40-comparison/roughness-vs-fit.png).

These results establish the field’s measured geometric tradeoff under the prescribed 1% local contraction and fixed 200-update budget. The whole-skin fit penalty is large relative to the local-field differences: at separate best states, identity skin is 1.211002 mm and local skin 1.227631 mm above no skin in global fit RMS. They do not establish that the field optimizes the protected NLF fold, nor do they substitute a no-skin matched-state conclusion where the saved runs have no overlap.

## Equilibrated membrane response

The CPU membrane analysis is saved in [data/45-skin-stress/summary.json](../data/45-skin-stress/summary.json). It uses the same 2,779-triangle local support in both skin cases, with reference area 0.004294359 m². At unchanged muscle controls, re-equilibrated step 0 increases the area classified as tensile in both principal directions from 9.199% to 21.082%. After the separate 200-update re-fits, that fraction is 9.711% with identity skin and 17.758% with local contraction; 82.181% of the latter support still has one tensile and one compressive direction.

At the final states, the support-averaged principal metric stress indicators change from [-0.008017, +0.026420] MPa to [-0.005857, +0.028320] MPa. These are eigenvalues of the Koiter metric constitutive indicator, not Cauchy stresses. They support increased tensile tendency under local contraction, while showing that the patch is still predominantly under mixed tension and compression. The protected NLF region has only small indicator changes, but prescribing zero local contraction there does not isolate it from the surrounding membrane or the re-fitted muscle controls.

## Interpretation limits

This is one prescribed field, one set of skin constants, and one fixed optimization budget. The current Koiter skin term contains membrane energy only; it has no bending term. All accepted forward and adjoint solves passed their declared tolerances, and four existing Koiter tests passed in the [skin validation receipt](../data/11-skin-validation/summary.json). Completion does not certify inverse stationarity or an admissible physiological state. The final skin states each retain 9 inverted tetrahedra (2 active), and the identity/local-contraction states have 7,031/7,035 non-SPD muscle activation controls, respectively. These quantities were recorded as diagnostics and did not interrupt the fitting and bumpiness comparison.
