# MouthOpen full-pose activation fit

The MouthOpen fit used full-S PSD6 activation with smoothness weight `η=7.2e-06` (10x the earlier `7.2e-7` baseline). It completed the declared 200-attempt budget with 199 accepted updates. Reference-area-weighted position RMS changed from **3.418 mm** to **1.344 mm**; normal-angle RMS changed from **8.780°** to **2.686°**. The run budget is not an optimizer-convergence claim.

![Neutral, transferred target, contact-off continuation, fitted shape, and fitted activation](../data/61-mouthopen-fit/mouthopen-trial-preview.png)

[Full-resolution figure](../data/61-mouthopen-fit/mouthopen-trial.png) · [Sampled principal-direction view](../data/61-mouthopen-fit/mouthopen-principal-directions-preview.png) · [Independent CPU analysis](../data/60-mouthopen-fit-analysis-002/analysis.json) · [Render manifest](../data/61-mouthopen-fit/manifest.json) · [Gradient-balance diagnostic](../data/56-mouthopen-fit-reuse/gradient-balance.json) · [Zero-activation forward report](49-contact-off-forward-results.md)

## Fit result

| Metric | Zero activation, full jaw | Fitted activation |
| --- | ---: | ---: |
| Position RMS (reference-area weighted) | 3.418081 mm | 1.343749 mm |
| Surface-normal angle RMS (reference-area weighted) | 8.780061° | 2.685633° |
| Inverted tetrahedra | 172 | 203 |
| Minimum J = det(F) | -16.915994 | -16.897725 |
| Inverted rest-volume fraction | 0.00004307 | 0.00004764 |

Stage 56 made **200 attempts**, accepted **199** and skipped **1**; status: `completed_attempt_budget`. Stage 55 was interrupted after 9 attempts (5 accepted, 4 skipped; status `interrupted_for_newton_search_policy_trial`). Stage 56 restarted from the same verified zero-S, full-jaw endpoint with fresh Adam and the reuse-only Newton search-shift policy.

The full-S smoothness-to-L2 gradient ratio is **0.1905**, measured with the dual effective-volume-weighted norm over the symmetric activation tensors: `η ||∇S R|| / ||∇S L2||`. The normal-loss gradient is excluded. The endpoint diagnostic's independent forward solve changed saved displacement by at most **0.000 μm** componentwise, so this ratio uses the saved activation tensor and that diagnostic displacement.

The graph roughness is `54.6612` and decomposes exactly into eigenvalue-amplitude roughness `26.222` plus orientation roughness `28.4392`. The audit verified this identity for the saved PSD tensor field.

Visual inspection of the comparison and sampled direction preview shows coherent patches alongside local directional variation around the mouth and chin. This MouthOpen trial has not produced a uniformly ordered activation field. There is no MouthOpen 1x control in this trial, so it does not isolate the effect of the 10x smoothness coefficient. The different target and full PSD6 parameterization also prevent a controlled quantitative comparison with the earlier fixed-axis Smile sweep.

## Numerical and geometry diagnostics

The 200 saved successful primal/adjoint evaluations had physical free-force residuals from `7.28e-13` to `9.88e-11` (absolute tolerance `1.0e-10`) and nonzero-RHS relative adjoint residuals from `9.26e-08` to `1e-07` (tolerance `1.0e-07`). Skipped proposals have no successful solve receipt.

The complete FEM boundary audit detected self-intersections at the zero-activation forward endpoint. The fitted endpoint also has detected self-intersections. The zero-activation forward state has 172 inverted cells (minimum `J=-16.92`, inverted rest-volume fraction `4.31e-05`); the fitted state has 203 (minimum `J=-16.9`, fraction `4.76e-05`), including 26 active and 177 inactive cells. This exploratory contact-off fit used a derived mesh with 2,249 tetrahedra removed because all four vertices were fixed. There were no contact forces, separate bone obstacles, or containment check, so the result does not establish mechanical validity.

## Pose, mesh, and material assumptions

The target is a transferred MouthOpen blendshape. The jaw pose is an area-weighted rigid fit to a 27-vertex chin patch: 9.5337° rotation, 6.4797 mm translation norm, and 0.2133 mm chin RMS. This is a geometric seed, not measured bone motion. The surviving original `IsFixed` constraints and material model were retained.

Activation started from exactly zero S on the verified full-jaw endpoint, with fresh Adam and PSD6 projection. The material model is `stable-neo-hookean-native-active-strain`; skin membrane energy is `disabled: no membrane potential is added`, muscle E is 0.012 MPa, fat E is 0.0112 MPa, and aponeurosis E is 1.693 MPa (all with the recorded nu values). The skin membrane energy was disabled.

## Reproducibility and receipts

The first CPU audit stopped on a bitwise tensor-reconstruction comparison whose maximum difference was `4.44e-16`. The corrected audit allows a few floating-point ULPs for that reconstruction and retains the exact recorded solver gates. The numerical fit was not repeated; the failed pipeline and [postprocessing recovery receipt](../data/69-finalization/recovery.json) are preserved.

Working directory: `exp/2026/09/29/mouthopen-activation`. Commands and run metadata below are read from completed Cherries logs; Comet links are included only when the log records a URL. The relevant logs and their SHA-256 digests are listed in the run table.

| Stage | Recorded command | Cherries/Comet URL | Run times | Git SHA | Log SHA-256 |
| --- | --- | --- | --- | --- | --- |
| 49 zero-activation continuation | `.venv/bin/python src/49-forward-contact-off.py` | [Comet](https://www.comet.com/liblaf/apple/95179f978ae346f9a4d8f2264980acd1) | `2026-09-29 08:24:30.688068+08:00 to 2026-09-29 08:27:55.738596+08:00` | `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` | `09decc88b0f2970e603961ae6ad32d3e7c8641199c2644ffd1035faa21c6dcf9` |
| 56 activation fit | `.venv/bin/python -u src/56-fit-mouthopen-reuse.py` | [Comet](https://www.comet.com/liblaf/apple/dc4685afadab4f3baf0bfacf39b443fd) | `2026-09-29 08:39:56.219227+08:00 to 2026-09-29 09:34:05.245536+08:00` | `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` | `357ee38ca7c07e64b64f870db7dce2c603b79dccac522a9b3b2622de4a513930` |
| 60 CPU audit | `.venv/bin/python -u src/60-analyze-mouthopen-trial.py --output 60-mouthopen-fit-analysis-002 --fit data/56-mouthopen-fit-reuse` | [Comet](https://www.comet.com/liblaf/apple/ad8348610c9e4599a81c01489cad7e88) | `2026-09-29 09:36:04.208230+08:00 to 2026-09-29 09:36:08.739978+08:00` | `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` | `1e15a86ff51b5404d18ae6bb64a9a50255494ef65e1c782e836a5c2c1f024bfb` |
| 61 final renderer | `.venv/bin/python src/61-render-mouthopen-trial.py --fit 56-mouthopen-fit-reuse --output 61-mouthopen-fit` | [Comet](https://www.comet.com/liblaf/apple/00b79e683b9e4c4cb88fc687755ce650) | `2026-09-29 09:35:53.673950+08:00 to 2026-09-29 09:36:03.832784+08:00` | `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb` | `f9c6c928c62465e1346cd39ca2b8a4cfca52cf44da4b077cc1532de59b159e79` |

The independent audit verified 102 forward and 109 fit source-module receipts; those entries may reference duplicate files. Exact declared-input and frozen-source hashes remain in the stage summaries and source manifests.

[Stage 49 source manifest](../data/49-forward-contact-off/source-manifest.json) · [Stage 56 source manifest](../data/56-mouthopen-fit-reuse/source-manifest.json) · [Stage 56 summary and declared-input receipts](../data/56-mouthopen-fit-reuse/summary.json)

| Artifact | SHA-256 |
| --- | --- |
| 49 final checkpoint | `0d39dc53676daf8dcc092376282fd7ce97bddab83159ff0609cbc17d3803c09c` |
| 56 final checkpoint | `ec7ce08c996be09470ccafa2083a9ceffb2ee8ae8d729fcad09b09020dbefac2` |
| gradient-balance diagnostic | `e2eb903c391e1859a853fff94357a533851559d870db3ce9ee19bb7f454aadad` |
| independent CPU analysis | `97ab01f8f9683f4e37522a6caa52cf08b62609a51c365a8510d4e440530335d6` |
| comparison figure | `d2ec1fcadae813b36268a280bfc1d4ff17573315a0b05e4d1fa4a18f59956c12` |
| sampled direction preview | `c03bf4acb3817bd197c14325339be58758f00afb7eccd4f76fe1faa9b0f1073f` |

The forward run, fit analysis, gradient diagnostic, and renderer manifests retain the full inputs, solver receipts, source snapshots, and image hashes.
