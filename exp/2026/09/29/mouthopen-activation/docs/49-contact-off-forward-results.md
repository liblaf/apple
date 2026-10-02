# MouthOpen reaches the full prescribed jaw pose

Allowing a small number of inverted cells and returning to the historical contact-off bulk model allowed the pruned MouthOpen mesh to reach **100% of the estimated jaw pose**. The final free-force norm is **4.21e-11**, below the declared `1e-10` threshold. There are **172 inverted cells out of 1,144,268 (0.01503%)**, and the complete FEM boundary has detected self-intersections. This is a converged numerical solution of the stated exploratory model, not a mechanically validated tissue configuration.

![Neutral, transferred MouthOpen target, and full-pose zero-activation result](../data/61-mouthopen-jaw-only/mouthopen-trial-preview.png)

[Full-resolution figure](../data/61-mouthopen-jaw-only/mouthopen-trial.png) · [Independent CPU audit](../data/60-mouthopen-analysis/analysis.json) · [Forward summary](../data/49-forward-contact-off/summary.json)

## Model and revised acceptance policy

The derived mesh removes exactly the 2,249 original tetrahedra whose four vertices satisfy `IsFixed`, as documented in the [pruning report](35-pruned-mouthopen-results.md). No further cells were deleted. The remaining 26,276 fixed vertices retain their original classification; 5,989 of them belong to the prescribed mandible. Other fixed vertices stay at rest. All original free vertices and fitting-surface vertices survive.

The user clarified that a few inverted cells are acceptable. The [revised protocol](45-inversion-tolerant-protocol.md) interprets this as a stop threshold of 0.1% of retained cells, or 1,144 cells. That threshold is an experimental acceptance rule, not a mechanical-validity criterion. The material energies, estimated jaw pose, strict force tolerance, and rollback rules remain fixed.

The original bulk model has no contact force. The first trials added geometric CCD constraints; those constraints prevented the jaw from advancing even after allowing local inversions. Stage 49 removes this added feasibility constraint and records complete-boundary intersections diagnostically. That policy was stated before launching the trial. It does not claim collision-free geometry or silently equate a converged solve with valid anatomy.

## Completed forward trials

| Stage | Geometric policy | Jaw pose reached | Inverted cells | Position RMS |
| --- | --- | ---: | ---: | ---: |
| 35 | Positive volume and full-boundary CCD | 4.2679% | 0 | 7.2277 mm |
| 45 | Limited inversions; full-boundary CCD | 5.1579% | 1 | 7.1505 mm |
| 48 | Limited inversions; CCD outside inverted-cell neighborhoods | 7.5581% | 2 | 6.9431 mm |
| 49 | Historical contact-off model; intersections reported | **100%** | **172** | **3.4181 mm** |

Every saved accepted endpoint passed the declared force tolerance. Stages 35, 45 and 48 stopped at their minimum pose increment; stage 49 completed all 12 pose attempts without rejection. They form a continuation chain, not independent comparisons from the same initialization. Stage 49 took 197.3 seconds after initialization and exited successfully after Cherries shutdown.

The independent local-fold audit reproduced stage 45's last rejected carry with full-boundary CCD fraction 0.75. Removing 10 incident faces around its one inverted cell gave fraction 1.0. Stage 48 still stopped after excluding the neighborhoods of its two inverted cells: its last rejected carry had a swept-path collision even though the retained endpoint surface did not intersect. An earlier rejected carry also had a retained endpoint intersection. These distinctions are preserved in the stage summaries and [local-fold audit](../data/46-local-fold-audit/summary.json).

## Full-pose endpoint

| Quantity | Value |
| --- | ---: |
| Position vector RMS, reference-area weighted | 3.418081 mm |
| Surface-normal angle RMS, reference-area weighted | 8.780061° |
| Chin-patch vector RMS, reference-area weighted | 0.883418 mm |
| Free-force norm, existing code units | 4.214645e-11 |
| Inverted cell count | 172 |
| Inverted cell fraction | 0.0150314% |
| Share of rest volume belonging to inverted cells | 0.00430662% |
| Minimum determinant `J=det(F)` | -16.915994 |
| Complete FEM boundary self-intersections detected | Yes |

The small inverted-cell count should not be confused with mild deformation: the most negative Jacobian is approximately -16.9. This report preserves both count and severity. The extracted FEM boundary has 28 nonmanifold edges at rest; the intersection diagnostic does not establish anatomical containment or evaluate separately registered bone and eye obstacles.

The [cell-level CPU audit](../data/64-forward-inversions/summary.json) finds that all 172 inverted cells are inactive and fat-dominant. Their fixed-corner counts are 6 with one, 45 with two, and 121 with three; 43 contain both fixed cranium and fixed mandible attachments. Sixty-five cells have `J <= -1` and six have `J <= -5`. The [sorted cell records](../data/64-forward-inversions/inverted-cells.json) retain original and pruned IDs, Jacobians, volumes and attachment labels. This locates the defects but does not establish their cause or make them harmless to the coupled solve.

The jaw is prescribed from a fresh rigid estimate based on the transferred chin patch, not a measured bone trajectory. Activation is exactly zero in this endpoint. Its position RMS improves from 7.60096 mm at neutral, but the cheek and attachment-region shape still differs visibly from the target. This forward result supplies the initialization for a separately recorded activation fit; no activation field or gradient ratio is inferred from it.

## Verification and rendering

The CPU audit independently recomputed all Jacobians, position and normal errors, chin RMS, inversion count and rest-volume fraction. It verified all eight input hashes, all 102 frozen-source records, the final checkpoint hash, and all 12 accepted checkpoint hashes and force receipts. It also reconstructed every prescribed fixed displacement at the final pose and confirmed that no fully fixed tetrahedra remain.

Final checkpoint SHA-256: `0d39dc53676daf8dcc092376282fd7ce97bddab83159ff0609cbc17d3803c09c`.

The new figure applies the prepared full pose directly to the target jaw and the saved checkpoint pose directly to the forward jaw. The earlier stage-40 renderer scaled its jaw overlay twice for a partial checkpoint; that did not affect numerical checkpoints or skin rendering. The new stage-61 renderer corrects the pose semantics. Its preview was visually inspected for common camera, full-pose target/state distinction and explicit exploratory labeling. Earlier rendered files are preserved as historical artifacts.

## Reproduction and run records

Working directory: `exp/2026/09/29/mouthopen-activation`. Use new output directories when repeating a completed stage.

```bash
CHERRIES_NAME='MouthOpen pruned contact-off continuation' CHERRIES_TAGS='mouthopen,pruned,contact-off,inversions,forward' .venv/bin/python src/49-forward-contact-off.py
```

Sources: [forward continuation](../src/49-forward-contact-off.py), [CPU audit](../src/60-analyze-mouthopen-trial.py), [renderer](../src/61-render-mouthopen-trial.py). The source snapshot and exact command are retained beside the numerical checkpoint and in the [terminal log](../logs/49-forward-contact-off-terminal.log). Automatic Git commits are disabled; no commits or pushes were made. The checkout has pre-existing uncommitted work.

Comet: [forward trial](https://www.comet.com/liblaf/apple/95179f978ae346f9a4d8f2264980acd1), [CPU verification](https://www.comet.com/liblaf/apple/60ebfa8396934cce8114ffa299a4753f).

The recorded Comet summary, with logging prefixes removed and routine metadata abbreviated:

```text
Comet.ml Experiment Summary
name: MouthOpen pruned contact-off continuation
url: https://www.comet.com/liblaf/apple/95179f978ae346f9a4d8f2264980acd1
completed_pose_fraction: 1.0
numerically_converged_full_pose: 1.0
force_norm: 4.214644920750305e-11
inverted_cells: 172
inverted_rest_volume_fraction: 4.306616355451765e-05
minimum_J: -16.915993722071168
cherries/start_time: 2026-09-29 08:24:30.688068+08:00
cherries/end_time: 2026-09-29 08:27:55.738596+08:00
cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
```
