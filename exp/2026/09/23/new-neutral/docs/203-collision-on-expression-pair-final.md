# Collision-on expression pair: final evidence

## Scope

This report records the saved collision-on Smile 006 endpoint and reserves the MouthOpen 018 section until its independent audit and review assets exist. It does not adopt either inverse fit as a converged expression solution.

## Smile 006: saved partial endpoint

Smile 006 ended with the numerical status `joint_projection_failed` after 12 accepted continuation updates, at optimizer step 44. The fit process exited with code 1; this was a numerical proposal failure, not a deadline interruption. `fit.deadline_cleanup` and `interrupted_signals` are empty. The process started at 2026-09-30 05:17:57.646101 UTC and completed at 05:35:28.757735 UTC. This completion time includes process and Cherries shutdown; the exact time of the final numerical operation is not recorded. The serial independent audit started at 05:35:29.121486 UTC, exited 0 at 05:36:21.751650 UTC, and recorded no remaining GPU compute applications.

The terminal collector classified the saved state as `audited_partial_endpoint`. That means the endpoint passed the declared forward/contact/geometry checks but the inverse did not converge. It is not evidence of stationarity.

| Quantity | Initial continuation state | Saved final state |
| --- | ---: | ---: |
| Positional skin fit RMS | 4.951002 mm | 4.944966 mm |
| Full objective | 1.461635203 | 1.458454930 |
| Residual force | 0.000767 N | 0.000978 N |
| Force acceptance threshold | 0.010000 N | 0.010000 N |
| Internal-force threshold | 0.001000 N | 0.001000 N |
| Jaw rotation magnitude | 0.645697° | 0.521138° |
| Jaw translation magnitude | 1.985434 mm | 2.005175 mm |
| Activation RMS | 0.002987 | 0.003342 |
| Maximum absolute activation entry | 0.003277 | 0.003784 |

The saved pose vector is `[-0.0090082503, -0.0006416414, -0.0010812281, 0.0000789472, 0.0015202350, -0.0013051351]`, ordered as the recorded rotation-vector radians followed by global translation metres. The reported norms above are the saved rotation magnitude and translation magnitude, not individual coordinate bounds.

### Mixed objective and raw roughness

The objective is the area-weighted positional L2 term plus `19.93620158 ×` oriented target-normal chord error plus `1.435406514e-5 × R`, where `R` is the raw same-muscle tensor-activation roughness. The final saved components are:

| Component | Saved value |
| --- | ---: |
| Positional L2 | 0.9275305802 |
| Normal chord error before weighting | 0.0266311665 |
| Weighted normal contribution | 0.5309243035 |
| Raw roughness `R` | 0.0031967495 |
| Weighted smoothness contribution | 0.0000000459 |
| Full objective | 1.4584549296 |

The small weighted smoothness contribution does not make the raw roughness zero. The review curves must plot the raw roughness separately from the objective contributions, because they use different scales.

### Forward and geometry status

The rebuilt independent audit recorded `forward_converged=true`, `contact_valid=true`, and `valid_forward=true` under the declared retained-tetrahedron approximation policy. It found 99 inverted retained tetrahedra out of 1,144,268, an inverted fraction of `8.6518193e-5` and inverted rest-volume fraction of `2.9434289e-5`; both are within the configured maxima of 100 tetrahedra and `1e-4` retained rest-volume fraction. The minimum determinant is -1.902886. `inversion_free=false` remains separate from `valid_forward=true`.

The model excluded 2,249 fully fixed tetrahedra from bulk mechanics. It retained the full original display boundary, DofMap points, skin, and collision construction. Geometry counts and inversion policy therefore apply to retained mechanical tetrahedra, while the review surface remains the full boundary.

The target is a transferred skin target. No target volume, rigid anatomy measurement, or contact observation was supplied. A saved full-surface fit image can show the simulated skin and fitted anatomy, but it cannot validate unseen volume deformation, rigid pose, or contact against measurements.

### Bound evidence

- [Saved summary](../data/remote-smile-recovery-006/inverse-smile-coupled-006/summary.json), [protocol](../data/remote-smile-recovery-006/inverse-smile-coupled-006/protocol.json), and [endpoint](../data/remote-smile-recovery-006/inverse-smile-coupled-006/endpoint.npz).
- [Independent endpoint audit](../data/remote-smile-recovery-006/inverse-smile-coupled-006/independent-audit.json) and [terminal supervisor receipt](../data/remote-smile-recovery-006/remote-smile-control-006/job.json).
- [Collector manifest](../data/remote-smile-recovery-006/sync-manifest.json) and [storage-recovery lineage receipt](../data/recovery006-lineage-final.json).
- Final reviewed assets: [Smile review directory](../data/review-smile-coupled-006-final/), including [front target-versus-fit](../data/review-smile-coupled-006-final/target-vs-fit-front.png), [fit and force curves](../data/review-smile-coupled-006-final/fit-error-force-curves.png), and [objective components](../data/review-smile-coupled-006-final/objective-components-curves.png).

The final recovery receipt binds the parent preflight and parent independent audit at `data/remote-smile-recovery-006/`; it does not substitute same-byte copies from the earlier compact sync directory. The renderer retained strict SHA-256 checks for every bound record.

## MouthOpen 018: saved time-budget endpoint

MouthOpen 018 ended with `time_budget_exhausted` at local iteration 100 and optimizer step 473. The fit exited 0 without a cutoff signal. Its absolute solver cutoff was 2026-09-30 05:55 UTC; the numerical process was observed absent at 05:55:22.382732 UTC and Cherries shutdown was verified at 05:55:59.292859 UTC. The precise time of the last numerical operation was not independently observed. This is a time-budget stop, not a numerical convergence claim or a numerical failure.

| Quantity | Initial continuation state | Saved final state |
| --- | ---: | ---: |
| Positional skin fit RMS | 1.845591 mm | 1.751491 mm |
| Full objective | 0.144081084 | 0.122982490 |
| Residual force | 0.000753 N | 0.000875 N |
| Force acceptance threshold | 0.010000 N | 0.010000 N |
| Internal-force threshold | 0.001000 N | 0.001000 N |
| Jaw rotation magnitude | 8.739041° | 8.881471° |
| Jaw translation magnitude | 4.323133 mm | 4.307252 mm |
| Activation RMS | 0.227849 | 0.272225 |
| Maximum absolute activation entry | 0.705419 | 1.350353 |

The final pose vector is `[0.1550048741, 0.0005257891, 0.0012630970, -0.0001204232, -0.0041740553, 0.0010560202]`, ordered as the recorded rotation-vector radians followed by global translation metres.

The final objective uses the same positional-plus-normal-plus-smoothness form, with normal weight `9.039348924` and smoothness weight `6.508331226e-6`. Its components are positional L2 `0.0527609147`, raw roughness `R=14.8658133810`, weighted normal contribution `0.0701248232`, weighted smoothness contribution `0.0000967516`, and full objective `0.1229824895`. Raw roughness must remain a separate curve because it is not on the weighted objective scale.

The independent audit passed the declared approximation policy: `forward_converged=true`, `contact_valid=true`, and `valid_forward=true`; `inverse_converged=false`. It found exactly 100 inverted retained tetrahedra, the configured count limit, with inverted rest-volume fraction `1.2990688e-5` below the `1e-4` limit. The minimum determinant is -10.966098 and `inversion_free=false`. The policy applies to 1,144,268 retained mechanical tetrahedra after excluding 2,249 fully fixed tetrahedra; the renderer still shows the original complete display boundary.

The endpoint is not stationary: its final scaled gradient L1 is `0.0160632897`, above the recorded `1.7403074e-5` threshold, and the saved summary records `inverse_converged=false`. Its target remains a skin-only transferred geometry, with the same full-surface and contact-observation limits as Smile.

- [Saved summary](../data/inverse-mouthopen-coupled-018/summary.json), [protocol](../data/inverse-mouthopen-coupled-018/protocol.json), [endpoint](../data/inverse-mouthopen-coupled-018/endpoint.npz), and [independent audit](../data/inverse-mouthopen-coupled-018/independent-audit.json).
- [Completion verification](../data/inverse-mouthopen-coupled-018/completion-verification.json) and [independent final CPU verification](../data/inverse-mouthopen-coupled-018/independent-final-cpu-verification.json).
- Final reviewed assets: [MouthOpen review directory](../data/review-mouthopen-coupled-018-published/), including [front target-versus-fit](../data/review-mouthopen-coupled-018-published/target-vs-fit-front.png), [fit and force curves](../data/review-mouthopen-coupled-018-published/fit-error-force-curves.png), and [objective components](../data/review-mouthopen-coupled-018-published/objective-components-curves.png).

## Review QA

Smile final assets were inspected. The front target-versus-fit panel uses matched true scale and clearly shows the target skin separately from the fitted full boundary and anatomy. Its fit/force figure shows positional RMS, residual force in newtons, and the 0.01 N threshold without annotation overlap. The mixed-objective figure keeps weighted components and raw roughness on separate panels, and places the recovery explanation in its own panel.

MouthOpen published assets were inspected. The target-versus-fit front panel uses matched true scale, and the fitted full boundary includes the saved anatomy. Its fit/force plot keeps the objective-change arrow away from the fixed-control refinement note and labels the residual in newtons. The objective figure retains its objective-change and refinement markers while plotting raw roughness separately from weighted terms.

## Published delivery

The pair is published at the collision-on expression review (private preview omitted), with separate MouthOpen (private preview omitted) and Smile (private preview omitted) reviews. Root compared every served byte against all 16 manifest assets plus each receipt: 17 assets per expression. The overview and final live-page notice were independently fetched and compared as well. Evidence is in `data/expression-pair-publication-20260930.json` and `data/expression-pair-publication-final-pages.json`; the latter records the final overview text after spacing edits.

The final MouthOpen display uses `data/review-mouthopen-coupled-018-published`, and Smile uses `data/review-smile-coupled-006-final`. The first MouthOpen review outputs are preserved; only annotation layout changed between renders. The first Smile render was rejected because its command selected identical-byte provenance copies at different paths from the final metadata. The successful invocation used exactly `data/remote-smile-recovery-006/inverse-smile-coupled-006-recovery-control/preflight.json` and `data/remote-smile-recovery-006/recovery003-independent-audit.json`; no assertion or hash check was relaxed.

Both original numerical solvers and Cherries were finished before 14:00 Asia/Shanghai. The local follow-up physical audit and renderers are verification/visualization, not resumed inverse optimization. Local monitor `continue-mouthopen-until-convergence` and finalizer `finalize-mouthopen-before-14-00` were paused only after both audits and publications succeeded. The remote owner was instructed to verify delivery and pause its repurposed monitor. No commits or pushes were made.

## Reproduction commands and Cherries runs

All commands ran from this experiment group with the repository `.venv/bin/python`, project `TMPDIR=tmp/final-pair-runtime`, normal `ProfileJoint`, and human-readable `CHERRIES_NAME` / `CHERRIES_TAGS`. ProfileJoint explicitly sets Git commit=false.

- Audit: `src/146-audit-mouthopen-coupled.py --run-dir data/inverse-mouthopen-coupled-018` (exit 0).
- MouthOpen review: `src/147-review-mouthopen-coupled.py --run-dir data/inverse-mouthopen-coupled-018 --parent-run-dir data/inverse-mouthopen-coupled-017 --output-dir data/review-mouthopen-coupled-018-published` (exit 0).
- Smile review: `src/181-review-expression-coupled.py --run-dir data/remote-smile-recovery-006/inverse-smile-coupled-006 --parent-run-dir data/remote-smile-old003-lineage-bundle/inverse-smile-coupled-003 --binding-mirror-root data/remote-smile-recovery-006 --recovery-lineage-receipt data/recovery006-lineage-final.json --recovery-preflight data/remote-smile-recovery-006/inverse-smile-coupled-006-recovery-control/preflight.json --recovery-parent-audit data/remote-smile-recovery-006/recovery003-independent-audit.json --output-dir data/review-smile-coupled-006-final --local-data-root data --terminal-receipt data/remote-smile-recovery-006/remote-smile-control-006/job.json --terminal-bundle-manifest data/remote-smile-recovery-006/sync-manifest.json` (exit 0).

Complete Comet summary blocks remain in the logs below.

- `tmp/203-final-mouthopen-audit.log`: <https://www.comet.com/liblaf/apple/8ee00594ed7b4b16afeff8e54d4b64e7>.

- `tmp/203-final-mouthopen-render-v3.log`: <https://www.comet.com/liblaf/apple/cff0564c009742ef8268c7aea9f27c11>.

- `tmp/203-final-smile-render-v2.log`: <https://www.comet.com/liblaf/apple/083cab2846a944ce93dd7835580944f9>.

Root subsequently paused the remote owner's repurposed `continue-collision-off-expression-fits` heartbeat through the app tool, preserving its existing fields. All three computation/finalization monitors are paused. The owner may finish its current independent read-only delivery verification.
