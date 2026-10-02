# Face actuation diagnosis

This experiment checks manual facial actuation, material sensitivity, the no-skin inverse baseline, and within-muscle smoothness with independent per-tetrahedron controls. Read [the results](docs/30-face-results.md) and [the execution decisions](docs/12-execution-decisions.md).

The complete face has 228,660 vertices and 1,146,517 tetrahedra. Current-fixture tests activate 120,020 cells in 35 muscle labels; the historical-configuration comparison activates all 288,235 historical muscle cells in 103 labels. Neither smooth-field comparison reduces the number of control coordinates.

## Environment and inputs

Commands below run from this directory, using the repository's existing virtual environment. The verified dependency versions, module source identities and hardware are recorded in `data/environment.json`; source snapshots are in `data/runtime-sources/`. The Git revision is recorded separately from editable package version strings. Concurrent GPU runs share one device, so their wall times are not comparative performance measurements.

```bash
cd exp/2026/09/07/face-actuation-diagnosis
DIAG_PYTHON=.venv/bin/python
export COMET_AUTO_LOG_GIT_METADATA=false
export COMET_AUTO_LOG_GIT_PATCH=false
export COMET_AUTO_LOG_ENV_DETAILS=false
```

Normal numerical runs use Cherries with `ProfileCometNoCommit`: Comet metrics, local records and terminal logging are enabled; automatic Git commits are disabled. The installed Cherries Comet asset hook does not upload the full meshes. The local NPZ/VTU files are the geometry and control evidence.

Required inputs include the sibling `face-activation-materials/data/10-fixture/{volume.vtu,skin.vtp,summary.json}`, the regional basis `face-activation-materials/data/12-controls.npz`, and the June source data named in [the historical export](docs/11-historical-baseline.md). The material continuation additionally uses the sibling `data/25-region5-modes-fat049/final.npz`. These are identified by hashes in each run's `provenance.json`. They are not embedded in the compact reproduction ZIP.

`data/12-historical-fixture/` restores the original active set and fixation. Its source is [12-prepare-historical-fixture.py](src/12-prepare-historical-fixture.py), which refuses to overwrite an existing fixture. Its metadata states that the additional historical muscle labels have no inferred fiber directions; use it for the tensor tests, not for fiber-specific models.

## Numerical runs

Choose new, empty output directories for reruns. The commands shown use new reproduction names, leaving the archived outputs intact. New inverse trend runs use a 64-step ceiling below; the archive retains its original configured budgets and actual termination records. The regional continuation remains a 40-step comparison.

```bash
CHERRIES_NAME='Manual face activation reproduction' \
CHERRIES_TAGS='face,manual,skin-comparison,reproduction' \
"$DIAG_PYTHON" src/10-manual-activation.py \
  --output-dir data/10-manual-reproduction
```

This solves three muscle patterns, each at 10%, 30% and 50% prescribed natural contraction, at skin factors 0 and 0.12. Each pattern/skin pair starts at rest and uses the same numerical load continuation. The reported muscle shortening comes from `F`, separately from the active map.

```bash
CHERRIES_NAME='Manual face material sensitivity reproduction' \
CHERRIES_TAGS='face,manual,material-sensitivity,reproduction' \
"$DIAG_PYTHON" src/11-manual-sensitivities.py \
  --baseline data/10-manual-reproduction/skin-0.00/smile-elevators/c50/state.npz \
  --baseline-diagnostics data/10-manual-reproduction/skin-0.00/smile-elevators/c50/diagnostics.json \
  --output-dir data/11-material-reproduction
```

The three sensitivities change selected-muscle Lamé parameters by ×10, all fat by ×0.1, or all aponeurosis by ×0.1. Every case uses the same accepted baseline controls and displacement seed. These are changes in passive stiffness and active-strain stress together, not calibrated independent active-force scales.

```bash
CHERRIES_NAME='Current-fixture Raw6 no-skin reproduction' \
CHERRIES_TAGS='face,inverse,no-skin,per-tet,reproduction' \
"$DIAG_PYTHON" src/20-run-face-inverse.py \
  --method Raw6 --skin-factor 0 --steps 64 --snapshot-interval 10 \
  --gradient-audit true --forward-rtol 1e-5 --forward-atol 1e-12 \
  --adjoint-rtol 1e-7 --output-dir data/20-raw6-reproduction

CHERRIES_NAME='Current-fixture Raw6-S no-skin reproduction' \
CHERRIES_TAGS='face,inverse,no-skin,per-tet,smoothness,reproduction' \
"$DIAG_PYTHON" src/20-run-face-inverse.py \
  --method Raw6-S --skin-factor 0 --smoothness 0.0001 --magnitude 0 \
  --steps 64 --snapshot-interval 10 --gradient-audit true \
  --forward-rtol 1e-5 --forward-atol 1e-12 --adjoint-rtol 1e-7 \
  --output-dir data/21-raw6-smooth-reproduction

CHERRIES_NAME='Constrained face continuation without determinant floor reproduction' \
CHERRIES_TAGS='face,inverse,region5-modes,floor-release,reproduction' \
"$DIAG_PYTHON" src/20-run-face-inverse.py \
  --method Region5Modes \
  --initial ../face-activation-materials/data/25-region5-modes-fat049/final.npz \
  --skin-factor 0.12 --magnitude 0.001 --smoothness 0.01 \
  --steps 40 --snapshot-interval 5 --gradient-audit true \
  --forward-rtol 1e-5 --forward-atol 1e-12 --adjoint-rtol 1e-7 \
  --output-dir data/22-floor-release-reproduction
```

The Region5Modes method uses its saved interpolation basis and magnitude penalty. Its method dispatch does not add the adjacency term, so the recorded `--smoothness 0.01` argument has no effect for that method, matching the earlier run. It continues from the archived controls/displacement but restarts the L-BFGS history. The two current-fixture per-tetrahedron runs start from zero activation and rest, and differ only by the smoothness penalty.

The initial historical Adam comparison restores the June active domain, fixation and material arrays, including the original Lamé convention. Its archived design uses the uniform Cartesian MSE objective in square millimetres, Adam learning rate 0.3 and epsilon 0.01, a declared 200-step ceiling, and the original forward/adjoint tolerances. The commands below request 64 steps for a new early-trend study. Both start from rest and zero activation; only the smoothness weight differs. The [complete matched contract](docs/12-matched-historical-adam.md) includes the 501,409-edge graph and the weight-selection rationale. The actual initial runs were interrupted at recorded steps 54 and 60 without terminal summaries; their exit cause was not captured.

```bash
CHERRIES_NAME='Historical Raw6 Adam reproduction' \
CHERRIES_TAGS='face,inverse,historical-matched,raw6,adam,no-skin,reproduction' \
"$DIAG_PYTHON" src/30-run-historical-adam.py \
  --output-dir data/30-historical-raw6-reproduction --smoothness-weight 0 --steps 64

CHERRIES_NAME='Historical Raw6-S Adam reproduction' \
CHERRIES_TAGS='face,inverse,historical-matched,raw6-s,adam,no-skin,reproduction' \
"$DIAG_PYTHON" src/30-run-historical-adam.py \
  --output-dir data/31-historical-raw6-s-reproduction --smoothness-weight 0.0005 --steps 64
```

For historical compatibility, isolated finite evaluations with unsuccessful solver receipts can still advance Adam. Three consecutive unsuccessful evaluations restore the best valid state, halve the learning rate, clear the moments and recompute the gradient before another update. Nonfinite results and exceptions are recorded as numerical stops. The exported best state requires successful forward and adjoint solves; this is stricter than the original June best-state rule. NPZ checkpoints store solver success flags; `solver-receipts.jsonl` stores the full receipts. Rendered histories omit invalid frames. Fixed-budget completion is not a convergence certificate.

The final matched comparison continues each original method from its immutable step-50 checkpoint. Both explicitly reset Adam moments. The archived configurations declare a 150-step ceiling, but the experiments were intentionally stopped around 64 new local steps to evaluate the early trend. The initial saved files lack optimizer moments, so this protocol cannot reproduce the uninterrupted original 200-step trajectory. It retains every control coordinate and the original physics, optimizer constants and regularization weights. Local step 0 re-equilibrates the saved controls from the saved displacement; its displacement change is recorded. The [continuation record](docs/18-interruption-and-continuation.md) gives the actual stopping steps and receipts.

For a new trend comparison, the following commands request 64 local steps directly. They preserve the optimizer settings and source checkpoints, but do not reproduce the timing of the archived graceful stop signals.

```bash
CHERRIES_NAME='Historical Raw6 Adam controlled continuation reproduction' \
CHERRIES_TAGS='face,inverse,historical-matched,raw6,adam,continuation,reproduction' \
"$DIAG_PYTHON" src/35-continue-historical-adam.py \
  --source-run data/30-historical-adam-raw6 \
  --source-checkpoint data/30-historical-adam-raw6/step-0050.npz \
  --output-dir data/38-historical-raw6-continuation-reproduction \
  --smoothness-weight 0 --steps 64

CHERRIES_NAME='Historical Raw6-S Adam controlled continuation reproduction' \
CHERRIES_TAGS='face,inverse,historical-matched,raw6-s,adam,continuation,reproduction' \
"$DIAG_PYTHON" src/35-continue-historical-adam.py \
  --source-run data/31-historical-adam-raw6-s \
  --source-checkpoint data/31-historical-adam-raw6-s/step-0050.npz \
  --output-dir data/39-historical-raw6-s-continuation-reproduction \
  --smoothness-weight 0.0005 --steps 64
```

The continuation writes an atomic `resume.pt` after every completed evaluation, including controls, displacement, gradient, Adam moments, counters and random-generator states. Repeating the same command for an interrupted run resumes that bundle after checking source and input hashes. This applies to the new continuation only; the original step-50 moment reset remains part of its experimental history. A later one-line tensor-copy correction was applied after graceful checkpoints at local steps 24 and 23. The full optimizer states were preserved, no additional moment reset occurred, and the old/new sources and transition evidence remain in the [execution record](docs/18-interruption-and-continuation.md#checkpoint-copy-correction-during-the-continuation).

The current-fixture inverse solver requires successful forward and derivative solves before accepting a trial. A numerical line-search stall has its own status. Tetrahedron inversion, distortion and raw active-map indefiniteness are retained as diagnostics. The historical-compatible optimizer has separately documented numerical-failure behavior. No listed status implies anatomical validity or global optimality.

## Analysis and rendering

The graph, stress and surface checks run on the CPU. The graph and surface records identify their inputs by hash; the surface checks use explicit saved endpoints. The stress calculation evaluates prescribed active maps at fixed rest geometry and records constitutive source locations. Surface roughness is measured on the common rest skin, without changing displayed geometry.

```bash
"$DIAG_PYTHON" src/16-actuation-stress-mechanism.py
"$DIAG_PYTHON" src/41-surface-roughness.py \
  --manifest docs/41-surface-roughness-manifest.json \
  --output data/41-surface-roughness-reproduction/summary.json
```

The default stress diagnostic writes its established output path; run in a copied experiment directory if retaining that existing CPU record. For final audits and rendering, use the final manifests named in the report. Pending or failed inverse endpoints are not substituted with `latest.npz`. The final renderer uses literal saved vertex coordinates and topology, a common camera, and 1× displacement. Lighting normals can be averaged without changing positions. Supplied-target overlays contain observed skin only; they do not assign target motion to interior tissue.

With both continuations terminal, build the common trace audit, ten-case surface comparison and exact saved checkpoint exports. Use these output names in a copied experiment directory; the exporters refuse to overwrite an existing artifact directory.

```bash
"$DIAG_PYTHON" src/34-audit-historical-adam-progress.py \
  --raw6-dir data/38-historical-adam-raw6-continuation \
  --raw6-s-dir data/39-historical-adam-raw6-s-continuation \
  --output-dir data/48-matched-historical-adam-continuation
"$DIAG_PYTHON" src/44-final-surface-comparison.py \
  --manifest docs/44-final-surface-comparison-manifest.json \
  --output data/44-final-surface-comparison
"$DIAG_PYTHON" src/33-export-adam-checkpoints.py \
  --run-dir data/38-historical-adam-raw6-continuation \
  --output-dir data/46-historical-adam-raw6-continuation-history
"$DIAG_PYTHON" src/33-export-adam-checkpoints.py \
  --run-dir data/39-historical-adam-raw6-s-continuation \
  --output-dir data/47-historical-adam-raw6-s-continuation-history
"$DIAG_PYTHON" src/57-build-final-diagnosis-manifest.py \
  --inventory docs/58-final-viewer-inventory.json \
  --promote-ready --output docs/60-final-diagnosis-render-manifest.json
"$DIAG_PYTHON" src/50-render-diagnosis.py \
  docs/60-final-diagnosis-render-manifest.json \
  --output data/60-final-diagnosis-viewer
"$DIAG_PYTHON" src/61-validate-diagnosis-artifacts.py \
  --viewer data/60-final-diagnosis-viewer \
  --inventory docs/58-final-viewer-inventory.json \
  --receipt data/61-final-viewer-validation.json
```

After completing the report, publish and validate its downloadable records:

```bash
"$DIAG_PYTHON" src/60-publish-diagnosis.py \
  --viewer-bundle data/60-final-diagnosis-viewer \
  --report docs/30-face-results.md --site site
"$DIAG_PYTHON" src/61-validate-diagnosis-artifacts.py \
  --viewer data/60-final-diagnosis-viewer \
  --inventory docs/58-final-viewer-inventory.json \
  --report docs/30-face-results.md --site site \
  --receipt data/61-final-site-validation.json
```

The site retains the earlier `matched-step20/viewer.html` and `continuation-step10/viewer.html` comparisons with their explicit interim labels. Published geometry is checked against its source arrays, and the downloadable archives have integrity records. The final publication checks and private HTTP address are recorded in the report.

## Output inventory

| Directory | Evidence |
| --- | --- |
| `data/10-manual-activation` | 18 manual equilibrium states and diagnostics |
| `data/11-manual-sensitivities` | Preserved failed launch; stopped before any solve because a receipt requested a nonexistent material key |
| `data/11-manual-sensitivities-v2` | Three successful exact-control material replays |
| `data/11-historical-no-skin` | Converted saved June endpoint; no new solve |
| `data/12-historical-fixture` | Original active mask and fixation, with provenance |
| `data/15-control-graph` | Current active-graph coverage audit |
| `data/16-actuation-stress` | Constitutive stress derivation and finite-difference checks |
| `data/17-actuation-stress-figure` | Separate PNG/PDF figure from the frozen three-case stress record |
| `data/20-raw6-no-skin` | Current-fixture unregularized inverse trajectory |
| `data/21-raw6-smooth-no-skin` | Matched current-fixture smooth inverse trajectory |
| `data/22-region5-no-floor` | Constrained inverse continuation with the geometric floor removed |
| `data/41-surface-roughness` | Initial historical/manual/material surface metrics |
| `data/45-diagnosis-comparison` | Initial endpoint collector and exact regularizer audit |
| `data/56-completed-manual-material-sensitivities` | Inspected true-scale manual/material figures and browser assets |
| `data/30-historical-adam-raw6`, `data/31-historical-adam-raw6-s` | Original interrupted Adam runs, recorded through steps 54 / 60 |
| `data/38-historical-adam-raw6-continuation`, `data/39-historical-adam-raw6-s-continuation` | Completed early-trend runs, terminal local steps 65 / 64, selected best 50 / 64 |
| `data/44-final-surface-comparison` | Ten-case metrics on the common rest surface and declared activation graphs |
| `data/46-historical-adam-raw6-continuation-history`, `data/47-historical-adam-raw6-s-continuation-history` | Exact verified best endpoints and valid saved checkpoints through local step 60 |
| `data/48-matched-historical-adam-continuation` | Shared trace prefix through local step 64, source hashes and separate termination records |
| `data/68-trend-completion.json` | Both clean process exits, intentional stop requests and retained checkpoint hashes |
| `data/69-completed-step60-surface` | Equal-step-60 fit, motion and surface high-pass comparison |
| `data/70-final-comparison-figure` | Separate PNG/PDF presentation of the frozen ten-case metrics |
| `data/60-final-diagnosis-viewer` | Completed 27-case inspection bundle with actual saved geometry |
| `site` | Final report, viewer, retained interim comparisons and downloadable records |

The figure ZIP contains separately reusable PNG/PDF assets linked from the report. The reproduction ZIP contains source snapshots, configurations, traces, compact solver/audit records and a manifest. The complete source face, full tetrahedral states, activation arrays and historical VTKHDF history remain local, with hashes and paths in the records. The browser bundle contains the saved surface/cutaway geometry required for inspection; it is not a substitute for the complete numerical state.
