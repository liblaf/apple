# Variable-length continuation comparison protocol

## Scope

`src/80-compare-continuations.py` post-processes independently saved runs from
`src/74-face-continuation.py`. It does not start an optimizer, change a
checkpoint, select a best iteration, or claim inverse convergence. It uses each
saved `final.npz` / `final.vtu` and
`summary.primary_endpoint`; `best.*` is checked as provenance only.

Global `step` permits continuation blocks of different lengths to share one
horizontal axis. `local_step` records the reset-relative count in each block.
The output keeps three questions separate:

- numerical continuation validity (checkpoint chain, source/input identity,
  trace lineage, state bindings, and solver-valid rows);
- saved fixed-budget fit, motion, stress, projected-gradient mapping, and
  physical-update evidence (not model capacity or inverse convergence);
- explicitly matched regularization comparisons, which retain fit, motion,
  stress, and normalized surface content together.

The surface values use `src/40-measure-surface.py` and its finite frozen skin.
Normal high-pass content is descriptive. It must be read with the same-field,
same-ROI normalized ratio and the vector fit/motion diagnostics; a lower value
alone does not establish a smoother or better expression.

## Manifest

Create `docs/80-comparison-manifest.json` only after every included run
completed. All paths are relative to the manifest. The `common_root_checkpoint`
is the original global-64 checkpoint. Every run names its immediate parent;
that parent can be the common root, an earlier listed run's final
`optimizer-latest.pt`, or an explicit intermediate
`optimizer-step-<global_step>.pt` from an earlier listed completed
regularization run. The validator resolves every chain back to the common root
and rejects a cycle, unlisted parent, or hash mismatch. Multiple declared probes
may share the same parent.

```json
{
  "schema_version": 1,
  "title": "Variable-length tensor active-stress continuations",
  "output_dir": "../data/80-continuation-comparison",
  "surface_protocol": {
    "fixture_vtu": "../../face-actuation-diagnosis/data/12-historical-fixture/volume.vtu",
    "skin_vtp": "../../face-actuation-diagnosis/data/12-historical-fixture/skin.vtp",
    "scales_mm": [2, 5, 10],
    "primary_scale_mm": 5,
    "mouth_radius_mm": 10
  },
  "common_root_checkpoint": {
    "path": "../data/21-psd/optimizer-latest.pt",
    "sha256": "<sha256>",
    "global_step": 64
  },
  "runs": [
    {
      "id": "baseline-16",
      "label": "Baseline continuation, 16 updates",
      "role": "baseline16probe",
      "path": "../data/74-baseline-16",
      "parent_checkpoint": {
        "path": "../data/21-psd/optimizer-latest.pt",
        "sha256": "<same root sha256>",
        "global_step": 64
      },
      "expected": {
        "final_global_step": 80,
        "final_local_step": 16
      },
      "matched_group": "probe-16"
    }
  ]
}
```

Roles are exactly `baseline16probe`, `calibrated16probe`,
`selectedcontinuationblock`, or `laterregularization`. IDs must be unique.
`matched_group` means only that the manifest author intends a matched
comparison; the output still presents its actual fit, motion, stress, and
surface values.

## Required saved-run contract

Each run directory contains `config.json`, `resume.json`, `provenance.json`,
`summary.json`, `trace.csv`, `latest.npz`, `best.npz`, `final.npz`,
`best.vtu`, `final.vtu`, `optimizer-latest.pt`, and a
`step-<final_global_step:04d>.npz` / `.vtu` pair. A latest VTK file is not
required. A source-74 regularization run also has one exact
`optimizer-step-<global_step>.pt` checkpoint for every saved trace state; an
intermediate parent uses that file rather than `optimizer-latest.pt`.

`resume.json` is the source of continuation lineage. It must contain:

```text
parent_checkpoint {path, sha256}
parent_global_step
requested_additional_updates
requested_final_global_step
controls_sha256
seed_displacement_sha256
```

The first trace row must have `step == parent_global_step` and `local_step ==
0`; its final row must equal `requested_final_global_step` and the manifest
expected steps. `summary.primary_endpoint` and `summary.initial_endpoint` are
the final and initial trace rows. The summary status is
`completed_fixed_budget_continuation` and
`inverse_convergence_claimed` remains false.

`trace.csv` requires these runner-native columns:

```text
step,local_step,data_objective_mm2,area_fit_rms_mm,area_motion_rms_mm,
detF_min,inverted_tetrahedra,solver_valid,projected_gradient_mapping_rms,
physical_update_rms_mpa,cumulative_physical_update_rms_mpa
```

The validator checks `resume.controls_sha256` and
`seed_displacement_sha256` directly against the immediate parent checkpoint's
saved `q` and `u`, and checks `final.npz`, final history, and
`optimizer-latest.pt` bindings. For an explicit intermediate parent, it also
requires a unique solver-valid parent-step trace row and proves the exact
`optimizer-step-NNNN.pt` `q/u` state against its same-step NPZ and VTU
(`RestPosition`, points, and `Displacement`). The intermediate parent must be
owned by a listed completed regularization run. `Q` and `active_ids` are checked
directly from the endpoint NPZ archives; the continuation runner need not
duplicate hashes in its JSON receipts.

For source and input identity, the validator compares each run's standard
base-20 `provenance.sources` / `provenance.inputs` mappings with its immediate
parent for `experiment/20-face-inverse.py`, `experiment/face_physics.py`,
`experiment/tensor_active.py`, `experiment/tensor_controls.py`, and every
`liblaf/apple/` source, plus all input records. If a runner nests those mappings
under `provenance.base20`, that equivalent mapping is also accepted.

## Run command

Do not run comparison post-processing until the manifest and its complete run
set exist.

```bash
cd exp/2026/09/07/tensor-active-stress
env COMET_AUTO_LOG_GIT_METADATA=false \
  COMET_AUTO_LOG_GIT_PATCH=false \
  COMET_AUTO_LOG_ENV_DETAILS=false \
  CHERRIES_NAME='Tensor active-stress continuation comparison' \
  CHERRIES_TAGS='cpu,face,tensor-active-stress,continuation,postprocess' \
  .venv/bin/python \
  src/80-compare-continuations.py \
  --manifest docs/80-comparison-manifest.json
```

Common-camera endpoint skin PNG/PDF rendering is a separate later stage. It
must consume these actual-final records and cannot replace them with best
states or alter the mesh geometry.
