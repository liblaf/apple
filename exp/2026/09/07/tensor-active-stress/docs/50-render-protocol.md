# Frozen face-comparison rendering protocol

## Status and purpose

This protocol was fixed while the four inverse runs were still in progress. `src/50-render-face-comparison.py` will execute only after the run outputs are complete and acknowledged. It renders saved geometry and never calls the forward model, adjoint, optimizer, or a deformation transform.

The comparison contains, in this order:

1. Raw6 historical active-strain continuity reference;
2. PSD active stress;
3. PSD active stress with smoothness;
4. PSD active stress with smoothness and soft-rank penalty;
5. the exact Smile target skin.

Raw6 is not an unrestricted active-stress control and is not used to claim an isolated effect of the PSD constraint.

## Frozen renderer and source archive

The wrapper imports the established `face-actuation-diagnosis/src/50-render-diagnosis.py` and reuses its:

- state and topology validation;
- exterior and `IsFace` surface extraction;
- front, mouth, and three-quarter cameras;
- target-skin construction;
- `DominantMaterialPhase` inference;
- browser geometry serialization and deduplication;
- Three.js viewer.

The earlier file remains unchanged. At execution, the wrapper copies the earlier renderer, itself, this protocol, and the endpoint manifest into `data/50-face-comparison/sources/`. It verifies that each archived copy has the same SHA-256 hash as its source and records both identities in `summary.json`.

Before writing `summary.json`, the wrapper inventories every other generated file by output-relative path, byte count, and SHA-256. This includes browser geometry JSON, case receipts, viewer files, static PNG/PDF figures, vendor files, and archived sources. The summary excludes itself from that map.

## Immutable geometry contract

The common reference is an actual new step-0 VTU. Its saved `RestPosition` array supplies reference points. The wrapper rejects a reference containing `ActivationFiber`, so the browser cannot silently inherit the earlier geometry-estimated fiber display.

Every endpoint and history frame remains an unchanged full tetrahedral VTU. The wrapper reads its points as the saved state, checks point count, cell count, cell types, and tetrahedral connectivity against the common reference, and extracts surfaces only in memory for rendering. It rejects filenames containing `latest`.

Each arm must have a completed `summary.json` whose status is `completed_fixed_budget` and whose primary endpoint is solver-valid step 64. A solver-failed arm is not silently replaced with a different checkpoint. If that occurs, execution stops until the manifest and comparison claim are explicitly revised.

History is exactly steps 0, 16, 32, 48, and 64 for every arm. Each frame path is explicit in the manifest. A frame must be in the same run directory as its final endpoint and must be named `step-NNNN.vtu`. A paired `step-NNNN.npz` must exist, and its scalar `step` field must equal the declared step; the wrapper reads only that scalar rather than loading the saved control or displacement arrays. Both the VTU and paired NPZ receive hashes in the history receipt. There is no interpolation.

## Static figures

Two independent comparison figures are produced:

- `face-comparison-front.png` and `face-comparison-front.pdf`;
- `face-comparison-mouth.png` and `face-comparison-mouth.pdf`.

Each figure is a one-row, five-column panel in the fixed case order. All five surfaces in a figure use one camera, parallel projection, true deformation scale, identical lighting, and uniform skin color `#d9a486`. Material colors and scalar maps are excluded from the static panels so surface shape and bumps remain visible without a color confound.

The endpoint columns show actual `IsFace` exterior triangles extracted from each saved final VTU. The target column shows only the Smile target skin constructed from `TargetDisplacement` on the common step-0 `RestPosition`; it does not fabricate a target interior tissue state. The PDFs contain the same rendered pixels and labels as their PNG counterparts.

The front camera is fitted once to the union of the compared saved geometry. The mouth camera is the established renderer's fixed rest-lip camera and is shared without per-arm reframing.

No cutaway, fiber overlay, displacement magnification, or independently fitted camera is allowed.

## Interactive viewer

The browser viewer exposes the four arms as cases and provides:

- common rest surface;
- final saved endpoint;
- exact target-skin state and overlay;
- explicit 0/16/32/48/64 history;
- front, three-quarter, and mouth cameras;
- material toggle.

The geometry displayed for rest, history, and endpoints is the full exterior boundary extracted from the unchanged full-tetrahedron VTU. `DominantMaterialPhase` is inferred exactly as the earlier renderer specifies: the class is the index of the largest saved `FatFraction`, `MuscleFraction`, or `AponeurosisFraction` on each exterior cell. The target has no material class.

The reference contains no `ActivationFiber`, and the generated viewer manifest must record `fibers: null`. Cutaway data are not generated.

The wrapper adds a system/light/dark selector after the frozen renderer writes the viewer. A `?theme=light` or `?theme=dark` query parameter selects the initial theme for embedding. This changes page and canvas colors only; it does not change geometry, camera, or material classification.

## Endpoint manifest

After the four runs complete, create `docs/50-render-manifest.json` with exact paths. The shape is:

```json
{
  "schema_version": 1,
  "title": "Tensor active-stress face comparison",
  "target_label": "Exact Smile target",
  "reference_vtu": "../data/20-raw6-reference/step-0000.vtu",
  "history_steps": [0, 16, 32, 48, 64],
  "static_material": "uniform-skin",
  "viewer_material": "DominantMaterialPhase",
  "cases": [
    {
      "id": "raw6",
      "label": "Raw6 reference",
      "endpoint_vtu": "../data/20-raw6-reference/final.vtu",
      "summary_json": "../data/20-raw6-reference/summary.json",
      "expected_final_step": 64,
      "history": [
        {"step": 0, "vtu": "../data/20-raw6-reference/step-0000.vtu"},
        {"step": 16, "vtu": "../data/20-raw6-reference/step-0016.vtu"},
        {"step": 32, "vtu": "../data/20-raw6-reference/step-0032.vtu"},
        {"step": 48, "vtu": "../data/20-raw6-reference/step-0048.vtu"},
        {"step": 64, "vtu": "../data/20-raw6-reference/step-0064.vtu"}
      ]
    },
    {
      "id": "psd",
      "label": "PSD active stress",
      "endpoint_vtu": "../data/21-psd/final.vtu",
      "summary_json": "../data/21-psd/summary.json",
      "expected_final_step": 64,
      "history": [
        {"step": 0, "vtu": "../data/21-psd/step-0000.vtu"},
        {"step": 16, "vtu": "../data/21-psd/step-0016.vtu"},
        {"step": 32, "vtu": "../data/21-psd/step-0032.vtu"},
        {"step": 48, "vtu": "../data/21-psd/step-0048.vtu"},
        {"step": 64, "vtu": "../data/21-psd/step-0064.vtu"}
      ]
    },
    {
      "id": "psd-smooth",
      "label": "PSD plus smoothness",
      "endpoint_vtu": "../data/22-psd-smooth/final.vtu",
      "summary_json": "../data/22-psd-smooth/summary.json",
      "expected_final_step": 64,
      "history": [
        {"step": 0, "vtu": "../data/22-psd-smooth/step-0000.vtu"},
        {"step": 16, "vtu": "../data/22-psd-smooth/step-0016.vtu"},
        {"step": 32, "vtu": "../data/22-psd-smooth/step-0032.vtu"},
        {"step": 48, "vtu": "../data/22-psd-smooth/step-0048.vtu"},
        {"step": 64, "vtu": "../data/22-psd-smooth/step-0064.vtu"}
      ]
    },
    {
      "id": "psd-smooth-rank",
      "label": "PSD plus smoothness and soft rank",
      "endpoint_vtu": "../data/23-psd-smooth-rank/final.vtu",
      "summary_json": "../data/23-psd-smooth-rank/summary.json",
      "expected_final_step": 64,
      "history": [
        {"step": 0, "vtu": "../data/23-psd-smooth-rank/step-0000.vtu"},
        {"step": 16, "vtu": "../data/23-psd-smooth-rank/step-0016.vtu"},
        {"step": 32, "vtu": "../data/23-psd-smooth-rank/step-0032.vtu"},
        {"step": 48, "vtu": "../data/23-psd-smooth-rank/step-0048.vtu"},
        {"step": 64, "vtu": "../data/23-psd-smooth-rank/step-0064.vtu"}
      ]
    }
  ]
}
```

After root acknowledges all endpoints, run:

```bash
cd exp/2026/09/07/tensor-active-stress
env COMET_AUTO_LOG_GIT_METADATA=false \
  COMET_AUTO_LOG_GIT_PATCH=false \
  COMET_AUTO_LOG_ENV_DETAILS=false \
  CHERRIES_NAME='Tensor active-stress face comparison' \
  CHERRIES_TAGS='cpu,face,tensor-active-stress,surface,render,comparison' \
  .venv/bin/python \
  exp/2026/09/07/tensor-active-stress/src/50-render-face-comparison.py \
  --manifest exp/2026/09/07/tensor-active-stress/docs/50-render-manifest.json
```

The output directory must be empty. `summary.json` is the artifact receipt; the viewer entry point is `viewer.html`.
