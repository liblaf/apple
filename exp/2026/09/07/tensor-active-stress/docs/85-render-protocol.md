# Saved continuation geometry render protocol

## Scope

`src/85-render-continuations.py` renders only saved meshes from a manifest. It
does not solve, interpolate, deform geometry for display, use a best state in
place of an actual endpoint, construct target interior tissue, add fibers, or
make a cutaway. The shared rest/reference and the exact target skin always come
from `data/21-psd/step-0000.vtu`. Every `TargetDisplacement` selected by
`IsFace` must be finite. This frozen reference has no `TargetFinite` array, so
the renderer uses no extra validity selector.

The static figures use uniform skin color, a common physical camera within a
panel, and deformation scale 1. The viewer exports exact saved exterior
triangles and provides its inherited material toggle and system/light/dark
theme selector. Fiber and cutaway controls are removed.

## Manifest

Create `docs/85-render-manifest.json` after the selected saved states and a
completed `src/80-compare-continuations.py` receipt exist. Every file is a
`{"path": "...", "sha256": "..."}` record with a path relative to the
manifest. `static_panels` contains one to five named cases; the exact target is
added as the final panel automatically. Keep later regularization in its own
panel if adding it would exceed five named cases.

```json
{
  "schema_version": 1,
  "title": "Saved tensor active-stress continuation geometry",
  "output_dir": "../data/85-continuation-render",
  "target_label": "Exact Smile target skin",
  "reference": {
    "vtu": {"path": "../data/21-psd/step-0000.vtu", "sha256": "<sha>"},
    "npz": {"path": "../data/21-psd/step-0000.npz", "sha256": "<sha>"}
  },
  "verified_comparison": {
    "summary_json": {"path": "../data/80-continuation-comparison/summary.json", "sha256": "<sha>"}
  },
  "cases": [
    {
      "id": "baseline-80",
      "label": "Baseline continuation, global step 80",
      "kind": "completed_endpoint",
      "source80_endpoint_id": "baseline-16",
      "endpoint": {
        "step": 80,
        "npz": {"path": "../data/74-baseline-16/final.npz", "sha256": "<sha>"},
        "vtu": {"path": "../data/74-baseline-16/final.vtu", "sha256": "<sha>"}
      },
      "history": [
        {"step": 64, "npz": {"path": "../data/21-psd/step-0064.npz", "sha256": "<sha>"}, "vtu": {"path": "../data/21-psd/step-0064.vtu", "sha256": "<sha>"}},
        {"step": 80, "npz": {"path": "../data/74-baseline-16/step-0080.npz", "sha256": "<sha>"}, "vtu": {"path": "../data/74-baseline-16/step-0080.vtu", "sha256": "<sha>"}}
      ]
    },
    {
      "id": "psd-64",
      "label": "PSD, global step 64",
      "kind": "parent_or_intermediate",
      "endpoint": {"step": 64, "npz": {"path": "../data/21-psd/final.npz", "sha256": "<sha>"}, "vtu": {"path": "../data/21-psd/final.vtu", "sha256": "<sha>"}},
      "evidence": {
        "summary_json": {"path": "../data/21-psd/summary.json", "sha256": "<sha>"},
        "trace_csv": {"path": "../data/21-psd/trace.csv", "sha256": "<sha>"},
        "reason": "Original completed endpoint used as the continuation parent."
      },
      "history": [{"step": 64, "npz": {"path": "../data/21-psd/step-0064.npz", "sha256": "<sha>"}, "vtu": {"path": "../data/21-psd/step-0064.vtu", "sha256": "<sha>"}}]
    }
  ],
  "static_panels": [
    {"id": "primary", "title": "Actual saved endpoints and exact target", "case_ids": ["psd-64", "baseline-80"]}
  ]
}
```

A `completed_endpoint` requires the named endpoint in the verified source80
summary and its recorded final NPZ/VTU hashes. A `parent_or_intermediate`
requires its original completed summary, trace row at the declared global step,
and a concrete reason. The summary, trace, endpoint NPZ, and endpoint VTU must
belong to one completed run directory. The trace row is checked against metrics
recomputed from the saved VTU. A primary endpoint must use that run's
`final.npz` and `final.vtu` and equal the typed summary endpoint; an intermediate
must use its exact `step-NNNN.npz/.vtu` pair and lie strictly inside the
completed trace. This permits the shared parent and explicit saved intermediates
without pretending that they were source80 comparison endpoints.

## Validation and output

For every endpoint and history frame, the renderer requires `q`, `u`, `Q`,
`active_ids`, `step`, and `solver_valid` in the NPZ. It checks finite arrays,
step equality, shared topology, `points == RestPosition + u`, active VTK stress
matrices equal saved `Q`, and inactive matrices are zero. History consists only
of the explicit manifest frames; no missing state is inferred. Its final frame
must match both the endpoint NPZ hash and the endpoint VTU hash.

The output contains panel `<id>-front.png/.pdf` and `<id>-mouth.png/.pdf`,
`viewer.html`, its lazy geometry files and vendor modules, per-case receipts,
`viewer-manifest.json`, `summary.json`, and a SHA-256 inventory. The summary
records that the target is skin-only and no target interior state exists. It
also records the wrapper hash, resolved manifest/output configuration, output
receipt, and byte-identical archived copies of the wrapper, this protocol, and
the frozen renderer.

Run only after the manifest exists, using the project interpreter and the
noncommitting Cherries profile:

```bash
cd exp/2026/09/07/tensor-active-stress
env COMET_AUTO_LOG_GIT_METADATA=false COMET_AUTO_LOG_GIT_PATCH=false \
  COMET_AUTO_LOG_ENV_DETAILS=false \
  CHERRIES_NAME='Tensor active-stress continuation geometry' \
  CHERRIES_TAGS='cpu,face,tensor-active-stress,continuation,render' \
  .venv/bin/python \
  src/85-render-continuations.py --manifest docs/85-render-manifest.json
```
