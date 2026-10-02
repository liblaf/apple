# Expression-general coupled endpoint audit and full-tetmesh review

## Purpose

`180-audit-expression-coupled.py` and `181-review-expression-coupled.py` extend the saved-endpoint verification and full-boundary renderer to any expression named by an inverse protocol. They are intended for the local collision-on Smile run after it has completed; they do not start a solver or modify an endpoint.

## Inputs and expression selection

Both scripts read `protocol.json` from `--run-dir`. The authoritative expression is `protocol["expression_name"]`; that name selects the corresponding transferred target slice from the verified blendshape archive. No expression name, target index, or history is supplied by the command line.

When a run is copied from paratera-4090, pass `--binding-mirror-root` as the
local `new-neutral` directory. The scripts retain the original absolute
bindings in the protocol and accept the mirror only when its file hash matches
the saved binding. They do not rewrite the protocol to local paths.

The audit rebuilds the inverse physics from the saved repaired reference and verifies:

- the original `IsFixed` DofMap and free lips;
- saved active strain, skin prestretch, fixed values, collision state, force, and intersections;
- retained-tetrahedron determinant metrics and the configured inversion allowance;
- the saved expression-specific, area-weighted positional RMS.

For a mixed objective, it also verifies that `positional_fit_rms_mm` equals the L2 position component scaled by `objective_terms.scale2_m2`. It never converts the total regularized loss into an RMS.
Normal and smoothness components are recorded as saved-component consistency in
this endpoint audit; their independent objective replay remains a separate
preflight check.

## Full-surface review

The review writes the complete original tetmesh boundary plus cranium, mandible, and eyes. It reads only the current run rows when `--parent-run-dir` is omitted, so a fresh neutral-start Smile run produces no fictitious MouthOpen lineage. If an actual parent is specified, checkpoints and endpoints are hash-bound before its rows are included.

The fit/force curve labels the explicit positional L2 RMS. Regularized rows require `positional_fit_rms_mm` and `loss_components`; legacy L2-only rows retain their saved `fit_rms_mm` interpretation.

## Commands after a valid saved endpoint

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Audit coupled Smile endpoint' CHERRIES_TAGS='smile,audit,coupled,collision-on' \
  uv run python src/180-audit-expression-coupled.py \
  --run-dir data/inverse-smile-coupled-001 \
  --binding-mirror-root "$PWD"

CHERRIES_NAME='Review coupled Smile endpoint' CHERRIES_TAGS='smile,review,coupled,full-tetmesh' \
  PYVISTA_OFF_SCREEN=true uv run python src/181-review-expression-coupled.py \
  --run-dir data/inverse-smile-coupled-001 \
  --output-dir data/review-smile-coupled-001 \
  --binding-mirror-root "$PWD"
```

No renderer or audit was run while this module was prepared because the local GPU was allocated to the active MouthOpen run.
