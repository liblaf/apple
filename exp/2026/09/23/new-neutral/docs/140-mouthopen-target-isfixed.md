# MouthOpen target on the corrected IsFixed neutral

## Purpose

Transfer the 36 original expression displacement fields onto the independently
verified `forward-isfixed-001` endpoint. The output is a kinematic target
bundle for the MouthOpen inverse run; it does not solve the expression.

## Runs

The transfer used `src/90-transfer-blendshapes.py` with
`--run-dir data/forward-isfixed-001`, `--review-dir data/review-isfixed-001`,
and `--output-dir data/blendshapes-isfixed-001`.

- Transfer Comet run: <https://www.comet.com/liblaf/apple/a93f31b0e0b64c8099b62b004b495bf0>
- Independent audit: `src/92-audit-blendshapes.py --bundle-dir data/blendshapes-isfixed-001`
- Audit Comet run: <https://www.comet.com/liblaf/apple/43aaafbf2f2b49fb934a20d7c450ad18>

Both Cherries runs completed normally on 2026-09-29.

## Result

The bundle contains 36 expressions, 15,299 skin vertices, and 29,899 skin
triangles. Its `blendshapes.npz` fields are `expression_names`,
`skin_global_ids`, `skin_triangles`, `source_neutral_points_m`,
`new_neutral_points_m`, `expression_displacement_m`, and `target_points_m`.

For MouthOpen, the transferred source offset has RMS 7.801662716469344 mm and
maximum 25.42723399877028 mm. Reconstruction error for every target is at
most 2.220446049250313e-16 m.

The target source endpoint is
`data/forward-isfixed-001/endpoint.npz`, SHA-256
`942b6b7207c4b903755081d2bb279263b034e18048e21e8906478702fef6e61d`.
The bundle records both the canonical forward audit and the stronger corrected
IsFixed audit. The latter verifies the saved fixed mask and source IsFixed
boundary exactly. The base endpoint is converged, collision-feasible, valid,
and has zero inverted tetrahedra (`detF_min = 0.4839740442689695`).

## Audit scope

The independent transfer audit verified topology and global-ID ordering,
byte-exact preservation of each source skin displacement field, the target
identity `new_neutral + delta`, the zero-weight neutral, the full-volume
defined-mask convention, and the corrected IsFixed provenance. It does not
establish a forward equilibrium, contact feasibility, or volume validity for
the MouthOpen target itself.

## Reproducibility

The complete target and audit are under
`data/blendshapes-isfixed-001/`, with source hashes in `manifest.json` and
audit hashes in `independent-audit.json`. The inverse runner consumes this
bundle directly through its existing `blendshapes.npz` schema.
