# Blendshapes transferred to neutral 005

All **36 source blendshapes** were transferred onto the saved new neutral by
preserving their original displacement fields. The mapping uses the existing
vertex correspondence and unchanged triangle connectivity:

```text
new_neutral = repaired_reference + saved_neutral_displacement
target[name] = new_neutral + original_expression_displacement[name]
```

At arbitrary blend weights, skin coordinates are
`new_neutral + sum(weight[name] * displacement[name])`. Weight zero returns
the new neutral exactly. This follows the prior experiment's displacement
transfer convention; no rotation or deformation-gradient transformation is
applied to the offsets.

## Outputs

The bundle is [`data/blendshapes-005`](../data/blendshapes-005/).

| Artifact | Contents |
| --- | --- |
| `blendshapes.npz` | Ordered names, source/new neutral skin coordinates, unchanged offsets, all target coordinates, local triangles and volume vertex IDs |
| `neutral-with-blendshapes.vtp` | New neutral skin with all 36 vector fields |
| `neutral-with-blendshapes.vtu` | Full new-neutral volume with the original 36 vector fields and `BlendshapeDefined` mask |
| `targets/<name>.vtp` | Each individual expression at weight 1 |
| `targets.zip` | Neutral skin, all 36 target skins and README; 38 members, 43.83 MiB |
| `manifest.json` | Input/output hashes, transfer rule, ordering and neutral status |
| `independent-audit.json` | Independent coordinate, field, topology and ZIP checks |

Coordinates and offsets are in meters. The skin has 15,299 vertices and 29,899
triangles, with finite offsets for every expression at every vertex. The full
228,660-vertex volume has offsets defined at 195,460 vertices. Its remaining
33,200 vertices retain the source NaNs; the transfer does not invent offsets.
The scalar uint8 `BlendshapeDefined` mask applies to the full volume.

The original offsets are preserved exactly. Recovering an offset by subtracting
the new neutral from its target introduces at most `2.220446e-16 m` of floating
point cancellation, below the coordinate-scaled `2.051069e-15 m` check. Stored
target coordinates equal the direct new-neutral-plus-offset expression exactly.

## Validation and scope

The independent audit verified every source and artifact hash, all 36 ordered
offset fields, point IDs, local triangle indices, neutral/target coordinates,
volume connectivity, the defined-offset mask, and every ZIP member's bytes.
Exported skin and volume expression fields agree with the source, including
the source's undefined full-volume entries.

This is a geometric transfer. The new neutral's force solve converged and its
contact checks passed, but it still has two inverted tetrahedra with minimum
det(F) -0.220242. The exports preserve that diagnostic status. The expression
targets are not independently solved equilibria or contact-validated states,
and the deformed volume is not a stress-free FEM reference.

## Reproduction

Run from `exp/2026/09/23/new-neutral`, choosing fresh output directories when
repeating the preparation and review:

```bash
CHERRIES_NAME='Transfer all source blendshapes to new neutral' \
CHERRIES_TAGS='neutral,blendshapes,transfer,geometry,active-strain' \
.venv/bin/python -u src/90-transfer-blendshapes.py

CHERRIES_NAME='Audit blendshapes transferred to new neutral' \
CHERRIES_TAGS='neutral,blendshapes,transfer,audit' \
.venv/bin/python -u src/92-audit-blendshapes.py

CHERRIES_NAME='Review blendshapes transferred to new neutral' \
CHERRIES_TAGS='neutral,blendshapes,transfer,review' \
.venv/bin/python -u src/91-review-blendshapes.py
```

The [transfer run](https://www.comet.com/liblaf/apple/9e8e20cde4194760a5c762cbaea84ae5)
records 36 expressions, 15,299 skin vertices, reconstruction error
`2.220446e-16 m`, and neutral validity false. Completed transfer, audit and
render logs are preserved in the bundle. Ruff and compilation checks passed.

The first preparation preflight used an overly small absolute subtraction
tolerance; the final run uses a coordinate-scaled floating-point bound. The
first audit additionally required an unnecessary skin mask; the corrected
audit checks finite skin fields and the declared full-volume mask. Neither
correction changes expression offsets or target geometry.

## Tailnet review

Open the transferred blendshapes (private preview omitted) for
the 36-expression gallery, selectable detail images, original/new neutral
comparison and downloads. The main neutral page links this review.
Expression images label the known **base neutral** issue; they do not claim
an inversion count for each expression.

The [independent audit run](https://www.comet.com/liblaf/apple/9aed8b4a2dd446ca94196237c067f506)
and [final render run](https://www.comet.com/liblaf/apple/76dffddeca674d529f37f77512224b38)
completed. All 48 added/updated HTTP files match their local SHA256 hashes,
including every expression image and download; all local HTML links resolve.
[HTTP verification](../data/http-verification-blendshapes-005.json) and
[hosting receipt](../data/serve-blendshapes-005.json) record the served result.
The existing transient tailnet service was retained.
