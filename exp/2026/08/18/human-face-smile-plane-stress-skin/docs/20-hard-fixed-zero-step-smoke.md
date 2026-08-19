# Corrected hard-fixed zero-step smoke

## Status

The corrected material preparation and isolated zero-step integration smoke are
complete and passed. After this smoke received its own review, the separately
approved formal 40-update inverse and analyzer `30` also completed. Their
scientific result is documented in the
[formal baseline report](30-corrected-baseline-screen.md); no second inverse is
automatically approved.

This smoke verifies that the selected model can be constructed, solved, read
back, and differentiated with the intended constitutive law, membrane domain,
and conservative cut boundary. It does not measure inverse target fidelity or
Bumpy after optimization.

## Reviewed setup

- facial membrane: all three triangle vertices satisfy `IsFace=true`;
- membrane size: 15,299 points, 29,899 triangles, one component, area
  `0.04287998059707302 m2`;
- skin: homogeneous `E=0.2 MPa`, `nu=0.49`, no prestrain;
- skin Lamé conversion: plane stress,
  `lambda=0.1289643374128175 MPa` and
  `mu=0.06711409395973154 MPa`;
- Koiter measure: fixed original reference area;
- volume aponeurosis, fat, and muscle: unchanged 3D Lamé conversion;
- artificial cut: every vertex incident to a triangle touching `GroupId=-1`
  is fixed to exact zero displacement;
- cut counts: 13,165 triangles and 6,980 incident vertices, comprising 380
  pre-existing fixed vertices and 6,600 newly fixed vertices;
- full model boundary condition: 33,636 fixed vertices and 100,908 fixed
  degrees of freedom;
- inverse state: fresh exact-zero displacement and activation, no transferred
  state, zero optimizer updates, one forward/adjoint evaluation.

The corrected [manifest](../data/10-corrected-baseline-manifest.json) has
SHA-256
`d999be4fc941253b8daa84dca4a52ab44bd02b3e42ce2af6d151a2e14b64a21a`.
The filtered [skin VTP](../data/10-corrected-baseline/skin-isface-e0200-p000.vtp)
has SHA-256
`4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`.
All topology, material, solver-content, formula, and file readback gates passed.

## Command

The authoritative smoke used the isolated `v3` root:

```bash
SMOKE_ROOT=tmp/hard-fixed-smoke-v3
SMOKE_STEM=20-corrected-baseline-hard-fixed-smoke
SMOKE_NAME='Smile corrected hard-fixed zero-step smoke v3 canonical archive'
SMOKE_TAGS='human-face,smile,skin,plane-stress,isface,cut-fixed,'\
'inverse,smoke,canonical-archive,debug'
DEBUG=1 MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true \
CHERRIES_NAME="$SMOKE_NAME" CHERRIES_TAGS="$SMOKE_TAGS" \
uv run --frozen python src/20-inverse-plane-stress-screen.py \
  --stage smoke \
  --inverse-max-steps 0 \
  --mandatory-baseline-steps 0 \
  --output-summary "$SMOKE_ROOT/$SMOKE_STEM-summary.json" \
  --output-table "$SMOKE_ROOT/$SMOKE_STEM-table.md" \
  --live-plot-dir "$SMOKE_ROOT/figs/live-corrected-baseline-hard-fixed-smoke"
```

The run exited successfully in 20.6 seconds. `DEBUG=1` prevented Git or Comet
mutation in the existing dirty worktree.

## Results

| check | result |
| --- | ---: |
| aggregate / case status | `complete=true` / `ok` |
| evaluations / optimizer updates | 1 / 0 |
| target error RMS | 5.310139 mm |
| target error / target RMS | 1.0 |
| forward | `primary_success`, 1 step |
| adjoint | `success` |
| adjoint relative residual | `4.890982876e-4` |
| activation gradient norm | `0.1474337323` |
| initial and final activation max | 0 |
| cut displacement max | `0 m` exactly |
| fixed value max | `0 m` exactly |
| inverted tetrahedra | 0 |
| folded skin triangles | 0 |

The zero target displacement ratio is expected: with p000 and zero activation,
the stress-free corrected model remains at the undeformed state. The nonzero,
finite activation gradient and successful adjoint are the important integration
checks for a future inverse run.

The result and one-frame VTKHDF read back as 228,660 points and 1,146,517
tetrahedra. Their displacement, hard-fixed marker fields, fixed mask/value, and
recovered activation arrays agree exactly. All 6,980 cut vertices have
bit-exact zero displacement in both files. JSON and JSONL are strict-finite;
the individual final summary equals the aggregate case exactly; recorded
target, result, history, and trace sizes and SHA-256 identities all match.

The authoritative [aggregate](../tmp/hard-fixed-smoke-v3/20-corrected-baseline-hard-fixed-smoke-summary.json)
has SHA-256
`4ea9cfdcd66b7bd1475a5d231df949966119d85d44b0b13fba08eaae62d8fdae`.
The result, history, target, and trace hashes are respectively:

- `6f8c28fc4d11a3fe5004fd2ea9b3b6259b7a874d069f340dce9a28d214c2912f`;
- `76b48ff0dd6685bb653e68261b9827a712dcaddfee5b31e4fc4c9ad7fb0cad0d`;
- `d5b185a486734093d83ffa7d28aeef944e11bb2c5aa3ec0fbc129caa5bc8e66f`;
- `eabc5007434e1abfe2e683f3ce0cd0e52ede494c85ab3d97a0983865c49cc95f`.

## Cherries archive check

The first smoke exposed that the inherited runner logged summaries before the
outer corrected-schema rewrite. A same-name snapshot refresh then exposed a
separate Local plugin bug when its optional log file was absent. Neither issue
changed numerical outputs, but both would weaken formal-run provenance.

The final producer therefore:

1. logs the actual corrected manifest explicitly;
2. writes unique post-rewrite `*-summary-final.json` copies whose bytes must
   equal the live canonical summaries;
3. logs those unique files instead of relying on the broken same-name overwrite
   path.

The `v3` snapshot contains the manifest at its exact SHA and both final summary
copies byte-for-byte equal to the live canonical files. The earlier `v1` and
`v2` smoke outputs were left in place and were not deleted or overwritten.
Warnings about absent inherited default/formal outputs remain bookkeeping-only;
the run created no formal screen data or figures.

The executed producer SHA-256 is
`d0c3db48ca1532a2eb3682f5f35eda0450c95b5925edb6c36b17aa6fa1e43020`.
The blocked analyzer is pinned to that producer and currently has SHA-256
`0718c7eba6a8d6881b72ef3ac97d25c588437107f6ce82d2f5c12c68f87bfb6d`.

## Decision boundary

This smoke was a GO for the corrected implementation path, not an automatic
inverse authorization. A later separate decision approved exactly one
fresh-zero corrected baseline:

```text
IsFace-only + plane stress + fixed reference area + hard-fixed cut
E=0.2 MPa, p000, Adam LR=0.3, 40 updates / 41 evaluations
```

That trajectory and its standard views are now reviewed in the
[formal baseline report](30-corrected-baseline-screen.md). They do not support
simply extending `p000`, and no second material case has been approved. Old
full-boundary + 3D-Lamé results remain immutable and are retained for the group
meeting under the label
`Historical full-boundary + 3D Lamé (superseded; mechanism-only)`; see the
[forward-probe report](15-forward-domain-conversion-probe.md).
