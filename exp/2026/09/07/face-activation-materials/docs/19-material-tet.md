# Exact material-tet visualization

## Purpose

This artifact makes the compression of fixture cell 662949 directly visible. It
uses only the cell's exact four saved vertices and six straight edges. The four
panels show rest geometry, the saved FiberRegion-B-v2 endpoint, and the stable
and logarithmic Neo-Hookean fat `nu=0.49` fixed-control replays.

## Command

From `exp/2026/09/07/face-activation-materials`:

```bash
CUDA_VISIBLE_DEVICES='' DEBUG=1 \
  CHERRIES_NAME='Cell 662949 exact material tet final local' \
  CHERRIES_TAGS='cpu,material-tet,exact-geometry,final-local' \
  uv run python src/44-render-material-tet.py
```

The final local run completed in 2.9 seconds and reported zero solver runs.

## Results

| State | J |
| --- | ---: |
| Rest | 1.000000000000 |
| Saved baseline, 21-v2 | 0.200011317572 |
| Stable fat, `nu=0.49` replay | 0.539704697364 |
| Neo fat, `nu=0.49` replay | 0.637215974831 |

The fixed vertex is global point 111321. Every panel uses one orthographic
camera, one set of bounds, and one orthonormal frame derived only from the rest
tetrahedron. Displacements are shown at scale 1. No smoothing, geometric
amplification, synthetic displacement, or neighboring-cell context is used.

## Outputs

`data/44-material-tet/` contains:

- `cell-662949-four-states.png` and `.pdf`;
- `cell-662949-geometry.json`, including mesh/global/upstream point IDs, global
  positions in metres, local positions in millimetres, affine transforms,
  exact topology, J values, and input hashes;
- `cell-662949-browser.json`, a 1.8 KB browser payload with exactly four
  vertices and four triangles for each state;
- a source snapshot and SHA-256 artifact manifest.

The PNG was inspected at its original resolution. The PDF is a single page, all
JSON files parse, and the browser payload has four vertices and four triangles
for every state.
