# Full tetrahedral mesh render of the corrected neutral

The corrected `forward-isfixed-001` endpoint is rendered using the complete
tetrahedral mesh boundary. The main `review-isfixed-001` front and side comparison
images now use this boundary. The additional `full-tetmesh/` page includes those
views, a boundary-edge view, and a whole-tetrahedron cutaway. Bones and eyeballs
retain their saved neutral poses. All displacements are displayed at true scale.

## Command and run

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='Render corrected neutral complete tetrahedral mesh' \
CHERRIES_TAGS='neutral,isfixed,tetmesh,render,anatomy' OMP_NUM_THREADS=4 \
.venv/bin/python -u \
  src/139-render-full-tetmesh.py > tmp/render-full-tetmesh.log 2>&1
```

The process exited with code 0 and completed normal Cherries shutdown. Comet run:
<https://www.comet.com/liblaf/apple/8fc64986b95c4a0282dd02f6067047ad>

Recorded summary: 228,660 vertices; 1,146,517 tetrahedra; 128,172 full boundary
triangles. The source checkout was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`
with existing local modifications; no commit was made. The exact renderer is
archived alongside the new images and its hash is recorded in `receipt.json`.

## Validation and outputs

- Input reference and endpoint hashes were verified against the saved review.
- All 1,146,517 input cells are tetrahedra and enter boundary extraction.
- The complete boundary has 64,042 vertices and 128,172 triangles; the previous
  skin surface has 15,299 vertices and 29,899 triangles.
- There is no anatomical filtering or decimation in the full boundary views.
- Rendered solved coordinates exactly equal reference coordinates plus saved
  endpoint displacement, and exactly match the published solved volume.
- `FixedMask` agrees with `IsFixed` in all three coordinate components.
- The half-mesh cutaway selects 597,625 whole cells by reference centroid, using
  identical cell IDs for the reference and solved views. Anatomy is clipped at
  the same midplane for this view only.
- Saved cranium, mandible, and eye coordinates/connectivity were independently
  checked against their registered source geometry.
- Ruff passed. All four PNGs were inspected visually.
- Main page, new subpage, four images, and receipt returned HTTP 200 over the
  existing tailnet server and matched the local files byte for byte.

Assets are in `data/review-isfixed-001/full-tetmesh/`: `index.html`, four PNGs,
two full-boundary VTP files, renderer source, and `receipt.json`. Full VTU volume
downloads remain linked from the page. The existing runtime-only tailnet service
continues serving the review; no persistent service was added.

Preview: PRIVATE_PREVIEW_URL

## Scope

This is saved-state visualization; physics was not rerun. An opaque rendering
shows the exposed boundary, while the cutaway reveals interior cell faces.
The complete boundary includes all exposed anatomical group boundaries. The
volume download contains every tetrahedron. Numerical validity remains tied to
the existing corrected forward solve and its independent audit.
