# MouthOpen rendered from the full tetrahedral mesh

The four completed MouthOpen endpoints are rendered using the complete boundary extracted from the simulation's pruned tetrahedral mesh. Both the opaque shape row and the translucent activation context use this boundary, including cavity surfaces. The camera fits every deformed volume vertex across all four stages and the prescribed anatomy.

![Full tetrahedral mesh comparison](../data/90-mouthopen-full-tetmesh-002/mouthopen-full-tetmesh-preview.png)

[Full-resolution figure](../data/90-mouthopen-full-tetmesh-002/mouthopen-full-tetmesh.png) · [render manifest](../data/90-mouthopen-full-tetmesh-002/manifest.json) · [boundary topology and parent-tet IDs](../data/90-mouthopen-full-tetmesh-002/full-boundary-topology.npz) · [numerical results](87-mouthopen-four-stage-results.md)

The source mesh contains **227,900 vertices and 1,144,268 tetrahedra**. Its complete boundary has **63,282 vertices and 126,648 triangles**, including **47,983 boundary vertices outside the fitted face subset**. Rendering uses the saved displacement at each original boundary vertex ID. The mesh is the actual pruned fixture used by the solver; the 2,249 fully fixed tetrahedra removed before that solve are absent from this fixture.

Activation glyphs remain at active tetrahedron centers, with the existing visibility policy and common color and line-length scales. Fit and normal RMS labels measure the original face target with reference-area weights; a full-volume target was not fitted. The same four saved endpoint hashes, metrics, and gradient ratios were verified against the preceding comparison. This rendering performs no new numerical solve.

The final preview was inspected visually. All eight panels share the same camera projection. Output sizes are 10,240 × 5,760 pixels for the full figure and 1,920 × 1,080 for the preview. Input, source, image, panel, and boundary-topology hashes passed verification. The rendering retains the inverted cells and boundary intersections documented in the numerical report; it does not establish mechanical validity.

An independent topology audit matched the saved boundary against a fresh extraction and verified that every one of its 126,648 triangles belongs to its recorded parent tetrahedron. Eligible activation glyph counts match the previous render. Visible glyph counts differ slightly because the camera fitted to the full mesh changes the rasterized occlusion mask.

Run from `exp/2026/09/29/mouthopen-activation`:

```bash
CHERRIES_NAME='MouthOpen full tetrahedral mesh comparison' \
CHERRIES_TAGS='mouthopen,four-stage,render,full-tetmesh' \
.venv/bin/python -u \
  src/90-render-full-tetmesh.py --output 90-mouthopen-full-tetmesh-002
```

The process and Cherries shutdown completed with exit code 0. [Terminal log](../logs/90-full-tetmesh-002-terminal.log) records the run. [Comet experiment](https://www.comet.com/liblaf/apple/97970360c1ff4249b7a4fc056a63d9ab): start `2026-09-29 14:53:45.056103+08:00`, end `2026-09-29 14:54:03.030128+08:00`, recorded Git revision `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. The run used the working tree and the experiment profile with automatic commits disabled. The output includes its rendering source snapshot. Ruff and Python compilation checks passed.
