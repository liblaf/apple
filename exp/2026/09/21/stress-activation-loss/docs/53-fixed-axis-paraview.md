# L2 + normal fixed-axis tetrahedral mesh

[Deformed tetmesh](../data/53-paraview-l2-normal-fixed-001/l2-normal-fixed-axis.vtu): the completed fixed-axis stage at update 200 from capture `51-visualization-checkpoints-002`. It contains 228,660 points and 1,146,517 tetrahedra, with the original connectivity and exact saved `X+u` coordinates. An independent audit verified geometry and tensor fields after reloading.

`ActivationInverseMatrix` stores saved B on active cells and identity on inactive cells. `RestPosition`, `Displacement`, `DetF`, `IsInverted`, and `PrincipalActivationAmplitude` are available for inspection. Historical activation attributes are prefixed `Fixture` to distinguish them from the fitted state. The 91 inverted cells are retained.

The ParaView startup script loads this VTU plus registered `13-cranium.ply` and `13-mandible.ply`, displaying the tetmesh as Surface With Edges and bones as Surface. It saves a reusable `fixed-axis-tetmesh-bones.pvsm` and a screenshot after rendering succeeds.

Export command, run from this experiment group:

```bash
CHERRIES_NAME='ParaView export, L2 normal fixed-axis tetmesh' CHERRIES_TAGS='active-strain,paraview,fixed-axis,tetmesh' uv run python src/53-export-fixed-paraview.py
QT_QPA_PLATFORM=xcb paraview --disable-registry --script src/54-open-fixed-paraview.py
```

The export requires a fresh output directory. [Export receipt](../data/53-paraview-l2-normal-fixed-001/export.json), [Cherries log](../logs/53-export-fixed-paraview.log), and [Comet run](https://www.comet.com/liblaf/apple/6509e6779e36456080be51bf53410c60). Cherries shutdown completed normally; no new numerical solve or commit was performed. Base Git revision: `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.
