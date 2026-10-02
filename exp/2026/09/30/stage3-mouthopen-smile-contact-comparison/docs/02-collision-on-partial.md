# Stage 3 MouthOpen-to-Smile transition render

The renderer produced accepted-state diagnostic snapshots from `exp/2026/09/30/stage3-mouthopen-smile-contact-comparison/data/20-collision-on` using the prepared Stage 3 fixed-axis contraction endpoint tensors. Both branches use the 0.01 N force gate. The collision-on branch also passes its declared soft-tissue-to-bone/eye contact gates. The shared camera includes all deformed states from both runs. The collision scope excludes soft-soft and rigid-rigid pairs, and neither render establishes mechanical validity. The historical Smile Stage 3 solve was invalid; its saved S tensor is used only as a fixed control and every displayed displacement is freshly re-equilibrated.

- Status: `running_transition`
- Force tolerance: `1e-08 MPa m²`
- Activation: Stage 3 fixed-axis contraction
- Contact enabled: `True`
- Selected states: 1
- Full tetmesh boundary: 63,282 vertices, 126,648 triangles
- Selected state: β=0.000, force=0.00624886 N, minimum active gap=77.4826 μm, active contacts=571, inverted tetrahedra=363
- Manifest: [`output/weekly-2026-09-30/stage3-render-work/02-collision-on-partial/manifest.json`](../../../../../../output/weekly-2026-09-30/stage3-render-work/02-collision-on-partial/manifest.json)
