# Fixed-activation MouthOpen-to-Smile contact render

The renderer produced accepted-state diagnostic snapshots from `exp/2026/09/30/mouthopen-smile-collisions/data/75-fixed-activation-contact`. The endpoint activation tensors are prescribed from the saved MouthOpen and Smile states; the run does not optimize activation. Each displayed state must pass the saved force and declared soft-tissue-to-bone/eye contact gates. The contact scope excludes soft-soft and rigid-rigid pairs, and this render does not establish mechanical validity.

- Status: `blocked`
- Force tolerance: `1e-08 MPa m²`
- Selected states: 2
- Full tetmesh boundary: 63,282 vertices, 126,648 triangles
- Selected state: β=0.120, force=0.00896873 N, minimum active gap=72.5139 μm, active contacts=590, inverted tetrahedra=383
- Manifest: [`exp/2026/09/30/mouthopen-smile-collisions/data/76-fixed-contact-terminal-preview/manifest.json`](../data/76-fixed-contact-terminal-preview/manifest.json)
