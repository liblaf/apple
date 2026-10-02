# Fixed-boundary pose audit

`pose-jump-002` is a numerical pose-jump diagnostic, not a physically valid
MouthOpen state. Its final contact-on solve has 282 inverted tetrahedra. Of
those, 71 have all four original FEM vertices in the model fixed map, so a
free-vertex volume projection cannot change their orientation.

Every one of the 71 cells mixes the two distinct prescribed boundary motions:
35 have three cranium and one mandible vertex, 20 have two of each, and 16 have
one cranium and three mandible vertices. Cranium and mandible fixed-node sets
are disjoint. The historical fixed set overlaps them, as expected for the
legacy mask, but does not create an ambiguous cranium-versus-mandible
assignment. The cells form 15 disconnected vertex-sharing components; the
largest contains 15 tetrahedra.

The repaired-reference signed-volume ratio for each of these cells is exactly
one within the saved calculation. The reference repair therefore did not
weaken them. Their inversion is produced by imposing a rigid mandible motion
beside fixed cranium vertices. The q=0 source at 4.1294 degrees already has
17 inversions, including 8 all-fixed cells. After the 11.5588-degree pose
jump, the harmonic collision-off relaxation has 299 inversions and the final
contact-on solve has 282, including the same 71 all-fixed cells.

Reproduce the receipt on CPU from the repository root:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pyvista as pv
import torch

data = Path("exp/2026/09/23/new-neutral/data")
reference = np.load(data / "reference-clearance-002/reference-clearance.npz")["repaired_points_m"]
mesh = pv.read(data / "reference-clearance-002/repaired-reference-volume.vtu")
tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
with np.load("exp/2026/09/21/joint-activation-material-mandible/data/frozen-neutral-004/state.npz") as source:
    fixed = np.unique(np.concatenate([source["historical_fixed_node_ids"], source["cranium_node_ids"], source["mandible_node_ids"]]))

def signed(points):
    edges = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(edges)

rest = signed(reference)
for name in ("source", "relaxed", "proposal-final"):
    state = torch.load(data / "pose-jump-002" / f"{name}.pt", map_location="cpu", weights_only=False)
    ratio = signed(reference + state["displacement_m"].numpy()[:len(reference)]) / rest
    inverted = np.flatnonzero(ratio <= 0)
    all_fixed = inverted[np.isin(tets[inverted], fixed).sum(axis=1) == 4]
    print(name, len(inverted), len(all_fixed), float(ratio.min()))
PY
```

The complete cell IDs, component bounding boxes, source hashes, fixed-map
counts, and stage comparisons are in
`data/pose-jump-002/fixed-tet-audit.json`. Under the current reference and
Dirichlet map, a valid forward requires a boundary/topology or pose change;
postprocessing free nodes cannot resolve these all-fixed inversions.
