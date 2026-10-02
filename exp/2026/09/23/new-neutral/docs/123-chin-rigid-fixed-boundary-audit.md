# Chin rigid-pose fixed-boundary audit

The accepted 6-DoF chin estimate kinematically inverts 168 of 2,320 FEM
tetrahedra whose four vertices are already fixed. Their smallest signed
Jacobian ratio is -79.28335. The remaining all-fixed inversions are 65 cells
with three cranium and one mandible vertex, 54 with two of each, and 49 with
one cranium and three mandible vertices.

This reproduces the fixed-map limitation identified in the earlier pose-jump
audit at the full accepted chin pose. `FullSkullPhysics.full_boundary_displacement`
sets original mandible FEM vertices to their rigid SE(3) displacement and
leaves all other original fixed FEM vertices at zero. Appended complete-bone
nodes are fixed collision nodes and are not FEM tetrahedron vertices.

The reference-repair displacement at all 29,601 fixed FEM nodes is exactly
zero. The inversions therefore result from the prescribed mixed cranium and
mandible boundary motion, rather than altered fixed reference coordinates.
This is a kinematic diagnostic. It does not solve equilibrium, change the
fixed map, or establish forward validity.

Reproduce on CPU from the experiment directory:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Audit chin rigid fixed-boundary tetrahedra' \
CHERRIES_TAGS='mouthopen,rigid-pose,fixed-boundary,cpu' \
uv run python src/123-audit-chin-rigid-fixed-boundary.py
```

Receipt: `data/chin-rigid-pose-001/fixed-boundary-audit.json`. Cherries:
<https://www.comet.com/liblaf/apple/b5ec9bd54af1405497b895dee63fc3c7>.
