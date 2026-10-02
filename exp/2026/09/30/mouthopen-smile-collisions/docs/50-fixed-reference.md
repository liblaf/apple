# Repaired reference transfer for pruned activation fixture

The full historical and repaired meshes have identical tetrahedron connectivity. The pruned fixture maps to them by its saved original point and cell IDs. Every prescribed point has identical coordinates in the historical, repaired-source, and final repaired references.

The final reference has 227,900 points, 1,144,268 positive-volume tetrahedra, and 288,172 active cells. Minimum rest volume: 9.90084865e-16 m³. The Smile and MouthOpen activation arrays and the point/cell maps are byte-identical to their sources.

The repaired source reference differs from the historical free-point reference. The output therefore records both coordinate differences in `summary.json`. The derived mesh updates bulk rest volumes, active dual-volume weights, graph weights, and skin rest areas. Skin IDs and observation weights are retained exactly. These are fixture preparation artifacts, not a new equilibrium solve.
