# Eye-clear initial displacement

`84-repair-eye-initialization.py` produced
`data/eye-initialization-003/seed.npz`. It starts from the adopted Run011
neutral, moves only free soft-tissue nodes, and preserves the constitutive FEM
reference, skull, mandible, and exact eye collider.

The original neutral had 687 soft-eye intersection pairs over 244 soft faces,
with a minimum nodal signed distance of -0.501638 mm and no fixed-node
conflict. A closest-surface projection plus radial source-eye support-plane
constraints cleared every source-eye triangle pair. A graph-harmonic extension
then propagated that surface correction to the volume.

The resulting state has zero source-eye pairs, 100.000 um minimum nodal
clearance, unchanged fixed nodes, and is finite. The independent exact triangle
audit also reports zero soft-cranium, soft-mandible, and soft-eye pairs. The
repair moved at most 2.454 mm and creates seven inverted tetrahedra; this is
reported for the forward solve rather than used as an exclusion gate, following
the user-approved tolerance for a few inverted cells.

Comet: <https://www.comet.com/liblaf/apple/330cb05570cd47439438e23c2790dddf>
