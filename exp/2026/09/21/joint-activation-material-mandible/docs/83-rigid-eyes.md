# Rigid source eyes

`83-prepare-rigid-eyes.py` froze the registered melon source
`20-eye.ply` as `data/rigid-eyes-001`.

The collider contains 1,298 vertices, 2,560 triangles, and two connected
components. `eyes.npz` retains the original coordinate, vertex, and triangle
ordering exactly. Component labels are metadata only. The source contains 32
open directed-edge incidences after exact index inspection; this is why signed
containment must use an exact-weld proxy rather than treating the raw collider
as a watertight solid. IPC always uses the untouched source triangle mesh.

Comet: <https://www.comet.com/liblaf/apple/7e2da19871be4a398ab27c2094f71b84>
