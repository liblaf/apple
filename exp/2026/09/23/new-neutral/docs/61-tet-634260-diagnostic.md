# Continuation tetrahedron 634260

Tetrahedron 634260 is inverted at every inspected continuation checkpoint and
has the minimum detF ratio through Newton step 500. Its detF ratio changed from
`+0.06109` at the immutable parent endpoint to `-0.03041` at step 100, then
`-0.09669`, `-0.16722`, `-0.27212`, and `-0.33909` at steps 200, 300, 400,
and 500. Checkpoint 300 also has a second inverted tetrahedron, 695230; the
other inspected checkpoints have only 634260 inverted.

The element has vertices 118401, 117646, 116427, and 119656. All are free,
interior nodes; none is a skin vertex or fixed boundary node. It is pure fat
with zero active fraction and lies 13.97 mm from the nearest skin point. The
available evidence does not connect this inversion to a surface contact or a
direct active-strain load.

The clearance repair had already reduced this element's signed reference volume
to 64.67% of its original volume. The parent endpoint compressed it further to
6.11% while still positive. That made it the immediate local quality weakness
before continuation, though it was not the parent's global minimum-detF cell.

The bulk stable Neo-Hookean implementation evaluates a quadratic function of
the physical determinant `J = det(F)` and does not restrict its domain to
positive `J`. The IPC barrier applies to selected surface contact pairs, not
tetrahedron orientation. Newton acceptance therefore has no determinant gate,
so force convergence and noninterpenetrating surface contact cannot establish a
physically valid volume configuration after this crossing. The numerical values
and classification are stored in
`data/forward-repaired-reference-003/tet-634260-diagnostic.json`.
