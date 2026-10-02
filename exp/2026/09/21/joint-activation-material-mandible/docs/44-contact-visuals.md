# IPC contact visual audit

## Scope

`src/44-render-contact.py` reconstructs the exact native IPC state from
`data/contact/config.json` and the frozen contact surface map. It renders free
soft-tissue contact against both pure cranium and pure mandible surfaces, with
the mixed-label triangles omitted by the declared bonded-transition policy
overlaid separately as magenta wireframe. The registered-template lip defect is
intentionally absent because it is not part of the exact FEM contact mesh.

The final reference render ran locally through Cherries with:

```bash
DEBUG=1 \
CHERRIES_NAME="IPC contact visual QA final" \
CHERRIES_TAGS="joint-inverse,contact,visual-qa,cpu" \
uv run --frozen python src/44-render-contact.py
```

It wrote ten figures, `summary.json`, and `report.md` under
`data/contact-visuals/`. The renderer also accepts a later neutral state through
`--checkpoint PATH`; it rebuilds the broad phase and contact gradient at that
deformed state.

The bone colliders are the pure labeled boundary faces of the cropped FEM
fixture. They are not the complete independently registered source-bone meshes.
All 21,891 vertices on the 41,903 pure-cranium triangles belong to the recovered
22,091-node cranium Dirichlet support; all 7,395 vertices on the 13,763
pure-mandible triangles belong to the 7,510-node differentiable rigid-jaw
support. No participating collider vertex lies outside its runtime support.
Historical `IsFixed` alone would miss 1,070 of those cranium vertices and 1,267
mandible vertices; the runtime correctly uses the recovered group supports.

The cropped face fixture retains all Melon mandible support nodes but only
22,091 of 37,846 full-head cranium support nodes. Median FEM-collider-vertex to
source-surface distance is at machine precision for both bones, and the 95th
percentiles are 3.36 and 4.70 micrometres. This verifies that the source and FEM
surfaces share a world frame. Reverse source-vertex proximity does not measure
contact coverage: 43.60% of source cranium vertices and 70.30% of source
mandible vertices happen to lie within 0.1 mm of the pure FEM collider, while
large posterior source regions extend beyond the cropped simulation domain.

## Relevance against the simulated soft-tissue domain

| Cranium flags | Mandible flags |
| --- | --- |
| ![Cranium collider relevance](../data/contact-visuals/05-cranium-collider-relevance-front.png) | ![Mandible collider relevance](../data/contact-visuals/05-mandible-collider-relevance-side.png) |

The relevance test starts from the 34,245 vertices on the actual pure-soft FEM
boundary. A primary flag lies within 1 mm of the independently registered source
bone, more than 2 mm from the declared pure FEM bone collider, and more than
1 mm from the bonded mixed-face surface. It finds 411 cranium-near vertices,
covering 0.4573% of pure-soft vertex-area weight, and 182 mandible-near vertices,
covering 0.2490%. Their bounds place them beside simulated midface/oral soft
tissue, so they are local collider-relevance flags rather than posterior crop
alone. Marker color is the 2.00--8.44 mm cranium or 2.01--11.01 mm mandible gap
to the declared collider.

The source-only reverse sample separates the large crop effect: of 9,072 source
cranium vertices more than 2 mm from the FEM collider, 8,449 are also more than
2 mm from the pure-soft domain; for mandible the corresponding counts are 2,326
and 1,775. These samples lie outside both collider and simulated soft boundary
and are not direct evidence of a local contact hole. The issue maps therefore
show only source-near regions beside simulated soft tissue. They do not prove
anatomical contact, penetration, or a required contact law.

The frozen collider remains a well-defined numerical experiment model. These
local flags require anatomical relevance review before its pure-face coverage
can support claims about complete soft-tissue/bone contact.

The exact FEM-face audit independently recovers every pure face and finds zero
missing or extra soft-to-bone broad-phase candidates at reference and current
neutral. It also shows that the “bonded mixed” name is a policy: 413 of the
6,926 mixed faces contain only cranium and mandible labels, two contain both
bones plus soft tissue, and the nearest-source label transfer contains no
free-versus-bonded attachment field. See
[`docs/18-fem-contact-surface-audit.md`](18-fem-contact-surface-audit.md).

A direct triangle audit shows why the complete registered source bones cannot
simply replace or augment that collider. In the reference state, source
cranium versus pure soft tissue has 1,857 raw intersection pairs: 594 are
confined to bonded mixed-face geometry, 1,263 are not confined, and 149 of the
latter lie more than 1 mm from bonded geometry. Source mandible has 766 raw
pairs: 274 confined, 492 unconfined, and 14 more than 1 mm away. At the current
contact-enabled neutral checkpoint the corresponding counts are
1,336/145/1,191/114 for cranium and 773/79/694/49 for mandible. The complete
source surfaces therefore do not provide a collision-free initial contact
configuration. A reviewed binding map and repair, collision-free offset, or
remeshing of remaining free patches must precede their use as colliders. The
classification, signed clearances, and maps are in
[`docs/17-source-bone-contact-audit.md`](17-source-bone-contact-audit.md).

## Active locations

| Front cutaway | Side cutaway |
| --- | --- |
| ![Front contact locations](../data/contact-visuals/01-active-contact-locations-front.png) | ![Side contact locations](../data/contact-visuals/01-active-contact-locations-side.png) |

Blue spheres are soft-tissue/cranium collision stencils and orange spheres are
soft-tissue/mandible stencils. Magenta wireframe is the mixed-label bonded
attachment topology excluded from IPC. The frozen native-filter preflight
recorded 169 active representatives and a minimum active gap of 0.0225836 mm
inside the declared `dhat=0.1 mm` barrier zone. The rendered reconstruction had
171 representatives: 12 cranium and 159 mandible.

The count difference is an `IMPROVED_MAX_APPROX` representation detail. Repeated
identical LBVH rebuilds select 169--171 representatives at degenerate locations,
while barrier energy, minimum gap, nodal force gradient, and geometric locations
remain invariant to numerical precision. Active-pair count must therefore not
be used as a convergence or physical contact-area metric.

![Gap histogram](../data/contact-visuals/03-active-gap-histogram.png)

In the rendered state the cranium-contact gap range is 0.06814--0.08547 mm with
a 0.07675 mm median. The mandible-contact range is 0.02258--0.09982 mm with a
0.07570 mm median. The histogram counts collision-set representatives rather
than independent anatomical patches.

## Bone-side contact forces

| Front cutaway | Side cutaway |
| --- | --- |
| ![Front bone force](../data/contact-visuals/02-bone-contact-force-front.png) | ![Side bone force](../data/contact-visuals/02-bone-contact-force-side.png) |

The plotted force is the negative IPC energy gradient. The model gradient has
units MPa m²; multiplying by `1e6` yields newtons. Color shows nodal magnitude
on a shared logarithmic scale and arrows share a 4 m/N display scale. The
reference gradient acts on five cranium nodes and fourteen mandible nodes.

![Force summaries](../data/contact-visuals/04-bone-force-summary.png)

| Bone | Resultant | Sum of nodal magnitudes | RMS over active bone nodes | Maximum node |
| --- | ---: | ---: | ---: | ---: |
| Cranium | 0.00226336 N | 0.00230763 N | 0.000600171 N | 0.00105126 N |
| Mandible | 0.000504444 N | 0.000964516 N | 0.0000859700 N | 0.000140193 N |

The resultant is the norm of the vector sum on the selected bone; the sum of
magnitudes is non-cancelling; RMS uses only nonzero bone nodes. The all-node
action-reaction residual is `2.38e-19 N`. These locations and nodal forces are
not physical pressure. A pressure plot would require a validated nodal or
contact-patch area division.

## Interpretation

The native preflight is numerically valid: it has no initial intersections,
positive active gap, finite nonnegative barrier energy, and unit CCD fractions.
The reference barrier forces are small because this is the undeformed state and
the `0.01 MPa` barrier stiffness is a declared numerical scale rather than a
measured interface property.

This evidence validates the ownership and units of the contact term. It does
not validate anatomy, establish neutral equilibrium convergence, or support
activation/material/jaw trends. The same figures must be regenerated from the
contact-enabled converged neutral checkpoint and final joint state before those
states are admitted.

`data/contact-visuals/summary.json` provides the dashboard-ready asset list,
captions, hashes, gap quantiles, collision types, per-bone force definitions,
and the explicit limitation on pressure interpretation.
