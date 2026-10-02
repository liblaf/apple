# Frozen FEM contact-surface audit

`src/18-audit-fem-contact-surface.py` tests the selected pure-face IPC map
against the complete labeled FEM boundary. It does not change the contact
surface, contact parameters, support map, or input manifest.

## Pure-face coverage and kinematics

The 128,172-face boundary partitions into 41,903 pure cranium faces, 13,763
pure mandible faces, 65,580 pure soft faces, and 6,926 mixed faces. The runtime
selects all 121,246 pure faces; there are no omitted pure cranium, pure
mandible, or pure soft faces. All 21,891 pure-cranium vertices belong to fixed
cranium support, all 7,395 pure-mandible vertices belong to differentiable
rigid-jaw support, and none of the 34,245 pure-soft vertices belongs to either
bone support. The fixed and moving bone supports are disjoint.

An independent unfiltered broad phase was post-classified by the same binary
soft/bone ownership. At reference it contains exactly 2,162 soft-to-bone
candidates, identical to all 2,162 candidates retained by the runtime filter.
At the current contact-enabled neutral checkpoint the counts are 1,953 and
1,953. Missing and extra sets are empty in both states. Every retained
candidate and every active collision stencil contains soft tissue and exactly
one target bone. Reference candidate ownership is 636 cranium and 1,526
mandible; current ownership is 652 cranium and 1,301 mandible. The same mesh
and `can_collide` filter are used when swept CCD rebuilds candidates, so fixed
cranium and moving mandible do not take different collision-selection paths.

This establishes complete candidate handling for the selected pure-face
discretization. It does not establish complete registered source-bone coverage.

## What the omitted mixed band means

The 6,926 omitted mixed faces occupy 2.3286% of FEM boundary area. Their exact
vertex-label patterns are:

| Vertex ownership | Faces | Area |
| --- | ---: | ---: |
| 1 cranium + 2 soft | 2,390 | 0.00127310 m² |
| 2 cranium + 1 soft | 1,975 | 0.00113261 m² |
| 1 mandible + 2 soft | 1,108 | 0.000381295 m² |
| 2 mandible + 1 soft | 1,038 | 0.000419766 m² |
| 1 cranium + 2 mandible | 199 | 0.0000562334 m² |
| 2 cranium + 1 mandible | 214 | 0.0000441396 m² |
| 1 cranium + 1 mandible + 1 soft | 2 | 0.000000553922 m² |

Thus `bonded_mixed_triangles_omitted` is a policy name, not a literal anatomy
classification: 413 omitted faces are cranium-mandible faces with no soft
vertex, and two contain both bones and soft tissue. The 70 mixed connected
components include the main cranium-soft and mandible-soft transition bands,
complex oral components, and 336 faces in small components that touch only one
pure class by boundary edge. A further 511 vertices occur only on mixed faces
and are absent from the collision mesh.

This ambiguity comes from source construction. Melon
`src/42-gen-masks.py` assigns each FEM boundary point the `GroupId` of the
closest source triangle with snapping enabled. It creates no independent
free-versus-bonded attachment field. The current rule can therefore be defended
as a conforming-mesh discretization choice, but the data cannot prove that
every mixed component is a correct anatomical attachment.

## Concrete omitted interactions

The mixed surface was intersected against each selected pure bone surface in
both states. Reference produces 7,752 raw mixed/cranium and 3,772
mixed/mandible triangle pairs; current neutral produces 7,635 and 3,765. Every
pair shares a global FEM vertex. After exact shared-vertex exclusion, all four
nonadjacent-intersection counts are zero. No concrete penetrating free-face
pair is therefore hidden by the mixed-face omission in either audited state.

The full FEM boundary has 43 edges incident to four faces: 20 lie wholly in
pure cranium, 18 wholly in pure soft tissue, three wholly in the mixed band,
and two cross pure/mixed classifications. This is a mesh-topology limitation,
although it does not change the exact static candidate-set equality above.

## Conclusion

There are zero unhandled soft-to-bone candidates on the selected FEM contact
surface at reference and current neutral. The frozen surface is internally
complete under the declared pure-face model and can remain unchanged for the
numerical experiment. Mixed attachment anatomy, four-face boundary edges, and
coverage relative to complete registered source bones remain explicit modeling
uncertainties. The current neutral checkpoint is unconverged, and these counts
are contact-map QA rather than final inverse-optimization trends.

`data/fem-contact-surface-audit/summary.json` records source and checkpoint
hashes, all component and face-pattern counts, exact candidate comparisons,
support ownership, and the no-mechanics-change receipt.
