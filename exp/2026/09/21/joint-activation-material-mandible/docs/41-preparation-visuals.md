# Preparation visual QA

## Scope

`src/41-render-preparation.py` renders the frozen full-active fixture and the two
saved neutral prestress pilots with fixed cameras and shared displacement
scales. The figures inspect the actual FEM topology and saved displacements;
they do not infer Tuesday trends. Expression or material trends are reportable
only from the final converged large joint optimization.

The renderer ran locally through Cherries with:

```bash
DEBUG=1 \
CHERRIES_NAME="Preparation visual QA final" \
CHERRIES_TAGS="joint-inverse,preparation,visual-qa,cpu" \
uv run --frozen python src/41-render-preparation.py
```

It wrote 16 PNGs and `summary.json` under
`data/preparation-visuals/` in 10.26 seconds. Ruff format and check passed. The
run used CPU/offscreen PyVista and did not touch the active GPU process.

## Frozen anatomy and constructed materials

| Front anatomy | Side anatomy |
| --- | --- |
| ![Front anatomy](../data/preparation-visuals/01-anatomy-front.png) | ![Side anatomy](../data/preparation-visuals/01-anatomy-side.png) |

The blue and orange surfaces are the cranium and complete mandible regions of
the tetrahedral fixture. The red field is active muscle fraction and the pale
surface is the registered skin. These views expose the actual attachment
geometry rather than an independent display mesh.

| Dominant material on a coronal slice | Constructed aponeurosis on a sagittal slice |
| --- | --- |
| ![Material partition](../data/preparation-visuals/03-material-coronal-slice.png) | ![Aponeurosis](../data/preparation-visuals/04-aponeurosis-sagittal-slice.png) |

The aponeurosis is a constructed fraction field, not a segmented anatomical
layer. Its location can support the engineering experiment, but it remains a
modeling assumption relevant to the absent nasolabial fold.

## Soft tissue and bone contact ownership

| Front | Side |
| --- | --- |
| ![Front contact partition](../data/preparation-visuals/05-contact-surface-partition-front.png) | ![Side contact partition](../data/preparation-visuals/05-contact-surface-partition-side.png) |

The contact map is a partition of one connected FEM boundary using its global
fixture node IDs. It contains 128,172 triangles: 41,903 pure cranium, 13,763
pure mandible, 65,580 pure soft tissue, and 6,926 mixed bone-soft transition
triangles. The mixed magenta triangles are shared-topology bonded seams. They
must not be duplicated into separate contact meshes or treated as sliding
interfaces. Pure soft faces against nonadjacent pure cranium or pure mandible
faces are contact-eligible.

At reference, pure-soft vertices approach the pure cranium to 0.0712 mm and the
pure mandible to 0.0275 mm. There are 124/647/1,805 soft vertices within
0.25/0.5/1.0 mm of cranium, and 86/334/866 within those distances of mandible.
A sub-millimeter broad phase is therefore practical, but contact initialization
must handle the small existing gaps. Exact free-surface intersection counts are
zero for soft-cranium, soft-mandible, and mandible-cranium in the reference and
both saved neutral pilots.

## Neutral deformation

Both heatmaps use the same 0--0.6067 mm scale. The 20x overlays make the drift
direction visible while retaining the actual deformed surface in blue.

| Pilot | Drift heatmap | Reference / actual / 20x overlay |
| --- | --- | --- |
| 0.806 N/m | ![001 drift](../data/preparation-visuals/10-prestress-001-drift-front.png) | ![001 overlay](../data/preparation-visuals/11-prestress-001-overlay-front.png) |
| 8.06 N/m | ![010 drift](../data/preparation-visuals/10-prestress-010-drift-front.png) | ![010 overlay](../data/preparation-visuals/11-prestress-010-overlay-front.png) |

| Saved pilot | Median | 95th percentile | Maximum | Surface RMS from checkpoint |
| --- | ---: | ---: | ---: | ---: |
| `neutral-prestress-001` | 0.0153 mm | 0.0381 mm | 0.0652 mm | 0.0195 mm |
| `neutral-prestress-010` | 0.1414 mm | 0.3773 mm | 0.6067 mm | 0.1881 mm |

These are saved numerical pilots, not preparation convergence evidence. The
renderer accepts later checkpoints through `--checkpoints` and should be rerun
on the contact-enabled converged state.

## Oral topology and inherited source defect

![Registered source lip intersections](../data/preparation-visuals/29-source-lip-intersections.png)

The registered template retains 17 nonadjacent upper/lower lip intersections
after removing every triangle incident to the shared lip seam. This is a real
source-anatomy defect. Exact source-to-FEM lip correspondence is sparse and
non-bijective, so it cannot be used as a blanket failure of the tetrahedral
simulation boundary.

| FEM state | Mandible/upper raw pairs | Mandible/lower raw pairs | Nonadjacent pairs | Adjacency overruns |
| --- | ---: | ---: | ---: | ---: |
| Reference | 12 | 2,277 | 0 | 0 |
| `neutral-prestress-001` | 11 | 2,299 | 0 | 0 |
| `neutral-prestress-010` | 11 | 2,289 | 0 | 0 |

![FEM oral topology at the stronger saved pilot](../data/preparation-visuals/30-fem-oral-prestress-010.png)

Every raw FEM oral pair shares an exact global boundary vertex or edge. A pair
is excluded from penetration admission only when both reported collision
segment endpoints remain on that shared vertex or edge within a tolerance
derived from VTK segment precision and model coordinate scale. All reference
and saved-pilot pairs meet that test. The 371 and 362 lower-oral pair identities
that appear in the two pilots are therefore changes along a bonded interface,
not free-surface penetration. Raw counts remain in the receipt for diagnosis.

## Readiness conclusion

The corrected FEM geometry audit passes both saved neutral pilots: mandible
support is exact, no nonadjacent or adjacency-overrun oral intersections occur,
and no free soft-tissue intersection with either bone occurs. This establishes
numerical geometry admissibility for those stored states. It does not establish
anatomical validation, contact-enabled equilibrium convergence, or a valid
large joint fit.

The final simulation admission should require:

1. finite support and six-DoF pose consistency;
2. no inverted tetrahedra and the declared determinant and surface-area
   budgets;
3. CCD/barrier checks for free soft tissue against both cranium and mandible,
   excluding only exact shared-topology bonded seams;
4. no new nonadjacent surface intersection or collision segment extending
   beyond a shared vertex/edge;
5. contact-enabled neutral equilibrium convergence before continuation; and
6. convergence and validity of the final large joint optimization before any
   Tuesday material, activation, or jaw trend is reported.

The unresolved registered-lip defect, incomplete lower-teeth correspondence,
mixed-source oral cavity, transferred expression targets, heuristic
aponeurosis, and implicit muscle attachment geometry remain declared anatomy
limitations. They constrain anatomical interpretation but do not justify
rejecting a numerically valid FEM state solely because bonded boundary triangles
touch along their shared topology.

Machine-readable values, input hashes, checkpoint hashes, and every generated
filename are in `data/preparation-visuals/summary.json`. The corrected contact
regression receipt is `data/prepared/neutral-pilot-oral-audit-v2.json`.
