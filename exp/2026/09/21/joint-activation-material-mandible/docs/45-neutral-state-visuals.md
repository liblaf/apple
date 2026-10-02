# Current contact-enabled neutral state

## Checkpoint identity and status

`src/45-render-neutral-state.py` read
`data/neutral-convergence-010-contact/terminal.pt` only after its size and
modification time were stable. The loaded 10,289,194-byte snapshot has SHA-256
`4e8f4f340a8d4bf62911dfbba2030068d5d901696bf30b2808c1536508f893c9`
and matches prepared input and manifest hashes.

This is update 3 after three accepted steps. It is the current
contact-enabled terminal state, not either older `neutral-prestress` pilot. It
passes the neutral geometry budget and contact validation, but the optimizer is
not converged and `preparation_complete=false`. The projected-gradient infinity
norm is 0.7063 against the required 0.001. These figures must therefore be
described as current-state diagnostics, not a converged preparation result.

The render ran locally through Cherries with:

```bash
DEBUG=1 \
CHERRIES_NAME="Current contact-enabled neutral visual QA final" \
CHERRIES_TAGS="joint-inverse,neutral,contact,visual-qa,cpu" \
uv run --frozen python src/45-render-neutral-state.py
```

## Surface displacement

| Front | Side |
| --- | --- |
| ![Front neutral drift](../data/neutral-state-visuals/01-contact-neutral-drift-front.png) | ![Side neutral drift](../data/neutral-state-visuals/01-contact-neutral-drift-side.png) |

Both views use the same 0--0.58770 mm scale. Surface displacement has median
0.13517 mm, 95th percentile 0.37908 mm, 99th percentile 0.47796 mm, and maximum
0.58770 mm. The saved solver metric is 0.18552 mm surface RMS, within its
0.25 mm budget.

| Front overlay | Side overlay |
| --- | --- |
| ![Front neutral overlay](../data/neutral-state-visuals/02-contact-neutral-overlay-front.png) | ![Side neutral overlay](../data/neutral-state-visuals/02-contact-neutral-overlay-side.png) |

Gray is the frozen reference, blue is the actual contact-enabled terminal, and
the red wireframe amplifies the terminal displacement 20 times. The overlays
show the spatial drift pattern; the red surface is only a visualization and is
not a solver state.

## Tetrahedral cross-sections

| Coronal | Sagittal |
| --- | --- |
| ![Coronal section](../data/neutral-state-visuals/03-contact-neutral-coronal-section.png) | ![Sagittal section](../data/neutral-state-visuals/03-contact-neutral-sagittal-section.png) |

The colored sections are cut through the actual deformed tetrahedral fixture;
black wire is the frozen reference section. They share the surface heatmap
scale. The current state has minimum `det(F)=0.64208`, 0.1% quantile 0.97192,
maximum 1.77059, zero inverted tetrahedra, and minimum skin-area ratio 0.95853.

## Current shared material state

| Coronal baseline stress | Sagittal baseline stress | Skin stiffness |
| --- | --- | --- |
| ![Coronal baseline stress](../data/neutral-state-visuals/04-baseline-stress-coronal.png) | ![Sagittal baseline stress](../data/neutral-state-visuals/04-baseline-stress-sagittal.png) | ![Uniform skin stiffness](../data/neutral-state-visuals/05-skin-stiffness-front.png) |

The two sections show the largest principal value of the checkpoint bulk
baseline tensor after weighting the shared fat, aponeurosis, and muscle tensors
by each tetrahedron's material fractions. Both use the same symmetric
`-11.039--11.039 kPa` scale. The largest principal value over modeled cells
ranges from -11.039 to 0.23507 kPa; the smallest principal value ranges from
-32.490 to -0.072902 kPa. These are optimized model fields from a
research-informed sensitivity configuration, not registered measurements.

The skin field is intentionally uniform. The approved basis has one shared
log-stiffness multiplier, 1.05918 at this checkpoint, which gives an effective
Young modulus of 0.211837 MPa from the 0.2 MPa reference. The fixed color scale
spans the model's 0.066667--0.6 MPa multiplier bounds. The checkpoint skin
baseline resultant is isotropic at 8.06 N/m.

## Contact at the current terminal

The solver receipt reports 138 active IPC representatives, minimum active gap
0.0492566 mm, barrier energy `6.21369e-15 MPa m³`, unit CCD fractions, and a
numerically valid frictionless contact state. The current visualization rebuild
also selected 138 representatives while reproducing the minimum gap, barrier
energy, locations, and nodal gradient. As in the reference audit,
representative count is not a physical or convergence metric.
The contact-force figures are kept separately from the reference contact audit:

| Current contact locations | Current bone-side forces |
| --- | --- |
| ![Current contact locations](../data/contact-neutral-visuals/01-active-contact-locations-side.png) | ![Current bone-side contact force](../data/contact-neutral-visuals/02-bone-contact-force-side.png) |

The rebuilt current state has 12 cranium and 126 mandible representatives. Its
bone-side resultant is 0.00056909 N on cranium and 0.000162618 N on mandible.
The associated gap histogram, force summary, captions, and units are in
`data/contact-neutral-visuals/summary.json` and `report.md`. Force values are
negative IPC energy gradients converted from MPa m² to newtons; they are not
pressure without a validated area division.

## Pure-surface completeness and attachment scope

The exact 128,172-triangle FEM boundary partitions without overlap or omission
into 41,903 pure cranium, 13,763 pure mandible, 65,580 pure soft-tissue, and
6,926 mixed-label triangles. IPC includes every pure-soft face against every
pure-cranium and pure-mandible face. The mixed transition triangles are omitted
from sliding contact under the declared bonded policy; that policy is a
discretization choice rather than an independently transferred anatomy tag.

This proves simulation-surface selection completeness under the declared
ownership rule. It does not prove that the mixed-label correspondence is the
correct anatomical muscle-bone, muscle-skin, oral, or periosteal attachment.
Every one of the 21,891 pure-cranium collider vertices belongs to the recovered
fixed cranium support, and every one of the 7,395 pure-mandible collider
vertices belongs to the differentiable rigid-jaw support. Historical `IsFixed`
would omit 1,070 and 1,267 of those vertices, respectively, so runtime ownership
is established by recovered group support rather than that historical flag.

The collider is still only the pure labeled boundary of the cropped face FEM.
The fixture retains 22,091 of 37,846 full-head Melon cranium support nodes,
although it retains all 7,510 mandible support nodes. Near-zero median and
sub-5-micrometre 95th-percentile FEM-to-source distances verify that both live
in the same world frame, but reverse source-vertex proximity is a crop and
tessellation diagnostic rather than direct contact coverage.

The soft-domain-conditioned audit is more relevant: 411 pure-soft vertices
(0.4573% of pure-soft vertex-area weight) lie within 1 mm of source cranium but
more than 2 mm from the pure cranium collider and more than 1 mm from bonded
mixed faces. The mandible count is 182 vertices (0.2490%). These regions are
localized in the midface/oral simulated domain in the issue maps under
`data/contact-visuals/`; they remain frozen-model relevance flags. Thus the
128,172-face partition is complete for declared simulation ownership but is not
complete anatomical bone coverage or a bijective source correspondence.

The complete-source triangle audit also finds that this current neutral state
is not collision-free against those source meshes. Cranium has 1,336 raw
pure-soft intersection pairs: 145 are confined to bonded mixed-face geometry,
1,191 are unconfined, and 114 lie more than 1 mm from bonded geometry. Mandible
has 773 raw pairs: 79 confined, 694 unconfined, and 49 more than 1 mm away. In
the fixed 411/182 source-near cohorts, 51 cranium-near and 14 mandible-near
vertices are inside the watertight source bones, with minimum signed clearances
of -0.11405 and -0.07383 mm. These results block direct insertion of complete
source bones as IPC obstacles; they do not by themselves label every overlap
an anatomical defect. See
[`docs/17-source-bone-contact-audit.md`](17-source-bone-contact-audit.md) for
the binding-coincidence classification and issue maps.

The mixed-source oral cavity, incomplete lower-teeth mapping, registered lip
defect, transferred expression targets, heuristic aponeurosis, and implicit
attachment geometry remain anatomical limitations.

`data/neutral-state-visuals/summary.json` contains dashboard-ready filenames and
captions, checkpoint and trace hashes, solver status, fixed visualization scale,
numerical geometry, contact receipt, oral diagnostic, and the surface-selection
audit. Its material-state block records constant bulk tensors or exact spatial
basis metadata, skin resultant, uniform stiffness multiplier, units, and
modeling status. Tuesday trends remain reserved for the actual final large
joint optimization after converged preparation; final inverse convergence is
not required.

The renderer supports the admitted Spatial80 checkpoint schema and verifies
its exact basis receipt before reconstructing per-cell stresses. It now adds
separate fat, aponeurosis and muscle largest-principal sections, before fraction
weighting, so the stiffer aponeurosis does not hide the other fields. Each
caption gives its tissue/checkpoint-specific signed scale. The real frozen
Spatial80 update-5 checkpoint produced 15 shape/material PNGs successfully in
`data/spatial25-u0005-shape-visuals-v2`; it remains unconverged and is labeled
accordingly. Numerical-only `stationary_outside_neutral_budget` states receive
an explicit shape-limit-failure label.
