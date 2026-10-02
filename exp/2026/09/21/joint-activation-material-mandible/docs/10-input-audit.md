# Joint inverse input and anatomy audit

## Decision

The historical full-active Apple fixture, expression cohort, activation graph,
and complete retained cranium/mandible supports are frozen and usable. The
six-DoF mandible anatomy gate is **blocked**. Do not present a joint
activation/material/mandible result as admitted until the lip/oral contact
surface and contact or inherited-exclusion semantics are repaired and reviewed.

The machine-readable decision is in
`data/prepared/manifest.json:gates.full_six_dof_jaw`. The compact arrays are in
`data/prepared/inputs.npz` (SHA-256
`72b69e30759b2c1922220a625aa54b763c684d9b9887233bbe4e3e358e59b5bd`).
`PreparedInputs.load()` verifies the NPZ, every array, and every frozen source
before exposing `.arrays` and `.manifest`.

## Frozen fixture and cohort

- Volume: 228,660 points and 1,146,517 tetrahedra from the June historical
  fixture. Its 288,235 active cells and 103 muscle labels are retained exactly;
  the later 120,020-cell named-face subset is not used.
- Supports: 7,510 mandible nodes and 22,091 retained cranium nodes. Every
  fixture node maps by `vtkOriginalPointIds`, exact coordinate, and exact
  `GroupId` to Melon. The mandible support retains all 7,510 Melon nodes. The
  face subvolume retains 22,091 of 37,846 full-head cranium nodes. These are the
  complete supports in this fixture, not teeth-only masks.
- Historical `IsFixed` has 27,036 nodes, all inside the recovered supports. It
  omitted 2,565 support nodes because the pipeline removes 2 mm lip, gingiva,
  and teeth proximity masks. The joint fixture must therefore use the recovered
  group supports rather than `IsFixed` as the jaw/skull definition.
- Training expressions, in fixed order: `Smile`, `LipsFunnel`, `BrowUpLeft`,
  `MouthOpenSlightly`. Reserved: `SmileClosed`, `JawLeft`. All six share 15,299
  exact volume-node observations and fixed neutral triangle-area weights.
  These are transferred Faceform fields and contain no measured rigid jaw pose.
- The same-muscle graph has 501,409 shared-face edges and 854 connected
  components. Conductance is shared-face area divided by centroid distance,
  multiplied by the harmonic mean of the two `MuscleFraction` values.

The NPZ stores active IDs, muscle IDs/fractions/effective volumes, graph arrays,
all support sets, observation IDs/weights/targets, fixed train/reserve indices,
the deterministic reserve split, and the mandible pivot/frame. It does not copy
the 77 MB volume or skin mesh; their absolute paths and SHA-256 identities are
in the manifest.

## Source anatomy audit

| Concern | Evidence | Status and consequence |
| --- | --- | --- |
| Common anatomy | Skin, bones, and muscles are separate registered sources. The skin is the `XYZ ReadyToSculpt` topology wrapped to GLB skin/eyes/gingiva (`melon/.../22-register-skin.py:13-25,69-127`); cranium and mandible are independently wrapped Sculptor objects. | Verified mixed-source construction. Treat correspondence as a modeling assumption, not same-subject anatomy. |
| Muscle attachments | The final Melon mesh has fractions and `MuscleId`, but no origin, insertion, attachment pair, sliding interface, tendon, or ligament arrays. Apple skin shares volume DOFs and has no independent interface state. | Verified absence. Generic bulk transmission is available; anatomical muscle-skin and muscle-bone attachment claims are not. |
| Aponeurosis | A hand-selected list of 23 named muscle families defines the SMAS source subset (`31-extract-muscles-smas.py:13-47`). Skin-normal rays up to 18 mm measure a span, missing values are heat-extended, two offsets are repaired and Boolean-differenced (`32-construct-smas-muscle-span.py:25-70,73-121`). The resulting sampled `SMAS - muscle` fraction becomes `AponeurosisFraction` (`41-volume-fraction.py:16-34`). | Verified heuristic construction. It is a usable material partition but not a traced aponeurosis, attachment map, continuity model, or fiber field. This can be recorded as a limitation for a numerical pilot; it blocks anatomical validation claims. |
| Lips and oral surface | The registered template has 74 open edges and is non-manifold. Upper/lower groups share vertices. After removing every triangle incident to all 58 shared upper/lower lip seam vertices, 17 source triangle-pair intersections remain in a 0.43 by 0.04 by 0.47 mm region near `[1.425, 2.161, 0.083]` m. Exact cell-pair IDs, contact points, and segment lengths are in the manifest. | Verified inherited intersection. This is a hard geometry gate for joint jaw optimization without a reviewed repair/contact model. Shared vertices and open edges alone remain QA flags rather than proof of pathological fusion. |
| Source lip to FEM map | Only 183/1,478 upper and 166/1,131 lower source lip vertices map exactly within `1e-12` m to fixture nodes. Nearest maps are non-injective; maximum distances are 1.153 and 1.225 mm. The exact-node conservative FEM classifier finds a different neutral set: two pairs while omitting 133 mixed-label triangles. | Verified insufficient correspondence. Source inherited pair identities cannot be propagated through FEM deformation. |
| Oral/teeth masks | `IsTeeth`, `IsGingiva`, and `IsLip` are 2 mm proximity masks (`42-gen-masks.py:85-104`), not separated contact surfaces. The fixture has no distinct lower-teeth surface or verified lower-teeth-to-mandible/FEM correspondence. | Verified construction; lower-teeth collision certification is unavailable. This remains a hard gate. |
| Expression transfer | Faceform's generic basemesh is wrapped to the current skin and its deltas are transferred (`61-delta-transfer.py:37-74`). Inner lips and mouth socket are excluded from the target during registration (`:25-34`). | Verified transfer chain. Use targets as fitting fields, not same-donor measurements or jaw-pose observations. |
| Nasolabial folds | No direct fold labels or observations exist, and the current aponeurosis/SMAS construction is heuristic. | Hypothesis only. A missing fold may involve the layer, target transfer, constitutive law, thickness, or activation; this audit cannot assign cause. |

The longer repository-wide provenance audit is
`apple/docs/research/2026-09-12-head-anatomy-pipeline-audit.md`.

## Jaw and oral geometry

The pivot is the midpoint of registered mandible landmarks 1 and 9. Their
bilateral posterior location overlaps the two posterior bone-contact clusters,
so they provide a reproducible hinge candidate. The landmarks are unnamed;
calling this a TMJ axis is an inference rather than source semantics. The frame
columns are the bilateral axis, projected world +Y, and the right-handed
anterior axis.

At rest, the separately registered source cranium and mandible have 128 contact
pairs (76 unique mandible cells):

- 80 pairs are in two bilateral posterior clusters. Their exact source cell
  pairs are frozen in the manifest.
- 48 are in three anterior/dental-region clusters. Positive opening clears the
  anterior contacts by 0.5 degrees; negative opening worsens them.

A candidate world-frame six-coordinate box was screened around a 1 degree
opening. The rotation-vector center is
`[0.0174528383, -0.0001128081, 0.0000559328]` rad; each world rotation component
varies by 0.05 degrees and each translation by 0.02 mm. All 64 corners had only
posterior skull contacts (71-82 pairs), zero upper-oral contacts, and zero
non-lower-skin contacts. Lower-oral source contacts remained (623-664 pairs).
The bilateral posterior contact-point AABBs and exact pose bounds are frozen in
`joint_pilot_contract`.

This corner screen is evidence for a narrow candidate, not admission. It does
not validate every interior pose, posterior/lower contact semantics, lip
contact, or teeth motion. `audit_deformed_oral_geometry(prepared,
deformed_points, pose_rad_m)` therefore:

1. checks the proposed pose bounds and the solved mandible support against the
   prescribed rigid transform;
2. checks source rigid mandible/cranium and template-skin intersections at the
   actual pose;
3. checks upper/lower lip and mandible/oral intersections on the actual deformed
   FEM boundary using exact node IDs;
4. rejects new boundary-cell pairs and inherited pairs whose intersection
   segment grows by more than `1e-8` m; and
5. always reports `admissible: false` while the manifest gate remains blocked.

For an independent zero-jaw prestress/equilibrium check, call the same function
with `neutral=True` and an exactly zero `pose_rad_m`, or call
`audit_neutral_oral_geometry(prepared, deformed_points)`. This path does not
apply the 1 degree expression box. It requires the solved mandible support to
remain at its zero-pose position and reports `neutral_invariants_ok`; it does
not pass the blocked full jaw gate.

Intersection-segment length is a diagnostic worsening proxy, not penetration
depth or a contact law. At neutral, the conservative exact-node classifier has
2 lip pairs, 12 mandible/upper-oral pairs, and 2,277
mandible/lower-oral pairs. The high inherited count reflects label adjacency
and unresolved interface semantics; it must not become a blanket exclusion.

## Required repair before joint fitting

The following work blocks a defensible activation/material/mandible fit:

1. Produce separated, manifold upper/lower lip and oral contact surfaces that
   map reproducibly to FEM boundary nodes, or provide a reviewed baseline
   contact-pair/exclusion map with a contact law and penetration metric.
2. Separate lower from upper teeth and establish a verified rigid
   lower-teeth-to-mandible correspondence in both source and FEM geometry.
3. Define posterior joint contact/support semantics. The provisional bilateral
   AABBs may be used for diagnostics, not silently excluded from mechanics.
4. Run the deformed-state audit on every optimizer proposal and solved state;
   fail on new or worsened contacts. A neutral-only or fixed-jaw physics run may
   test independent numerical gates but is not the requested joint result.

Mixed-source anatomy, heuristic aponeurosis, absent explicit muscle
attachments, and transferred expressions may be recorded as declared model
limitations for an engineering pilot. They prevent anatomical-validation
claims but do not by themselves prevent testing the passive/active solver after
the oral/jaw geometry gate is repaired.

## Reproduction and checks

The preparer is `src/10-prepare-inputs.py`; the loader and runtime audit are
`src/joint_data.py`. The final CPU preparation ran through Cherries' debug
profile and completed in 31.44 s without using the active GPU process. Logged
metrics were 288,235 active cells, 501,409 graph edges, 854 graph components,
7,510 mandible nodes, 22,091 cranium nodes, 15,299 observations, 128 rest
skull/mandible contacts, 17 source lip contacts, and 64 screened candidate
corners. Ruff formatting/checks passed, `PreparedInputs.load()` passed all
source/artifact/array hashes, and the neutral runtime audit reproduced the
frozen inherited contact counts.

An earlier normal-profile run wrote valid artifacts but was interrupted while
Comet attempted to upload an unusually large Git patch. It was superseded by
the completed non-committing debug-profile run; no simulation or GPU pilot was
started by this preparation task.

## Saved neutral pilot oral audit

The frozen `neutral=True` audit was applied post hoc to the displacement in
both `best-admissible.pt` checkpoints. The compact machine-readable result is
`data/prepared/neutral-pilot-oral-audit.json`. Both checkpoints retain exactly
zero mandible-support displacement and introduce no new or worsened contacts
in the lip-only classifier, but both fail the broader mandible/oral invariant:

| Checkpoint | New upper-oral pairs | New lower-oral pairs | Worsened inherited lower pairs | Lower intersection-segment sum |
| --- | ---: | ---: | ---: | ---: |
| `neutral-prestress-001` (0.806 N/m) | 1 | 371 | 96 | 0.194242 to 0.199677 m |
| `neutral-prestress-010` (8.06 N/m) | 3 | 362 | 98 | 0.194242 to 0.212557 m |

Intersection-segment length remains a diagnostic proxy rather than physical
penetration depth. These runs meet their declared deformation and determinant
budgets, but `neutral_invariants_ok` and `admissible` are false. They are
numerical-budget-only neutral pilots and are not fully oral-geometry-admissible.

## Topology-aware correction to the saved neutral audit

The paragraph above records the original conservative v1 interpretation and is
retained as provenance. The corrected regression receipt is
`data/prepared/neutral-pilot-oral-audit-v2.json`; it supersedes that numerical
interpretation without changing the v1 file or its raw VTK pair counts.

The v1 classifier compared collision-pair identities but did not exclude
triangles that are adjacent on the same FEM boundary. Exact global boundary
node inspection shows that all reference and saved-pilot mandible/oral and lip
pairs share a FEM vertex or edge. Every reported collision segment is confined
to that shared topology within a tolerance derived from the segment dtype and
model coordinate scale. There are zero nonadjacent pairs and zero adjacency
overruns. The 371/362 new lower pair identities and 96/98 longer inherited
segments are changes along the bonded interface, not free-surface penetration.

The unified FEM boundary also has zero pure-soft/cranium,
pure-soft/mandible, and mandible/cranium intersections in the reference and
both pilots. Consequently `neutral_invariants_ok` and
`numerical_geometry_admissible` are true in v2 for both stored pilots.
`anatomical_validation` and full `admissible` remain false: the registered
source lips retain 17 nonadjacent intersections, lower-teeth correspondence is
incomplete, and the full jaw/anatomy gate remains blocked. See
`docs/41-preparation-visuals.md` for the visual evidence and final numerical
admission semantics.
