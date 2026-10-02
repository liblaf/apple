# Collision-off tissue estimate and push-out initializer

The user requested a test of a collision-off forward estimate at a proposed jaw
pose, followed by penetration repair and a collision-on equilibrium correction.
This compares that initializer with the existing coupled tangent from the same
independently audited run005 checkpoint. Muscle active strain is held fixed to
isolate the effect of changing the mandible pose.

Run005 was cleanly interrupted after accepted update 61 to serialize local GPU
work. Its checkpoint, including both Adam moment pairs and counters (127), is
preserved. A fresh rebuild with `146-audit-mouthopen-coupled.py` passed: weighted
skin RMS 3.0464148747443907 mm, force 0.00947097518116147 N, 91 inverted retained
tetrahedra, inverted rest-volume fraction 3.378139517298343e-5. This is a valid
forward state under the declared approximation, not inverse convergence.

The test recomputes the chin pose from the corrected neutral and MouthOpen target,
using the same saved chin patch. It takes one relative rigid increment toward that
estimate, bounded by 1 degree and 1 mm, then repeats at one quarter of that
increment. Each method starts from the same saved displacement, pose, materials,
and collision state.

Physical acceptance retains exact saved skin pre-strain, authoritative IsFixed
constraints, the exclusion of 2,249 fully fixed tetrahedra, barrier stiffness
1.3544 MPa, absolute force tolerance 1e-8 (0.01 N), at most 100 inverted retained
tetrahedra, and inverted rest-volume fraction at most 1e-4. No minimum-J floor is
added. Collision is disabled only during the alternative initial estimate. The
repaired seed and final collision-on endpoint must pass the unchanged geometry
policy. The old-to-seed CCD test is recorded separately; endpoint feasibility
alone does not authorize a discontinuous jump in the ongoing inverse fit.

Topology inspection found closed cranium and mandible surfaces. The eye mesh
has 32 apparent boundary edges caused by 14 exact duplicate seam vertices;
exact welding produces a closed query mesh without changing collision geometry.
A first local-plane repair failed and increased inversion counts. A subsequent
normal-derived signed-distance query falsely classified 24 source vertices as
inside the cranium; an independent ray query classified all 24 as outside, and
the original source had zero VTK soft-cranium intersections. That normal-sign
variant is not evidence against a correct containment-based repair. The final
variant uses ray containment for sign, unsigned closest-feature distances,
and minimal nodal corrections at barycentric triangle contact points. Volume
extension keeps IsFixed coordinates exact. Physical acceptance gates are unchanged.

The comparison has finite diagnostic budgets of 600 seconds per collision-off
estimate and collision-on correction. Budget exhaustion means a failed test, not
convergence. Ordered one-off timings include initialization overhead and share
the GPU with an unrelated existing process; they cannot establish a general
speedup.

Command, from this experiment directory:

```bash
CHERRIES_NAME='Compare collision-off pushout jaw initializer' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,initializer,comparison' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/152-test-mouthopen-collision-off-seed.py \
  > tmp/152-test-mouthopen-collision-off-seed-001.log 2>&1
```

The initial comparison exited on the eye-topology assertion after preserving
both deformation estimates. Larger motion: collision-off force did not reach
1e-8 within 600 seconds; a fresh evaluation of its saved raw diagnostic state
was 0.07138137251836833 N with 86 retained inversions. Smaller motion:
collision-off force converged in 190.96 seconds to 0.009930407173852517 N
with 94 retained inversions. The raw larger state is explicitly approximate;
its accepted-Newton-state provenance is not verified. Both saved estimates are
bound to exact pose, active strain, and complete material arrays before reuse.

Additive repair comparisons 002, 003 and 004 preserve the local-plane,
normal-sign and ray-containment variants respectively. These do not rerun or
relabel the collision-off force solves. Their wall times are additional repair
costs. The ray variant finished normally after both candidates failed repair; no
collision-on corrector was reached and no candidate was accepted.

Repair command (same working directory and OMP_NUM_THREADS):

```bash
CHERRIES_NAME='Test ray-classified pushout of saved jaw estimates' \
CHERRIES_TAGS='mouthopen,isfixed,active-strain,skin-prestrain,initializer,ray-containment,comparison' \
OMP_NUM_THREADS=4 .venv/bin/python -u \
  src/155-test-saved-mouthopen-estimates.py \
  --output-dir data/collision-off-seed-comparison-004 \
  > tmp/155-test-saved-mouthopen-estimates-004.log 2>&1
```

## Verified result

| Relative jaw update | Collision-off force | Off inversions | Final push-out inversions | Final inverted rest-volume fraction | Repair time | Contact result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.703241 degrees / 1 mm | 0.0713814 N; 600 s budget exhausted | 86 | 346 | 6.29164e-5 | 43.39 s | Triangle intersections remain, although all sampled vertices are outside |
| 0.175810 degrees / 0.25 mm | 0.00993041 N; converged in 190.96 s | 94 | 1756 | 3.29359e-4 | 45.00 s | Triangle intersections and five inside cranium vertices remain |

The existing tangent failed its old-to-candidate motion CCD test at fractions
0.12890625 and 0.53125 of those respective proposed moves. Neither tested
initializer supplied an admissible seed. There is no end-to-end speedup result.
The eight projection rounds are a diagnostic repair budget, not convergence.
The collision-on force gate was never evaluated at a repaired endpoint because
repair did not supply a feasible seed; force success refers only to the smaller
collision-off state. This is a limitation of the tested initialization and
projection procedures, not proof that every contact-free predictor must fail.

The old admissible collision mesh was checked with ray containment: all 34,245
soft collision vertices classify outside cranium, mandible, and eyes. The 24
normal-sign false positives were therefore a query artifact, not inherited
solid penetration. The final ray repair binds its complete materials, q and pose
to the saved estimates and leaves IsFixed coordinates exact. The main fit's
physical acceptance policy and optimizer state were never loosened.

## Evidence and previews

- Initial strict attempt: `data/collision-off-seed-comparison-001/`;
  [Comet](https://www.comet.com/liblaf/apple/56569aaf108f475f83b78448458496ac).
  Process exit 1 reflects the subsequently diagnosed eye seam assertion; the
  original collision-off receipts and snapshots are preserved.
- Final corrected repair: `data/collision-off-seed-comparison-004/`;
  [Comet](https://www.comet.com/liblaf/apple/601c5e5b19d4416d8380bb88053967e1).
  Both numerical rejection results and Cherries shutdown completed; exit 0.
- Independent run005 audit: `data/inverse-mouthopen-coupled-005/independent-audit.json`;
  [Comet](https://www.comet.com/liblaf/apple/39f5c02f5fcb46ffb3c39918dd5391c8).
- Full original tet boundary with bones/eyes and collision-off force histories:
  `data/review-collision-off-seed-004/`; all plotted pushed states are rejected
  diagnostic iterates, not forward equilibria.

Validation included exact IsFixed/free DOF checks, unchanged saved skin arrays,
source checkpoint hashes and q/pose/material byte comparisons, recomputed free
forces, harmonic extension residual checks, actual collision queries, retained
tet metrics, closed topology inspection, and cube/sphere/source ray-sign checks.
No commits or pushes were made.

## Publication and continuation

The full surface stages and force curves were rendered and visually inspected.
Seven HTML/JSON/PNG assets were verified byte-for-byte over the tailnet at
<PRIVATE_PREVIEW_URL>.
The existing transient preview service was reused.

The original optimizer resumed in additive `inverse-mouthopen-coupled-006`
through `153-continue-mouthopen-after-seed-test.py`, with its exact q, pose,
displacement and both Adam moment pairs/counters from audited run005. First
accepted update: RMS 3.036463255343473 mm, force 0.009740429699486668 N,
91 retained inversions. Live receipts point to run006. The audited full-surface
page was refreshed to run005. The convergence heartbeat remains active.

The reusable strict collision-off helper now dispatches to the same ray-based
repair tested from the saved states. The old local-plane routine remains only
for preserved diagnostic provenance; the current collision-on inverse does not
load or use either alternative initializer.
