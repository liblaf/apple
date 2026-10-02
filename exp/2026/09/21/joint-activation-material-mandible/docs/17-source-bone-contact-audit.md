# Complete source-bone contact compatibility audit

`src/17-audit-source-bone-contact.py` compares the pure-soft boundary of the
frozen FEM fixture with the complete registered source cranium and mandible. It
is read-only and does not alter the frozen IPC surface, binding, or mechanics.
The reference state and the current contact-enabled neutral checkpoint are both
tested.

The source cranium and mandible are closed manifold triangle meshes with zero
open edges. Their median distance from matching FEM collider vertices is at
machine precision and their 95th percentiles are below 5 micrometres, so the
source and FEM geometries share a world frame. This supports signed-clearance
measurement. Clearance is negative inside the watertight source bone and
positive outside; containment determines the sign, independent of input normal
orientation.

## Reference state

| Source cranium | Source mandible |
| --- | --- |
| ![Reference cranium intersections](../data/source-bone-contact-audit/reference-cranium-source-intersections-front.png) | ![Reference mandible intersections](../data/source-bone-contact-audit/reference-mandible-source-intersections-side.png) |

The complete cranium has 1,857 raw triangle-intersection pairs with the pure-soft
FEM surface. Of these, 594 collision segments and midpoints are confined to the
bonded mixed-face geometry within the 8.84 micrometre float32 source-coordinate
tolerance. The other 1,263 are not confined; 1,114 remain within the declared
1 mm attachment neighborhood and 149 lie farther than 1 mm from bonded
geometry.

The complete mandible has 766 raw pairs: 274 confined bonded coincidences, 492
unconfined pairs, 478 within the 1 mm attachment neighborhood, and 14 farther
than 1 mm from bonded geometry. These meshes have no shared topology, so
"bonded coincidence" is a geometric classification rather than a topological
adjacency exclusion.

Within the fixed source-near relevance cohorts, the 411 cranium-near vertices
have signed clearance from -0.04369 to 0.99899 mm; 33 are inside the source
cranium. The 182 mandible-near vertices range from 0 to 0.99981 mm with none
strictly inside. The cohorts participate in 163 cranium and 33 mandible raw
intersection pairs; 132 and 9, respectively, lie farther than 1 mm from bonded
geometry.

## Current neutral state

| Source cranium | Source mandible |
| --- | --- |
| ![Current cranium intersections](../data/source-bone-contact-audit/current_neutral-cranium-source-intersections-front.png) | ![Current mandible intersections](../data/source-bone-contact-audit/current_neutral-mandible-source-intersections-side.png) |

The current neutral state has 1,336 cranium pairs: 145 confined coincidences,
1,191 unconfined, and 114 farther than 1 mm from bonded geometry. It has 773
mandible pairs: 79 confined, 694 unconfined, and 49 farther than 1 mm. Relative
to the reference cell-pair identities, cranium retains 693 pairs, adds 643, and
loses 1,164; mandible retains 360, adds 413, and loses 406.

The tracked cranium cohort now ranges from -0.11405 to 1.11445 mm signed
clearance with 51 vertices inside. Its incident intersection count is 110, of
which 91 lie farther than 1 mm from bonded geometry. The mandible cohort ranges
from -0.07383 to 0.99567 mm with 14 vertices inside. It participates in 57
pairs, 44 farther than 1 mm from bonded geometry. The current neutral optimizer
is still unconverged; these changes diagnose compatibility and are not final
joint trends.

Green markers in the figures are collision geometry confined to bonded mixed
faces. Amber markers are unconfined intersections within 1 mm of bonded
geometry. Red markers are farther than 1 mm. Black halos identify pairs incident
to the fixed 411/182 relevance cohorts. Intersection-segment lengths are
diagnostic proxies, not penetration depths or contact areas.

## Admission consequence

The complete registered source-bone meshes cannot be added directly as IPC
obstacles. They already intersect the pure-soft FEM surface in the reference
state and continue to do so at the current neutral state. A reviewed
bone-to-tissue binding or exclusion map is required first, followed by repair,
collision-free offsetting, or remeshing of the remaining free patches. The
source-near regions remain modeling limitations rather than automatic anatomy
defects, but the existing pure FEM colliders do not establish complete relevant
bone-contact coverage.

`data/source-bone-contact-audit/summary.json` contains source hashes, exact cell
counts, signed-clearance quantiles, intersection classifications, pair changes,
thresholds, asset captions, and the explicit no-mechanics-change receipt.
