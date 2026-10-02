# Review of the estimated 6-DoF mandible pose

The user requested a visual review before running smoothed carry. This task
therefore produced only a rigid geometry estimate, bounded geometric update
schedules, and preview assets. No forward solver was run for this new pose.
The saved plan status is `awaiting_user_pose_review`.

## Estimate and independent validation

`116-estimate-chin-rigid.py` fitted the same connected 27-vertex chin patch
from new neutral 005 to the transferred MouthOpen target using area-weighted
Kabsch alignment. The fitted rotation is unrestricted in three axes, and the
translation is unrestricted in world coordinates.

- Rotation magnitude: 10.014843 degrees.
- Translation about the existing pivot: (+0.516302, -5.821238, +1.222873) mm.
- Translation magnitude: 5.970662 mm.
- Weighted chin RMS: 0.179841 mm; maximum error: 0.388676 mm.

`118-preview-pose-path.py` independently applied the estimate using the native
`joint_equilibrium.rigid_displacement` on CPU. Its maximum discrepancy from
the NumPy/SciPy transform is 4.440892098500626e-16 m; the independently computed
chin RMS is 0.179840667384514 mm.

These are geometric checks, not force, collision or tetrahedron validity
claims. The chin is soft tissue rather than a direct observation of the bone.
The previous hinge-pose inversion findings remain relevant until the new
pose and its constraints are tested.

## Requested update limits

`mouthopen_pose_path.py` interpolates the relative rotation on SO(3) and the
translation of the shared jaw pivot. It asserts that every increment is at
most 1 degree and 1 mm translation. This does not cap the displacement of
every mandible vertex, which also includes rotation.

From the saved 4.129398-degree hinge checkpoint, the proposed schedule has
six increments: each rotates 0.986179 degrees and translates 0.995110 mm.
The planned endpoint equals the full chin estimate exactly. A separate
zero-pose schedule is included for reproducibility. Neither schedule has
been executed by a forward solver.

After user acceptance, the smoothed carry must use the exact relative rigid
transform for general 6-DoF poses. Subtracting two rotation vectors (as the
old fixed-axis hinge helper did) is not a valid relative rotation in general.
The existing full inverse runner remains a hinge runner and also needs its
pose parameters and derivative checks generalized before joint 6-DoF pose
and active-strain optimization. This preview does not claim those changes
are implemented.

## Preview

Tailnet: <PRIVATE_PREVIEW_URL>.

Orange wireframe is the neutral mandible; teal is the fitted mandible; gray
is the target skin; green marks the target chin patch. The fixed cranium and
eyeballs provide anatomical context. Front, side, and oblique views are
provided, plus opaque target-skin views and front/side chin-fit details.
Only the mandible is transformed; target skin is the saved blendshape target,
not a simulated tissue result.

The output is `data/review-repaired-reference-005/mandible-estimate/`, with
source SHA256 receipts, the numeric estimate, and `estimated-mandible.vtp`.
The existing transient preview service is reused. Links were added to the
neutral review and previous pose-jump review.

## Commands and run evidence

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='Chin pose native transform and bounded update preview' \
CHERRIES_TAGS='new-neutral,mouthopen,chin,rigid-pose,preview' \
.venv/bin/python src/118-preview-pose-path.py

CHERRIES_NAME='Chin estimated six DoF mandible pose final review' \
CHERRIES_TAGS='new-neutral,mouthopen,chin,rigid-pose,review' \
.venv/bin/python src/119-review-chin-rigid-pose.py
```

Cherries uses `ProfileJoint` with Git commits disabled. Logs are
`tmp/chin-rigid-path-preview.log` and `tmp/chin-rigid-pose-review-final.log`.
The estimate run is recorded separately in `116-chin-rigid-pose.md`.
The source and generated images were inspected, Ruff checks passed, and
native transform and increment limits were checked against actual assets.

Comet runs:

- Native transform and bounded schedule: <https://www.comet.com/liblaf/apple/b0eca705ee9c4e69bff399ea935a54bd>
- Final visual review: <https://www.comet.com/liblaf/apple/45b056f48e5741768c39a76ae6f7b9e5>

All five final image hashes were checked against the receipt. The final
index and side-view PNG returned HTTP 200 over the tailnet, and the side,
target-context and chin-detail images were visually inspected.
