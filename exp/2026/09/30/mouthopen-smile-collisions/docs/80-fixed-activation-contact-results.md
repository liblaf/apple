# Fixed-activation bone-and-eye contact continuation

**Completed and independently audited.** The `52 → 75 → 75-002` chain solved all 121 cosine-spaced MouthOpen-to-Smile frames with the original saved activation tensors. Fresh force, prescribed-boundary, scoped contact, and checkpoint/provenance checks passed. The 1080p movie passes a full decode check, and the complete tetmesh preview was visually inspected. Inverted cells remain, so this is an exploratory continuation rather than a mechanically validated deformation.

## Frozen model

The repaired pruned reference has 227,900 physical points, 1,144,268 retained tetrahedra, and 288,172 active cells. Smile and MouthOpen activation tensors are byte-preserved through the original point/cell maps and are reused without refitting. Repaired coordinates rebuild rest gradients, volumes, dual weights, graph weights, and skin rest areas; exact tensor reuse therefore does not claim unchanged stress.

The original `IsFixed` policy remains authoritative. The saved jaw pose is prescribed on fixed mandible FEM points and appended source-mandible nodes; source cranium and eyes remain fixed. The original fat, aponeurosis, and muscle materials are used without a skin membrane. Harmonic weights only initialize a carry; equilibrium enforces the prescribed boundary.

Contact is frictionless standard IPC between pure-soft FEM boundary triangles and complete source cranium, mandible, and eyes. Mixed attachment faces and duplicate FEM bone faces are excluded. Tissue self-contact and rigid-rigid contact are not modeled. The barrier uses `1.3544 MPa`, `dhat = 100 µm`, 10 nm minimum CCD separation, and 0.1 nm `TightInclusionCCD` tolerance.

Preparation and adapter admission are recorded in [50 summary](../data/50-fixed-reference/summary.json), [51 CPU check](../data/51-fixed-reference-contact-check/summary.json), and [53 GPU check](../data/53-fixed-reference-gpu-check/summary.json).

## Terminal chain

| Segment | Status | Frames | Accepted checkpoints | Wall time | CG cap | Comet |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [52](../data/52-fixed-activation-contact/summary.json) | blocked | 28/121 | 82 | 3600.597 s | 1000 | [receipt](https://www.comet.com/liblaf/apple/6b63979e5fce4eb4b52807e1f1c2c288) |
| [75](../data/75-fixed-activation-contact/summary.json) | blocked | 85/121 | 105 | 3600.625 s | 3000 | [receipt](https://www.comet.com/liblaf/apple/060fbd4fcf034a42bcef776898d7b13f) |
| [75-002](../data/75-fixed-activation-contact-002/summary.json) | completed | 121/121 | 56 | 1257.293 s | 3000 | [receipt](https://www.comet.com/liblaf/apple/b3d4f41d0e7d4578b1cc771d90d6062b) |

Each later segment carries verified provenance from its predecessor. The final segment's frozen input hashes, source manifest, and resumed checkpoint are in its [summary](../data/75-fixed-activation-contact-002/summary.json); it reports `provenance_verified: true` and unchanged exact endpoint-S reuse.

## Independently verified frames

The [final audit](../data/54-fixed-contact-audit-003/summary.json) recomputed force and geometry for all 56 accepted checkpoints and all 121 animation frames. All 177 states passed the 0.01 N free-force gate and scoped contact checks. Maximum difference between saved and freshly computed force norms was `7.88e-16 N`. Across the 121 animation frames, free force ranged from 0.00154 to 0.00994 N. Active IPC stencils ranged from 571 to 745; minimum active separation ranged from 58.02 to 82.97 µm. These distances are active-pair values, so they do not assert `dhat` clearance when contact is active.

| Endpoint | Free force | Minimum active gap | Active contacts | Inverted cells | Minimum J |
| --- | ---: | ---: | ---: | ---: | ---: |
| MouthOpen | 0.00538 N | 73.36 µm | 621 | 469 | −16.892 |
| Smile | 0.00961 N | 69.80 µm | 736 | 110 | −3.140 |

All 121 saved-state intersection checks passed within the declared contact scope. The independent audit checks saved states; CCD trajectories remain recorded solver evidence rather than independently replayed paths.

The user permitted a few inversions; the experiment imposed a 0.1% cap (1,144 cells). The independent audit confirmed 67–469 inverted tetrahedra over the chain and 110 at final Smile; minimum J ranged from −16.892 to −1.790 and was −3.140 at final Smile. These states are exploratory and do **not** establish mechanical validity, even when force and scoped-contact gates pass.

## Why the 3000-step cap was retained

The immutable conditioning probe exhausted at 1000 PCG steps and converged at 2326 unshifted steps and 2514 with its representative shift; see [71](../data/71-transition-pcg-probe/summary.json). The frame-19-to-frame-20 paired pilot removed cap retries and reduced Hessian products from 37,910 to 6,554 while meeting the same gates; see [72](../data/72-linear-cap-pilot/summary.json). Its shared-GPU wall time is not a clean cap-only attribution because PNCG paths also differed. No physical, contact, force, or acceptance setting changed for the cap extension; see the [75 protocol](75-continuation-protocol.md).

## Reproduce

The terminal segment used:

```bash
CHERRIES_NAME="Finish saved activation contact from 79 percent Smile" \
CHERRIES_TAGS="mouthopen,smile,contact,fixed-activation,repaired-reference,continuation" \
.venv/bin/python \
  src/75-resume-fixed-activation-contact.py \
  --output 75-fixed-activation-contact-002 \
  --resume exp/2026/09/30/mouthopen-smile-collisions/data/75-fixed-activation-contact
```

The [final audit](../data/54-fixed-contact-audit-003/summary.json) is `verified_completed`. The full tetmesh movie is rendered. It is H.264/yuv420p, 1920 × 1080 at 30 fps, with 179 decoded frames (121 solved states plus endpoint holds) and a 5.966667 s duration. Full ffmpeg decoding and faststart checks passed; the [video QA receipt](../data/60-fixed-activation-contact-render/qa-receipt.json) records the movie hash, decoded frame count, timing, and endpoint holds. The clean preview was inspected for the MouthOpen endpoint, three intermediate states, and the Smile endpoint; labels do not overlap, and the contact scope and inversion limitation are explicit. The three segments contain 241 unique accepted solves after excluding the two inherited resume checkpoints. Their earlier wall-budget stops preserved verified progress. No activation fitting or Git commit/push was performed.

## Delivered assets

- [MouthOpen to Smile animation](../data/60-fixed-activation-contact-render/mouthopen-to-smile-fixed-activation-contact.mp4): H.264, 1080p, 30 fps, 179 decoded frames.
- [Clean keyframe preview](../data/81-contact-keyframe-preview-003/contact-keyframes.png): full tetmesh and activation at shared camera and scale.
- [Render manifest](../data/60-fixed-activation-contact-render/manifest.json) and [preview manifest](../data/81-contact-keyframe-preview-003/manifest.json) bind the assets to the audited states.
