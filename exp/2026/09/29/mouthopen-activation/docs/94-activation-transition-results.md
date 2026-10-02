# Smile to MouthOpen through activation

The animation is complete: **1920 × 1080, 30 fps, 5.966667 seconds**, with 121 independently re-equilibrated transition states and one-second endpoint holds. Shape and activation share a fixed camera, and the activation color and line-length scales remain fixed.

[Watch the MP4](../data/92-smile-mouthopen-transition-render/smile-to-mouthopen-transition.mp4) · [Keyframe preview](../data/95-transition-preview/smile-to-mouthopen-preview.png) · [Render manifest](../data/92-smile-mouthopen-transition-render/manifest.json)

![Smile to MouthOpen keyframes](../data/95-transition-preview/smile-to-mouthopen-preview.png)

## What was animated

The learned-axis Smile and MouthOpen activation endpoints are transferred by exact original tetrahedron IDs onto the pruned MouthOpen mesh. The path is `S(alpha) = (1-alpha) S_Smile + alpha S_MouthOpen`, with `B = I + S`. The full tensor drives every forward solve. Both endpoints are rank one and positive semidefinite; an intermediate blend can have two contraction modes. The activation panel shows the strongest positive principal mode, which can change direction near repeated eigenvalues even when the tensor varies continuously.

The prescribed mandible rotation and translation also transition from the neutral Smile pose to the chin-derived MouthOpen pose. Rotation follows the saved rotation vector about the saved pivot; translation is linear in alpha. Jaw motion is prescribed, not predicted from activation. Alpha follows a cosine easing curve over 121 states. This is a quasi-static path, with no inertial or timing calibration and no intermediate inverse fits.

The geometry is the complete boundary of the simulation tetmesh: **227,900 volume vertices, 1,144,268 tetrahedra, 288,172 active cells; 63,282 boundary vertices and 126,648 triangles**. The 63 active cells among previously removed fully fixed tetrahedra are omitted by exact mapping. See the [protocol](91-activation-transition-protocol.md) for endpoint hashes and the interpolation and solver policies.

## Observed numerical results

All **121 requested states** passed the strict physical free-force threshold `1e-10`. Maximum residual was `9.90157e-11`, median `7.83108e-12`. The completed transition needed **zero rejected proposals and zero subdivisions**. The successful run's numerical section took **831.68 seconds**.

| State | Activation fraction | Face fit RMS (mm) | Inverted tetrahedra | Minimum physical J |
| --- | ---: | ---: | ---: | ---: |
| Re-equilibrated Smile | 0 | 1.567777 | 126 | -3.288955 |
| Frame 30 | 0.146447 | — | 84 | -2.093234 |
| Frame 60 | 0.5 | — | 142 | -7.600990 |
| Frame 90 | 0.853553 | — | 371 | -14.099496 |
| Re-equilibrated MouthOpen | 1 | 0.660866 | 476 | -16.892447 |

Intermediate frames have no measured target expression, so no intermediate fit metric is reported. Across all frames, inverted-cell counts ranged from **70 to 476**, below the declared limit of 1,144. All 121 complete boundary checks detected self-intersection. Contact remains off and no skin membrane is present. Small force residuals establish numerical equilibrium, not mechanical stability, anatomical validity, or collision feasibility.

The historical Smile checkpoint used approximate solves. Re-equilibration changed its fitted face by **0.419632 mm RMS**, with maximum displacement change **4.604648 mm** over all retained vertices. The saved strict MouthOpen endpoint replayed exactly from its own seed. The continuous transition reached a face within **0.000263860 mm RMS** of that saved endpoint, but maximum difference over all volume vertices was **1.176697 mm**. Thus the endpoint has nearly the same fitted face, while the complete volume state is not identical. The final frame is the actual continuation solution; it was not replaced with the saved endpoint.

Maximum consecutive-frame displacement RMS over all volume vertices was **0.097043 mm**, with median **0.069156 mm**. The independent audit verified all frame hashes and fixed jaw constraints, exact mesh/tensor transfer, positive-semidefinite blend samples, source hashes, endpoint drift and face fit, and independently recomputed deformation determinants at five keyframes. It checked saved strict residual receipts rather than repeating every GPU solve.

## Initial replay recovery

The first two initial Smile replays exhausted 100 and 1,000 Newton iterations respectively; they produced no accepted animation frames and are preserved in `data/91-smile-mouthopen-transition/` and `data/91-smile-mouthopen-transition-002/`. The second stopped at force `2.85444e-10` because the near-tolerance search policy repeatedly reset to an unshifted system and then selected a large stabilization shift after negative-curvature rejection.

The successful attempt used that finite failed iterate only as a warm start and set `reuse_shift_force_ratio=0` through a process-local wrapper. This preserves stabilization reuse through convergence. It changes the Newton search policy, while retaining the physical energy, force, Hessian, line search and acceptance threshold. The initial strict replay then converged in 28 additional Newton iterations. Frozen numerical model sources and the original fitted checkpoints were preserved.

## Evidence and reproduction

- [Numerical summary](../data/91-smile-mouthopen-transition-003/summary.json), [endpoint tensors and exact maps](../data/91-smile-mouthopen-transition-003/endpoints.npz), [frozen source manifest](../data/91-smile-mouthopen-transition-003/source-manifest.json).
- [Independent audit](../data/93-transition-audit/analysis.json), [numerical log](../logs/91-activation-transition-003-terminal.log), [audit log](../logs/93-transition-audit-terminal.log).
- [Numerical Comet run](https://www.comet.com/liblaf/apple/928016dfbd924cb593d88058434a0439): `2026-09-29 15:29:44.367272+08:00` to `15:43:45.201300+08:00`.
- [Audit Comet run](https://www.comet.com/liblaf/apple/fb8034471d8449e3a103d3e61e9558c0): `2026-09-29 15:44:20.274049+08:00` to `15:44:27.608421+08:00`.
- [Render Comet run](https://www.comet.com/liblaf/apple/c41bc740129e4d4d903613e5dc54ab3d): `2026-09-29 15:44:19.351937+08:00` to `15:50:06.779434+08:00`; [render log](../logs/92-transition-render-terminal.log).
- [Decoded preview Comet run](https://www.comet.com/liblaf/apple/7b7b150626ea4f6db29f1a219a70119b): `2026-09-29 15:53:10.415241+08:00` to `15:53:11.087902+08:00`; [preview manifest](../data/95-transition-preview/manifest.json).

All four final processes, including Cherries shutdown, exited successfully. Cherries recorded Git revision `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the scripts used the working tree with automatic commits disabled. No commits or pushes were made. The initial attempts, their logs, and all numerical checkpoints remain available.

The MP4 contains exactly **179 decoded frames**: 121 transition states plus 29 extra repeats at each endpoint. FFprobe verified frame count, resolution, rate and duration. Full-stream FFmpeg decoding completed without errors; 146 render-manifest hash records passed verification. Start, midpoint and end images were inspected from the decoded video. Framing and annotations were readable, the jaw opened through the sequence, and the full-boundary shape and activation views remained aligned. [Video verification receipt](../tmp/94-video-qa/verification.json).

The first renderer's contact sheet had overlapping labels. It remains preserved, while the linked final preview was recomposed from five decoded MP4 frames with separate labels and visually checked. This preview repair changed no animation frame, simulation state, or numerical result.

Run from `exp/2026/09/29/mouthopen-activation` with fresh output names when repeating:

```bash
CHERRIES_NAME='Smile to MouthOpen transition with continuous shift reuse' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,forward-equilibrium' \
.venv/bin/python -u \
  src/91-solve-activation-transition.py \
  --output 91-smile-mouthopen-transition-003 \
  --initial-seed data/91-smile-mouthopen-transition-002/failed-solver-state.npz

CHERRIES_NAME='Independent Smile to MouthOpen transition audit' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,audit' \
.venv/bin/python -u \
  src/93-audit-activation-transition.py

CHERRIES_NAME='Smile to MouthOpen full tetmesh animation' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,render,full-tetmesh' \
.venv/bin/python -u \
  src/92-render-activation-transition.py

CHERRIES_NAME='Decoded Smile to MouthOpen keyframe preview' \
CHERRIES_TAGS='mouthopen,smile,activation-transition,preview' \
.venv/bin/python -u \
  src/95-make-transition-preview.py
```

The inverse-fit configuration fields inherited by the forward script are unused; it performs no optimizer updates. Ruff and Python compilation checks passed for all four new scripts.
