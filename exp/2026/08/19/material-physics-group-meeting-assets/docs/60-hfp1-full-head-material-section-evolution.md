# HFP1 fixed-cohort crinkle-clip evolution

## Purpose

This post-hoc render applies a crinkle clip once to the complete HFP1 tetrahedral head at inverse step 0, then carries the selected tetrahedra through the remaining 40 saved states. It does not clip each deformed frame independently.

## Initial selection contract

The full 228,660-point, 1,146,517-tetrahedron head is evaluated at step 0 against

```text
normal = (0, 1, 0)
origin = (0, 2.1730086794286745, 0) m
retained half-space = y <= plane_y
```

The Y coordinate is unchanged from the superseded plane-section view. It comes from the initial full-Orbicularis maximum-Z vertex: global point ID 52,222 at `(1.4077719415114796, 2.1730086794286745, 0.09695972390415916) m`.

The crinkle predicate is `min(initial tetra vertex y) <= plane_y`. It retains every tetrahedron on the negative-Y side plus each complete boundary tetrahedron; it never cuts a tetrahedron into a polygon. The step-0 result contains 423,522 tetrahedra and 85,619 points. Of those, 16,732 strictly straddle the plane and another 11 touch it without a positive-area intersection.

The selected source-cell sequence has SHA256 `2cd6b6618b04b1b9ef5e365c26c1a4b7cf3cbf3b39c9b78000e88bbd05f8d204`; the source-cell-plus-connectivity digest is `e54791ee6386c8237475206fe07b32eebb9d253090b90dbfca3c1312ed58d18d`. Both are rechecked in all 41 generated VTU files and again by the ParaView renderer. Only coordinates change after step 0. The plane is not reapplied.

This is equivalent to ParaView `Clip` with `Crinkleclip = 1`, `Invert = 1`, and plane normal `(0, 1, 0)`. Keeping the negative-Y half exposes the jagged frontier to the existing camera on the positive-Y side.

## Skin trace and camera

The skin remains a separate Koiter membrane, not a fourth volume material. Its step-0 plane intersection contains 286 points and 285 connected line segments. Each intersection point stores its source mesh edge and initial interpolation weight; those weights are frozen and advected with the same saved states. The skin is therefore also not re-sectioned per frame.

The fixed parallel camera still focuses on the all-state Orbicularis-oris bounds at `(1.4067350332694464, 2.1730086794286745, 0.08153530579422003) m` with parallel scale `0.055 m`. The full crinkle result is computed before this display-only mouth crop. Its all-state bounds are X `1.3451348154–1.4689473781 m`, Y `2.1108289029–2.1821228653 m`, and Z `-0.0244642888–0.1012945682 m`.

## Material display

The selected volume is opaque and uses ParaView `Surface With Edges`. Dark edges (`RGB 0.10, 0.11, 0.13`, width `0.45 px`) outline the exposed tetrahedron faces, while modest diffuse lighting makes the crinkled depth legible.

| Display class | Selected tetrahedra | Color | Continuous source field |
| --- | ---: | --- | --- |
| Fat | 360,904 | gold | `FatFraction` |
| Muscle | 58,419 | red | `MuscleFraction` |
| Aponeurosis | 4,199 | blue | `AponeurosisFraction` |

The class is a visualization-only `argmax(FatFraction, MuscleFraction, AponeurosisFraction)`, verified in every frame. The solver still uses continuous fraction-weighted volume energies. No determinant panel is shown.

## Fixed-cohort evidence

The selected point count, tetrahedron count, material counts, source-cell digest, and topology digest remain unchanged at steps 0, 20, and 40. As a diagnostic, the number of selected tetrahedra that happen to strictly straddle the original world-space plane changes from 16,732 at step 0 to 17,568 at step 20 and 18,229 at step 40. That change is expected and demonstrates why reclipping later states would select a different cohort.

The skin trace remains one component with exactly 286 points / 285 lines. By step 40 it has moved as much as `0.00595836 m` from the initial plane because it is advected rather than recomputed there.

## Media and validation

The media uses all 41 saved states exactly once at 30 FPS: 41 unique 1200 × 1200 RGB PNGs, a 1.366667 s H.264/yuv420p MP4 with 41 unique decoded frames, and a poster byte-identical to frame 40. History step 40 reproduces the endpoint coordinates exactly. The PVSM is a substantive 400,046-byte temporal state.

Steps 0, 20, and 40 were inspected at native resolution. The jagged tetra faces, dark external cell edges, three categorical materials, advected teal skin trace, title, and legend are visible without clipping.

- [MP4 video](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution.mp4) — SHA256 `2d084fca25d040ba0b199719abd9b8e19b38c08643733ad429b6b962e0b1b7a1`
- [Final-frame poster](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution-poster.png) — SHA256 `4d65310d7fa0a3ccc2105be9ec9dc2070b52188f298f4e172bb2e5fabf58f72c`
- [ParaView state](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution.pvsm)
- [PNG frames](../data/60-hfp1-full-head-material-section-evolution/frames/)
- [VTU/VTP intermediates](../data/60-hfp1-full-head-material-section-evolution/inputs/)
- [Initial selection artifact](../data/60-hfp1-full-head-material-section-evolution/initial-crinkle-selection.npz)
- [Render contract](../data/60-hfp1-full-head-material-section-evolution/contract.json)
- [Generation receipt](../data/60-hfp1-full-head-material-section-evolution/receipt.json)
- [ParaView receipt](../data/60-hfp1-full-head-material-section-evolution/renderer-receipt.json)
- [Per-step trajectory](../data/60-hfp1-full-head-material-section-evolution/trajectory.csv)

The immediately superseded per-frame plane-intersection result, its published copies, report, and scripts are preserved at `tmp/superseded-hfp1-full-head-material-section-per-frame-plane-intersections-20260902/`.

## Reproduction

Run from `exp/2026/08/19/material-physics-group-meeting-assets`:

```bash
DEBUG=1 \
CHERRIES_NAME="HFP1 initial-frame fixed crinkle clip" \
CHERRIES_TAGS="human-face,HFP1,full-head,crinkle-clip,fixed-cell-cohort,mouth,paraview,evolution,ios,surface-with-edges" \
uv run python src/60-hfp1-full-head-material-section-evolution.py
```

`DEBUG=1` keeps the Cherries run local and preserves the no-commit/no-push boundary. The run completed with ParaView 6.1.1; its log is `logs/60-hfp1-full-head-material-section-evolution.log`. The pinned inputs include endpoint SHA256 `f93bf583...d53dd`, history `27f016f4...b86ce`, and skin `89e0b349...117d`.

## Limitation

This is a visualization of the existing `20-hfp1` trajectory; no physics was rerun. The inverse record is unconverged and stopped at `step_limit_smooth_decrease`. Opaque `Surface With Edges` shows the exposed faces of the selected volume, not tetrahedron faces buried in its interior. The categorical display should not replace the underlying continuous fractions.
