# HFP1 full-head material-section evolution

## Purpose

This post-hoc render sections the complete HFP1 tetrahedral head at every saved inverse state, then applies only a camera crop around Orbicularis oris. It is not an Orbicularis-only extraction: fat, muscle, and aponeurosis are all cut from the 1,146,517-tetrahedron volume, and the separate Koiter skin mesh is intersected by the same plane.

## Section and camera contract

The 228,660-point head is sectioned before any camera decision. No spatial clip, threshold, or region extraction is applied to the volume. All 41 states use

```text
normal = (0, 1, 0)
origin = (0, 2.1730086794286745, 0) m
```

The fixed Y coordinate comes from the initial full-Orbicularis maximum-Z vertex: global point ID 52,222 at `(1.4077719415114796, 2.1730086794286745, 0.09695972390415916) m`.

Only after the full-head polygons and skin line have been generated is a fixed parallel camera applied. Its focus is `(1.4067350332694464, 2.1730086794286745, 0.08153530579422003) m`, its parallel scale is `0.055 m`, and its source bounds are the all-state full-Orbicularis bounds. Those bounds fit the view; the complete head section deliberately extends beyond it. This is a display crop, not a geometry crop.

## Material display

The cut polygons are opaque and use ParaView `Surface With Edges`. Dark cell edges (`RGB 0.12, 0.13, 0.15`, width `0.35 px`) expose the tetrahedron/plane-intersection shapes.

| Display class | Color | Continuous source field |
| --- | --- | --- |
| Fat | gold | `FatFraction` |
| Muscle | red | `MuscleFraction` |
| Aponeurosis | blue | `AponeurosisFraction` |

The class is explicitly a visualization-only `argmax(FatFraction, MuscleFraction, AponeurosisFraction)`. The renderer rechecks that equality for every polygon in every frame. The solver still uses the continuous fraction-weighted volume energies; 232,741 whole-volume tetrahedra have more than one fraction above `0.001`, so the colors are not sharp physical interfaces.

The skin is not a fourth volume material. It is the separate Koiter membrane, shown as one connected thin teal outline (`RGB 0, 0.38, 0.38`, width `1.5 px`) in every frame. No determinant or other metric panel is present.

## Observed output

Across steps 0–40, the complete plane intersection contains 16,732–18,440 polygons and 11,388–12,530 points before camera cropping.

| Class | Polygons per full section | Area in section (m²) |
| --- | ---: | ---: |
| Fat | 13,778–14,622 | 0.00606138–0.00617251 |
| Muscle | 2,703–3,673 | 0.00119546–0.00152205 |
| Aponeurosis | 141–251 | 0.00005136–0.00010134 |

The skin section stays one component with 281–329 line cells. The full-section union spans X `1.3458026053–1.4681043566 m` and Z `-0.0225165790–0.1005671730 m`. Step 0 has 11,388 points / 16,732 polygons; step 40 has 12,530 / 18,440.

## Media and validation

The media uses all 41 saved states exactly once at 30 FPS: 41 unique 1200 × 1200 RGB PNGs, a 1.366667 s H.264/yuv420p MP4 with 41 unique decoded frames, and a poster byte-identical to frame 40. The VTP series retain one polygon per intersected source tetrahedron. History step 40 reproduces the endpoint coordinates exactly. The PVSM is a substantive 397,881-byte temporal state.

Steps 0, 20, and 40 and the final poster were inspected at native resolution. The three materials, internal cut-cell edges, and teal skin outline remain distinguishable; the edges do not obscure the fills; and neither the title nor the categorical legend is clipped.

- [MP4 video](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution.mp4) — SHA256 `c90edaa1997c6a393a320ca58f31ce2b055a4432d4de666f8ca2d360c5daa4fb`
- [Final-frame poster](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution-poster.png) — SHA256 `990573ee46a35eb6fe108bb75ecaf6837bac2de19ee84d43a70c85ad4cfbabbc`
- [ParaView state](../data/60-hfp1-full-head-material-section-evolution/60-hfp1-full-head-material-section-evolution.pvsm)
- [PNG frames](../data/60-hfp1-full-head-material-section-evolution/frames/)
- [VTP intermediates](../data/60-hfp1-full-head-material-section-evolution/inputs/)
- [Render contract](../data/60-hfp1-full-head-material-section-evolution/contract.json)
- [Generation receipt](../data/60-hfp1-full-head-material-section-evolution/receipt.json)
- [ParaView receipt](../data/60-hfp1-full-head-material-section-evolution/renderer-receipt.json)
- [Per-step trajectory](../data/60-hfp1-full-head-material-section-evolution/trajectory.csv)

The superseded complete-fit/no-internal-edge output and its former site copies are preserved at `tmp/superseded-hfp1-full-head-material-section-complete-view-no-edges-20260902/`.

## Reproduction

Run from `exp/2026/08/19/material-physics-group-meeting-assets`:

```bash
DEBUG=1 \
CHERRIES_NAME="HFP1 full-head material section with tet edges" \
CHERRIES_TAGS="human-face,HFP1,full-head,material-section,mouth,fixed-plane,paraview,evolution,ios,surface-with-edges" \
uv run python src/60-hfp1-full-head-material-section-evolution.py
```

`DEBUG=1` made this a local Cherries run so the explicit no-commit/no-push boundary was preserved. The run completed cleanly with ParaView 6.1.1; its log is `logs/60-hfp1-full-head-material-section-evolution.log`. The input identities are pinned in the generation receipt, including the endpoint SHA256 `f93bf583...d53dd`, history `27f016f4...b86ce`, and skin `89e0b349...117d`.

## Limitation

This is a visualization of the existing `20-hfp1` trajectory; no physics was rerun. The inverse record is unconverged and stopped at `step_limit_smooth_decrease`. The camera intentionally excludes distant parts of the already-computed full-head section, and the categorical display should not be used as a replacement for the underlying continuous fractions.
