# HFP1 Orbicularis oris: max-Z-anchored coplanar section evolution

## Purpose

This is a post-hoc visualization of the saved HFP1 inverse trajectory, focused on `Orbicularis oris001_Head_muscles_0` (ID 254).  It does **not** rerun forward or inverse physics.  The full selection used for determinant metrics is exactly

```text
ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5
```

with no selection crop: 10,484 tetrahedra and 3,248 points. The displayed geometry is a mouth-local cut through this full selection so that the initial muscle, deformed muscle, and skin are compared at one physical depth.

## Why the earlier skin curve appeared inside the muscle

The earlier view combined two different geometries: the complete three-dimensional muscle projected along `+Y -> -Y`, and a skin curve cut at one Y value. Projection discards Y depth, so that curve was not an outer silhouette of the volume. The apparent nesting failure was therefore a depth-comparison error, not by itself evidence that skin penetrated muscle.

The corrected video cuts the initial saved-state muscle, deformed muscle, and external skin at exactly the same fixed Y. The plane is anchored at the initial full-Orbicularis maximum-Z vertex: global point ID 52222 at `[1.4077719415114796, 2.1730086794286745, 0.09695972390415916]`, hence `Y = 2.1730086794286745 m`. Any contact or crossing now visible is a genuine same-plane relation rather than a projection artifact.

## Playback and view

The video contains the 41 consecutive recorded states, steps 0 through 40: one saved state per PNG and one PNG per encoded video frame.  It uses 30 FPS H.264/yuv420p playback, so its measured duration is 1.366667 s.  There is no interpolation, duplication, or deformation exaggeration; the final section is computed from the saved HFP1 endpoint exactly.

The camera is an orthographic head-superior view: it is placed on `+Y`, looks toward `-Y`, and uses `+Z` as image-up/anterior.  It is fitted only to the full Orbicularis bounds, not to the full-head skin contour, so the viewport stays on the mouth.  Camera and scalar ranges are fixed over the complete trajectory.  The iPhone-friendly 1200 x 1800 frame stacks three determinant panels:

- `DetF`: total deformation-gradient determinant;
- `DetAinv`: inverse activation determinant;
- `DetG`: elastic deformation-gradient determinant.

Blue/white/red encode the fixed full-muscle signed scalar range. Magenta marks cut cells whose source tetrahedra have both negative `DetF` and negative `DetAinv`. The dim gray initial saved-state cut (`0.22` opacity, `0.7 px`) is deliberately subordinate to the deformed muscle so it cannot obscure the skin. Black edges bound the deformed-muscle cut. Each label reports both the full-Orbicularis negative count and the number of unique source tetrahedra intersected by this section; the renderer deduplicates cut triangles by `SourceCellId`. Across the 41 frames, this plane intersects 529--1,149 unique muscle source tetrahedra.

The 2 px teal curve is the requested external-skin section. It is built only from external-surface triangles whose three endpoint vertices have `IsLip == true`: 2,275 points and 4,296 triangles. This semantic restriction excludes the nose. For each saved state, that dynamic lip-skin surface is intersected with the **same fixed** max-Z-anchored plane, `Y = 2.1730086794286745 m`. It is a section of the surrounding lip skin, not a muscle boundary or surface silhouette. Each frame contains exactly one arc, with 94--141 points and 93--140 line cells.

## Evidence and results

At step 0, all selected tetrahedra have positive determinants: `min(DetF)=0.953219`, `min(DetAinv)=1`, `min(DetG)=0.953219`, and no double-inverted tetrahedra.

Across the full muscle, the first negative `DetAinv` and `DetG` values occur at step 15; the first negative `DetF` and first double inversion occur at step 17. This particular fixed max-Z section intersects only one negative `DetAinv` and one negative `DetG` source tetrahedron at steps 23--27. It never intersects negative `DetF` or double-inverted source tetrahedra. At step 40 the full selection contains 29 negative-`DetF`, 58 negative-`DetAinv`, 33 negative-`DetG`, and 27 double-inverted tetrahedra, while all displayed-cut negative counts are zero. Thus this section makes the local muscle/skin geometry clear, but it does **not** expose the full-muscle inversion sites; the full counts in the labels remain essential context.

The original inverse run evaluated 41 states and selected step 40 as best, with zero recorded forward and adjoint failures.  It stopped at the configured step budget (`step_limit_smooth_decrease`) rather than an inverse-converged condition.  Therefore the clip documents the exact available trajectory, but should not be interpreted as a converged optimum.

## Assets

- [MP4 video](../data/50-hfp1-orbicularis-oris-topdown-evolution/50-hfp1-orbicularis-oris-topdown-evolution.mp4)
- [Final-frame poster](../data/50-hfp1-orbicularis-oris-topdown-evolution/50-hfp1-orbicularis-oris-topdown-evolution-poster.png)
- [ParaView state](../data/50-hfp1-orbicularis-oris-topdown-evolution/50-hfp1-orbicularis-oris-topdown-evolution.pvsm)
- [Generation receipt](../data/50-hfp1-orbicularis-oris-topdown-evolution/receipt.json)
- [ParaView renderer receipt](../data/50-hfp1-orbicularis-oris-topdown-evolution/renderer-receipt.json)
- [Full visualization contract](../data/50-hfp1-orbicularis-oris-topdown-evolution/contract.json)
- [Per-step determinant and skin-section trajectory](../data/50-hfp1-orbicularis-oris-topdown-evolution/trajectory.csv)
- [41 individual rendered frames](../data/50-hfp1-orbicularis-oris-topdown-evolution/frames/)

The receipts pin the input identities: the 41-state VTKHDF history is 2,072,672,205 bytes, SHA-256 `27f016f4a4b5cc4f54552ea7410c0a2feb758c646b24265e01680f31e29b86ce`; the endpoint, summary, and external-surface context are likewise hash-recorded there.

## Reproduction

Run from `exp/2026/08/19/material-physics-group-meeting-assets`:

```bash
DEBUG=1 \
CHERRIES_NAME="HFP1 max-Z-anchored Orbicularis and lip section" \
CHERRIES_TAGS="human-face,HFP1,orbicularis-oris,top-down,mouth,max-z-anchor,coplanar-section,evolution,paraview,step-0-40,ios" \
uv run python src/50-hfp1-orbicularis-oris-topdown-evolution.py
```

`DEBUG=1` deliberately disables Comet recording and Cherries Git-commit side effects for this local post-processing run.  It does not alter the source inverse result.  The pipeline validates the pinned source identities, prepares compact temporal inputs, renders with ParaView 6.1.1, and encodes with FFmpeg.

- [Orchestration script](../src/50-hfp1-orbicularis-oris-topdown-evolution.py)
- [ParaView renderer](../src/50-hfp1-orbicularis-oris-topdown-evolution-paraview.py)
- [Saved source trajectory](../../human-face-smile-fat-floor-skin-energy-prestrain-inverse/data/20-hfp1-steps.vtkhdf)
- [Source inverse summary](../../human-face-smile-fat-floor-skin-energy-prestrain-inverse/data/20-hfp1-summary-final.json)
