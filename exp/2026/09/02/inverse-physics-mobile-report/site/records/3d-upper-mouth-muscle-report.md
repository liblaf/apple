# Upper-mouth muscle section in the bumpy no-skin face result

## Conclusion

The active muscle nearest the saved bumpy top-lip surface is **Orbicularis
oris** (`MuscleId=254`, exact source name
`Orbicularis oris001_Head_muscles_0`), not the levator-labii candidate.  In the
materialized no-skin `lr3` best endpoint, the sufficiently muscular part of
this muscle (`ActivationMask && MuscleId == 254 && MuscleFraction >= 0.5`) has
10,484 tetrahedra, 71 `DetF < 0` tetrahedra, and 65 double-inverted
tetrahedra.  Its minimum `DetF` is `-7.394505`.

This localizes a severe folding route directly under the visibly bumpy upper
mouth.  It does **not** prove that these tetrahedra alone caused the surface
bumpiness: the evidence is post-hoc, the surface-to-volume transmission is not
isolated, and the historical inverse solve was not converged.

## Surface-anchored local section

The saved `IsLip` point marking—not an assumed anatomical coordinate—defines
the upper lip: the 852 `IsLip` points with `Y >= Q75 = 2.163818627362718`.
The whole selected Orbicularis oris set has minimum/median reference-centroid
distance to that surface of 0.8747 mm / 5.1370 mm.

The fixed local material section is reproducible:

1. Candidate cells are selected by the whole-muscle rule above and a reference
   centroid distance at most 2 mm from the upper-lip surface; this yields 637
   cells.
2. The seed is the candidate with lowest endpoint `DetF`: source cell 592453,
   reference centroid `(1.39646069, 2.16558851, 0.08848690)`, 1.4904 mm from
   the upper-lip marking.
3. The same-muscle 6 mm reference-centroid ball around that seed gives 1,078
   fixed source tetrahedron IDs.  The larger ball intentionally includes the
   immediately adjacent inner material; the 2 mm screen alone does not include
   the later deepest folds.

At the best checkpoint, step 194, the local section has 4 `DetF`-negative, 6
`DetAinv`-negative, 2 `DetG`-negative, and 4 double-inverted tetrahedra.  The
minima are `DetF=-3.013092`, `DetAinv=-0.917512`, and `DetG=-0.080727`.

`DetAinv < 0` and `DetG < 0` first appear at step 21.  `DetF < 0` and double
inversion first appear at step 52.  All four sign states remain present through
step 200.

## Artifacts

- [Global top-mouth context](../data/10-upper-mouth-muscle-folding/render/upper-mouth-global-context.png): orange is the full selected Orbicularis oris; magenta is the exact local section.
- [Step-194 determinant mechanism](../data/10-upper-mouth-muscle-folding/render/upper-mouth-primary-mechanism.png): three panels for `DetF`, `DetAinv`, and `DetG`; magenta cells are double-inverted.
- [Exact 201-frame, 30 FPS H.264/yuv420p video](../data/10-upper-mouth-muscle-folding/video/upper-mouth-muscle-evolution.mp4): one saved inverse state per frame, no interpolation or duplication.  It shows `DetF`; every frame's VTU still stores all three determinants.
- [Per-step determinant trajectory](../data/10-upper-mouth-muscle-folding/history/trajectory.csv)
- [Machine-readable selection and rendering receipt](../data/10-upper-mouth-muscle-folding/summary.json)
- [Exact source-ID frames](../data/10-upper-mouth-muscle-folding/history/frames/)

## Reproducibility and limits

Run from this group:

```bash
DEBUG=1 CHERRIES_NAME='Upper-mouth Orbicularis oris folding receipt' \
CHERRIES_TAGS='human-face,upper-mouth,orbicularis-oris,post-hoc,folding' \
uv run python src/10-upper-mouth-muscle-folding.py
```

The script reads only the saved endpoint and the 201-state VTKHDF history; it
does not rerun, repair, constrain, or otherwise alter face physics.  The saved
primary solve records `inverse/converged=false` and six forward failures.  The
localization therefore supports a plausible mechanism and a visual diagnostic,
not a converged causal attribution.
