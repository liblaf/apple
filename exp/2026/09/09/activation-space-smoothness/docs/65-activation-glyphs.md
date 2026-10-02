# Muscle activation glyphs

The glyph gallery visualizes the saved learned-axis controls for six states. It contains 12 standalone PNGs, sampled activation fields and prebuilt line glyphs in VTP format, and the unchanged rest skin. No inverse fitting or mechanical solve was run for this visualization.

## Meaning of the glyphs

For each active tetrahedron, the saved parameter q is the learned vector v. The model uses C = vvᵀ and B = I + C = A⁻¹. With s = ‖v‖², the preferred axial stretch in A is 1/(1+s), so commanded axial shortening is s/(1+s).

A centered rod represents the unoriented axis of v. Its two ends have equal meaning because v and −v produce the same tensor. Rods sit at tetrahedron centers in the reference configuration, over the rest skin. They show learned control directions, not prescribed anatomical fibers, force vectors, displacement vectors, or observed tissue strain.

The fixed linear color scale runs from 0% to 100% commanded shortening. Display length is **1.5 mm + 10.5 mm × shortening fraction**. The 1.5 mm minimum keeps weakly activated axes visible; neither this minimum nor the 12 mm maximum is a physical fiber length or displacement. A pale under-stroke improves contrast without changing the scalar values.

The linear scale separates small commands from large commands, but does not resolve small differences below 1%. For example, 95% of the sampled rate-0.3 endpoint values are at or below 0.908%. The exact values remain available in the VTP files.

## Sampling and cameras

The fixture contains 288,235 active tetrahedra. A fixed 6 mm grid in rest coordinates selects the smallest GlobalCellId in each occupied voxel, yielding 3,218 actual cells. This selection depends only on geometry and is reused across all six states. It includes all 103 activation regions in this fixture.

One cell per voxel does not retain every neighboring muscle: 1,222 occupied voxels contain multiple MuscleId values, and the selection omits 1,558 additional muscle/voxel combinations. The images are spatial samples, not exhaustive activation maps or estimates of regional averages. They do not necessarily include the strongest cell.

Both views use the existing frozen orthographic cameras. Centers outside the square viewport or more than 30 mm behind its focal plane are excluded. This leaves 2,113 rods in the overview and 156 in the mouth-corner view, identically across states. The images use 1,000 × 1,000 pixels, a 3 px colored line over a 5 px pale under-stroke at opacity 0.38, and rest-skin opacity 0.12. The underlying sampled VTPs retain all 3,218 cells without the view clipping.

## Saved states

The fit, motion, and inversion counts below describe the saved mechanical states. The background geometry in the glyph images remains the reference skin. These states have different fits and budgets and are not a matched test of smoothness.

| State | Update | Learning rate | Fit RMS (mm) | Motion RMS (mm) | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, smoothing off, latest full checkpoint | 16 | 7.857986 | 3.600656 | 4.472303 | 1,035 |
| Original, smoothing on | 128 | 7.857986 | 0.681094 | 5.179578 | 94 |
| Quarter rate, smoothing off | 64 | 1.964496 | 2.587802 | 4.358874 | 409 |
| Quarter rate, smoothing on | 64 | 1.964496 | 5.152102 | 7.308119 | 2,189 |
| Rate 0.3, smoothing on, phase boundary | 128 | 0.3 | 4.382659 | 1.745661 | 1 |
| Rate 0.3, smoothing on, endpoint | 256 | 0.3 | 2.876797 | 4.047514 | 73 |

The original unsmoothed run has no full NPZ at its last accepted surface state, update 28. Its panel therefore uses the latest numbered full checkpoint, update 16, rather than its failed controls or a surface-only record.

![Rate-0.3 activation overview at update 256](../data/65-activation-glyphs-v3/geometry/rate03-on/side-context.png)

![Rate-0.3 activation mouth close-up at update 256](../data/65-activation-glyphs-v3/geometry/rate03-on/region1-mouth-corner.png)

## Files and reproduction

The complete gallery (private preview omitted) includes each standalone view. The [ParaView bundle](../data/65-activation-glyphs-v3/glyph-data.zip) contains prebuilt line glyphs, sampled point fields, rest skin, and a README. Open the rest skin and a glyph VTP in ParaView to rotate the field; apply a Tube filter if cylindrical rods are preferred.

The exported arrays include GlobalCellId, MuscleId and name, ActivationControlId, MuscleFraction, raw squared control norm, and exact shortening fraction/percentage. LearnedAxisRest is an arbitrary signed representative of an unoriented axis; use centered line or cylinder glyphs, not single arrowheads. LearnedAxisDyadRest stores the sign-invariant n nᵀ tensor as nine components. MuscleFraction is available for filtering; it is not multiplied into the displayed shortening command.

The [renderer](../src/65-render-activation-glyphs.py) validates the learned-axis model, valid saved solver state, fixture and rest-point identity, active-cell mapping, rate and case identity, and C = qqᵀ. Its [receipt](../data/65-activation-glyphs-v3/summary.json) records 33 input and 27 output hashes. An [independent field check](../data/67-glyph-report-verification/activation-glyphs.json) verifies the six states' cell IDs, reference centers, muscle metadata, axes, dyads, shortening, centered line geometry, archive contents, and source/input/output hashes. Maximum rod-length discrepancy from the stated formula was below 5 × 10⁻¹⁶ m. The final image captions and glyph visibility were visually checked.

Run from `exp/2026/09/09/activation-space-smoothness/` using a new empty output directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
LIBGL_ALWAYS_SOFTWARE=1 \
CHERRIES_NAME='activation-glyphs-v3' \
CHERRIES_TAGS='activation,glyph,report' \
uv run python src/65-render-activation-glyphs.py \
  --output-dir data/65-activation-glyphs-v3
```

[Final Comet run](https://www.comet.com/liblaf/apple/1eb2132951414799af3ad86fea64fb82), named `activation-glyphs-v3`, completed at 12:58:50 Asia/Shanghai on September 9, 2026. Cherries recorded `.venv/bin/python3` as the resolved interpreter. Terminal output was also retained in `logs/65-render-activation-glyphs-v3.log`. The output directory is retained unchanged; use a different path to rerun. Earlier display and caption revisions remain in their own directories.

Receipt SHA-256: `c39a4d30d86ae8d3df31b78b1507b3a70ddf5de1094bdf0fde125f80feee6198`.

ParaView ZIP SHA-256: `1159396f0e5b38625ff2a51ad3d5c524d455e513659cf41d83fbf47ea1e3ca3d`.
