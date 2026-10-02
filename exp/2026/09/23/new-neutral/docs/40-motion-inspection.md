# Why the new neutral looks almost unchanged

The saved motion is small at true scale. Across the 15,299 displayed skin
vertices, displacement has median **0.1831 mm**, 95th percentile **0.8386 mm**,
and maximum **1.5891 mm**. **96.8%** move less than 1 mm. The face spans
198.1 mm vertically. The vertex RMS is 0.3720 mm; the solver's fitting-area
weighted RMS is 0.3340 mm, which uses a different weighting measure.

The skin vertices have exactly zero displacement in the geometric seed.
Therefore their reference-to-result and seed-to-result motion are identical.
The exported mesh was independently checked against reference plus saved
displacement, and its displacement scalar array matched the measured norms.

All bulk activation, including muscle, is identity and jaw rotation is zero.
The derived skin B replaces the previously fitted skin tension with identical
forces and tangents under this repository's physical-volume convention.
Changing this representation therefore did not add stronger actuation.
The original constitutive skin's `ActivationInv` and `SkinActivationInvDiag`
arrays are zero and `StressFreeAreaRatio` is one; no nonidentity stored skin
activation was discarded.

B acts in the tangential elastic norm, while the determinant penalty uses
physical area/volume. Consequently inverse(B) should not be interpreted as a
directly predicted stress-free shortening. Fixed bones/eyes, bonded bulk, and
contact all participate in equilibrium; their individual contributions to
the small displacement have not been isolated by an ablation.

The tailnet motion view (private preview omitted) shows front and side
panels at 1x and 10x displacement. Both color maps encode actual millimeters.
The 10x mesh is only a visualization. The saved forward endpoint still has
13 inverted tetrahedra and remains invalid. No new forward solve was run.

During this review, force labels in the main preview were corrected from raw
MPa m² values to N using a factor of 1e6. The displayed terminal force is
0.526903 N and its threshold is 0.576415 N. Raw solver data is unchanged.

## Reproduction

Working directory: `exp/2026/09/23/new-neutral`.

```bash
CHERRIES_NAME='New neutral displacement inspection' \
CHERRIES_TAGS='neutral,active-strain,review,displacement' \
OMP_NUM_THREADS=4 \
.venv/bin/python -u src/40-review-motion.py
```

The script refuses to overwrite an existing motion directory. The generated
[receipt](../data/review-active-strain-001/motion/motion-summary.json) records
input hashes, statistics, display factors, and image hashes. The source is
[40-review-motion.py](../src/40-review-motion.py).

[Comet run](https://www.comet.com/liblaf/apple/4b13c3890651420abd82e2589dfc6456)
completed with the following summary metrics:

```text
motion/max_mm                      1.5891400418592998
motion/median_mm                   0.1831159381506135
motion/p95_mm                      0.8386252847505999
motion/vertex_rms_mm                0.3719656105177831
motion/vertices_under_1mm_fraction  0.9683639453559056
```

Both front and side images were visually inspected. The renderer and motion
script passed Ruff. The updated page and its assets were checked over the
Tailscale address; see [HTTP verification](../data/http-verification-motion.json).
