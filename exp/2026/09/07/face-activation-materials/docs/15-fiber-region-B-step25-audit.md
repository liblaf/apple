# FiberRegion B step-25 audit

## Decision

Do not enlarge the activation matrix on the evidence from this run. At the
immutable step-25 checkpoint, the fit has improved from 5.095908 to 4.878577 mm
RMS, while the minimum `det(F)` has fallen to 0.257458, only 0.0575 above the
configured 0.2 floor. The minimum is a small, fat-dominant tetrahedron coupled
to a historically fixed mandible vertex. It is not a sliver, an artificial-cut
cell, a lip-contact event, or a cell with nonidentity activation.

The next controlled tests should isolate this mandible/mouth-socket coupling
and the activation direction field. A larger unconstrained parameterization
would add capacity without explaining why several region controls saturate at
29-35% equivalent shortening while most of the visible target remains
unreproduced.

## Immutable checkpoint

The audited file is
[`step-0025.vtu`](../data/21-fiber-region-B-v2/step-0025.vtu), SHA-256
`d719f71fab7e9cf263fdaa861f4057fc26d2c22c1b670b12ca0b3cfd9e907f22`, size
58,982,947 bytes. It was read while the live run continued;
the simulation directory and solver state were not modified or rerun.

The CPU audit recomputes deformation gradients from the saved coordinates and
the pinned rest fixture. Its full outputs are in
[`data/41-audit-fiber-region-B-step25`](../data/41-audit-fiber-region-B-step25/summary.json).
The checkpoint has no inverted tetrahedra and no rest or endpoint
self-intersections on either the 128,172-triangle complete boundary or the
29,899-triangle visible `IsFace` skin. The independently checked fixed-vertex
displacement is exactly zero.

## Minimum-determinant location

Cell 662949 has:

| property | value |
| --- | --- |
| `det(F)` | 0.257458 |
| principal stretches | `[0.204089, 1.019427, 1.237463]` |
| material fractions | 89.45% fat, 8.59% muscle, 1.95% aponeurosis |
| label | left Mentalis, control 9 |
| saved `A_inv` | identity; zero local active shortening |
| rest volume | 0.006557 mm3, 2.21st percentile |
| rest mean-ratio quality | 0.8182, 15.1st percentile |
| VTK scaled Jacobian | 0.5321, 7.50th percentile |
| rest edge range | 0.282-0.499 mm; ratio 1.77 |
| fixed incidence | one of four vertices, historical Mandible point |
| cut incidence | none |
| visible/loss incidence | none |
| nearest visible point | `LipOuterBottom`, 7.044 mm away |

The element is smaller and somewhat less regular than the mesh median, but it
is far from the worst rest elements. Its volume is above the 1st percentile
and its mean-ratio quality is above 15.1% of the mesh. Rest geometry can amplify
the compression, but a pre-existing sliver does not explain it.

Its fixed vertex is point 111321, an inherited `Mandible` constraint. It is not
one of the new artificial-cut constraints. The first four face-neighbor rings
around the cell contain only zero-activation Mentalis cells and inactive cells.
The nearest nonzero activation appears ten face-neighbor rings away, 1.91 mm in
centroid distance, in left Depressor labii inferioris at 32.76% equivalent
axial shortening. This is deformation transferred through mixed tissue toward
a fixed mandible attachment, rather than collapse inside the actuated cell.

The next three cells have `det(F)` 0.271603, 0.305435, and 0.377422. They are
97.7-100% fat, carry identity activation, and occupy the same mandible/gingiva
socket neighborhood. They contain one or two inherited fixed vertices. Two are
boundary cells touching gingiva, while none touches the artificial extraction
cut. This repeated pattern makes the fixed mouth-socket interface the immediate
quality limiter.

At step 25 there are four cells below `det(F)=0.5`, 24 below 0.75, 52 below
0.8, and 287 below 0.9. Half of the 52 cells below 0.8 carry a selected
expression label, and 36.7% of the lowest 1,000 do, versus 10.47% across the
mesh. The tail is concentrated near activation neighborhoods, but it extends
into passive fat and fixed-interface cells.

## Lip contact

IPC reports zero exact edge-triangle intersections on both audited surfaces.
The closest `LipTop`-to-`LipBottom` vertex distance is 0.12797 mm at rest and
0.12971 mm at step 25. The corresponding outer-lip distance changes from
0.62899 to 0.61754 mm. The lips remain extremely close in the supplied rest
geometry, but this checkpoint neither closes the smallest gap nor produces an
intersection. Lip contact is therefore not the cause of cell 662949's
compression. Contact remains disabled and these endpoint tests do not provide
continuous collision detection.

## Activation-to-surface coupling

The 35 named regions contain 40,731 mm3 of muscle-fraction-weighted volume.
Across that domain:

- 11.69% of fraction-volume lies in tetrahedra incident to at least one fixed
  vertex; 0.22% lies in all-fixed tetrahedra.
- Only 0.55% lies in tetrahedra incident to an observed `IsFace` vertex. Direct
  incidence is not required physically, but it confirms that almost all target
  motion depends on transfer through intervening mixed tissue and the skin.
- Only 6,973 mm3, or 17.12% of the selected fraction-volume, has any nonzero
  activation at step 25. For the activated subset, just 0.0096% of
  fraction-volume is incident to an observed vertex.

The largest regions remain at zero activation: occipitofrontalis (14,942 mm3),
orbicularis oris (4,620 mm3), both orbicularis oculi regions together
(7,189 mm3), and both platysma regions together (4,419 mm3). The checkpoint
instead concentrates contraction in much smaller regions:

| region pair or region | fraction-volume (mm3) | equivalent axial shortening | fixed-incident fraction-volume |
| --- | ---: | ---: | ---: |
| zygomaticus major, left/right | 444 / 400 | 35% / 35% | 21.9% / 15.0% |
| zygomaticus minor, left/right | 210 / 220 | 35% / 35% | 21.2% / 14.2% |
| risorius, left/right | 162 / 161 | 35% / 35% | 0% / 0% |
| depressor septi | 294 | 35% | 2.5% |
| depressor labii, left/right | 358 / 399 | 32.8% / 30.7% | 20.0% / 10.7% |
| levator labii, left/right | 171 / 220 | 29.8% / 28.9% | 40.5% / 23.2% |
| levator anguli oris, left/right | 196 / 193 | 29.0% / 30.4% | 15.7% / 11.8% |
| buccinator, left/right | 1,352 / 1,396 | 22.4% / 24.5% | 2.0% / 2.4% |

The high contraction values are derived from the maximum `A_inv` eigenvalue as
`1 - 1/lambda_max`, which equals axial active-map shortening for the saved
FiberRegion parameterization. Several controls are exactly at the 35% cap.

The target is largest at the lips. Area-weighted target/motion RMS is
11.34/1.29 mm on `LipOuterTop`, 10.20/1.63 mm on `LipTop`, 8.52/1.86 mm on
`LipBottom`, and 6.98/0.98 mm on `LipOuterBottom`. Per-group target projection
is only 0.073, 0.089, 0.140, and 0.082, respectively. The motion is not merely
too small; much of it is also not aligned with the supplied displacement.
Together with the rigid-alignment check, this points to activation directions,
attachment/coupling, or target/model mismatch rather than a dominant global
pose offset.

## Next controlled experiments

1. Freeze the step-25 state for diagnosis and reject any continuation that
   crosses the existing `det(F)=0.2` floor. The rapidly degrading fixed-interface
   cluster is already the limiting signal.
2. Run a forward-only counterfactual from rest with only left Depressor labii
   at its step-25 control. Record `det(F)` around cell 662949, lip-group target
   projection, and strain energy split by material. Repeat after setting that
   one control to 50% and 75% of its checkpoint value. This tests the causal
   path without changing the parameterization.
3. Repeat that counterfactual with the inherited mandible constraint policy
   bracketed: current pointwise fixation versus a reviewed attachment patch or
   weak spring. This requires a separate fixture because the current fixture is
   pinned. Compare displacement and reaction force at point 111321 and its
   neighbors; do not silently relax the production fixture.
4. Inspect and render the PCA fibers for the saturated zygomaticus, risorius,
   depressor, and levator regions against origin-to-insertion expectations.
   Their low surface projection despite cap-level shortening is a direct test
   of the geometry-derived direction prior.
5. Only after those tests, compare a few smooth scalar modes inside the same
   named regions. Keep the determinant floor, materials, target, and boundary
   bracket fixed. A larger general tensor space before these counterfactuals
   would obscure the identified attachment and direction questions.

The step-25 machine audit can be reproduced without running the solver:

```bash
DEBUG=1 uv run python src/40-audit-face-results.py \
  --result-dirs data/21-fiber-region-B-v2 \
  --endpoint-name step-0025.vtu \
  --output-dir tmp/41-audit-fiber-region-B-step25-recheck
```
