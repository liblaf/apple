# Pilot endpoint geometry audit

## Decision

The three-update `Region5` pilot passes the endpoint geometry validity gates:
all 1,146,517 tetrahedra retain positive `det(F)`, all activation tensors are
positive definite with determinant one, and IPC finds no self-intersections on
either the complete tetrahedral boundary or the visible `IsFace` skin at rest
or at the saved endpoint. The minimum `det(F)` is 0.745207. It belongs to an
inactive, pure-fat tetrahedron about 7.1 mm behind the lower-lip surface rather
than to a singular activation tensor.

The endpoint does not pass a convergence or fit gate. The run stopped at its
three-update budget with KKT residual 0.647, against a requested tolerance of
0.001. It moves the observed face by only 0.159180 mm RMS toward a 5.095908 mm
RMS target, leaving 4.985562 mm RMS residual and a target-projection amplitude
of 0.02191. This saved endpoint passes the geometry gates, but that does not
establish the safety of a longer optimization trajectory or make it a
successful inverse result.

## Audited state and method

The audit consumes the pinned `data/10-fixture` and any comma-separated list of
saved result directories. For each `final.vtu`, it requires topology identical
to the fixture and an exact `RestPosition` match. It then performs these CPU
checks:

1. Recompute `F = Ds Dm^-1`, `det(F)`, and all three principal stretches for
   every tetrahedron in batches. Compare the recomputed determinant with the
   saved `DetF` and report volume-weighted quantiles and the 25 lowest-`det(F)`
   cells with material, muscle, activation, position, and nearest named visible
   surface point.
2. Diagonalize every saved `ActivationInverseMatrix`, recompute its determinant,
   and report the elastic-map determinant `det(G) = det(F) det(A_inv)`.
3. Use IPC Toolkit 1.6.0 static LBVH edge-face tests on the complete tetrahedral
   boundary and the visible skin. Rest and endpoint hit sets are keyed by global
   point IDs so new and resolved intersections can be distinguished. IPC's
   `CollisionMesh` excludes edge-face pairs that share a vertex.
4. On the rest `IsFace` triangulation, form lumped vertex areas, rest vertex
   normals, and the cotangent FEM stiffness matrix. The scalar fields are
   `u dot n` and `(u - u_target) dot n`. At a named scale `l`, the low pass solves
   `(M + t K)y = Mx`, with `t = l^2 / 4`; the reported high pass is `x - y`.
   This makes `l` the RMS radius of the ideal two-dimensional heat kernel.
5. Define the mouth ROI by intrinsic mesh-edge distance no greater than 10 mm
   from any `Lip*` vertex. Report lumped-area-weighted statistics on the full
   face and this same ROI at 2, 5, and 10 mm. The 707 open membrane boundary
   edges use the natural Neumann condition.

The implementation is
[`40-audit-face-results.py`](../src/40-audit-face-results.py). The complete
machine-readable result is [`summary.json`](../data/40-audit-pilot/summary.json),
and the case-level values and culprit records are in
[`metrics.json`](../data/40-audit-pilot/20-pilot/metrics.json). The saved
`DetF` and the independent geometry recomputation agree exactly in float64 for
this endpoint.

## Volume deformation

| Field | minimum | volume q0.01% | volume q0.1% | median | volume q99.9% | maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `det(F)` | 0.745207 | 0.977767 | 0.988215 | 1.000001 | 1.010980 | 1.157302 |
| minimum principal stretch | 0.597738 | 0.918410 | 0.946040 | 0.998986 | 1.000000 | 1.001404 |
| middle principal stretch | 0.934935 | 0.982580 | 0.987992 | 1.000004 | 1.015692 | 1.095871 |
| maximum principal stretch | 0.997893 | 1.000000 | 1.000000 | 1.001039 | 1.054370 | 1.342484 |

There are zero cells below `det(F) = 0.5`, two below 0.75, three below 0.8,
and nine below 0.9. No cell is inverted. The volume-weighted 0.01% determinant
is already 0.977767, so the low minimum is a highly localized tail rather than
a volume-wide collapse.

The lowest cell is ID 691949, with principal stretches
`[0.612553, 1.055573, 1.152511]`. It is 100% fat, has no selected expression
control, and uses identity activation. Its nearest visible point is in
`LipOuterBottom`, 7.064 mm away. The second-lowest cell is another inactive,
97.7%-fat cell in the same location. The third-lowest cell has
`det(F) = 0.797730`; it is 96.7% fat by mixture but carries the left Mentalis
region control.

Of the 25 lowest-determinant cells, 24 are fat-dominant and nine carry a
selected activation control. Among the lowest 1,000 cells, 33.8% are selected
for activation, compared with 10.47% over the whole volume. This localizes the
tail to the lower-lip activation neighborhood, while the two worst cells show
that direct activation-tensor degeneration is not its cause. The current
evidence is consistent with deformation transfer across a mixed fat/muscle
neighborhood; it does not separate discretization, material contrast, and
mechanical coupling.

Every activation tensor is positive definite. Its eigenvalue range is
`[0.896664, 1.192789]`, and `det(A_inv)` lies within
`[0.9999999999999993, 1.0000000000000004]`. There are no nonpositive activation
eigenvalues. Consequently `det(G)` has the same distribution as `det(F)` to
roundoff and remains positive everywhere.

## Endpoint intersections

| Audited surface | vertices | triangles | rest hits | endpoint hits | new hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| complete tetrahedral boundary | 64,042 | 128,172 | 0 | 0 | 0 |
| visible `IsFace` skin | 15,299 | 29,899 | 0 | 0 | 0 |

The broad phase produced 42,807 candidate pairs for the complete boundary and
4,836 for the visible skin; exact IPC edge-triangle predicates rejected every
candidate. This establishes absence of endpoint self-intersection on the two
audited surfaces. It does not test continuous collision along the optimizer
trajectory, and the mechanical pilot has contact disabled, so this result does
not validate any unsaved intermediate path or contact response.

## Intrinsic surface-frequency audit

The total visible skin area is 0.0428800 m2. The intrinsic mouth ROI contains
2,926 vertices and has area 0.00648165 m2. Before filtering, scalar normal
motion is 0.088556 mm RMS on the full face and 0.208264 mm RMS in the mouth ROI.
The corresponding scalar normal residuals are 1.555554 and 2.592982 mm RMS.

The following values are area-weighted RMS followed by absolute q99 in
parentheses, in millimetres:

| scale | motion, full face | motion, mouth | residual, full face | residual, mouth |
| --- | ---: | ---: | ---: | ---: |
| 2 mm | 0.00516 (0.01805) | 0.01233 (0.04089) | 0.10856 (0.32356) | 0.27157 (0.92937) |
| 5 mm | 0.01332 (0.05264) | 0.03122 (0.10878) | 0.24901 (1.00417) | 0.60682 (2.34723) |
| 10 mm | 0.02523 (0.10561) | 0.05843 (0.15866) | 0.43322 (1.84517) | 1.00401 (3.56495) |

The actual endpoint motion is quiet at the finest scale: 99% of the full-face
area has 2 mm high-pass magnitude below 0.0181 mm, and 99% of the mouth ROI is
below 0.0409 mm. The absolute motion maxima are 0.238, 0.286, and 0.294 mm at
2, 5, and 10 mm and are localized to lip vertices. These extrema should remain
visible in longer runs, but their small area-weighted RMS does not support a
claim of a face-wide corrugation in this endpoint.

The residual high pass is much larger and follows the lips and surrounding
mouth pattern. This is expected from the failed fit: the endpoint realizes only
2.19% of the target projection, so its residual remains close to the supplied
target. The residual maps describe unresolved target structure here; they are
not evidence that the computed endpoint itself developed those wrinkles.

![Full-face high-pass maps](../data/40-audit-pilot/20-pilot/face-highpass-maps.png)

![Mouth high-pass maps](../data/40-audit-pilot/20-pilot/mouth-highpass-maps.png)

Both figures use one shared colour range per row across the three physical
scales. The rendered range is the unweighted vertex q99 for visibility; the
table above remains the area-weighted quantitative result. Reusable scalar
fields, including low-pass and high-pass motion and residual at every scale,
are stored in
[`face-diagnostics.vtp`](../data/40-audit-pilot/20-pilot/face-diagnostics.vtp).
Full-volume determinant, stretch, activation, and material fields are stored in
[`volume-diagnostics.vtu`](../data/40-audit-pilot/20-pilot/volume-diagnostics.vtu).

## Reproduction and evidence

The final audit used no GPU solve or GPU post-processing. It ran from the
experiment directory with:

```bash
CHERRIES_NAME='Audit pilot face geometry' \
CHERRIES_TAGS='face,inverse-physics,audit,cpu' \
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
uv run python src/40-audit-face-results.py
```

Comet experiment:
<https://www.comet.com/liblaf/apple/7c0f6593bcfb4fe4a2bd58c8d3ae37ab>.
The run completed in 8.08 seconds inside the measured audit body. The artifact
manifest covers eight files and was independently checked against their current
SHA-256 values and sizes. The two VTK diagnostics read back as 1,146,517
tetrahedra and 29,899 skin triangles with all expected diagnostic arrays.

To audit other saved endpoints against the same fixture, pass absolute paths or
paths relative to this experiment directory:

```bash
DEBUG=1 uv run python src/40-audit-face-results.py \
  --result-dirs data/case-a,data/case-b \
  --output-dir data/40-audit-comparison
```

The output refuses to overwrite a directory that already contains a completed
`summary.json`, which keeps each comparison auditable.
