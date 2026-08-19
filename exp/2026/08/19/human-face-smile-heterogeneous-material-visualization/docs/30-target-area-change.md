# Smile target-area-change fields

## Purpose

This visualization shows how the prescribed `Smile` target changes the area of
each triangle on the corrected `IsFace` skin, and how that raw geometric field
is converted into the smooth driver used to construct heterogeneous skin
Young's modulus and pre-strain.

The two rows answer different questions:

1. `TargetRestAreaRatio` is the actual target/rest triangle area ratio.
2. `exp(LogAreaDiffused)` is a processed, driver-equivalent ratio after the
   material heuristic's deadband, separate expansion/contraction caps, and
   5 mm diffusion.

The second row is not target geometry, deformed geometry, a stress-free area
ratio, or a forward/inverse physics result.

## Definitions

For triangle `i`, the raw geometric ratio is

```text
r_raw[i] = TargetArea[i] / RestArea[i]
```

Thus `r_raw < 1` means target-area contraction, `r_raw = 1` means area
preservation, and `r_raw > 1` means target-area expansion. The run verified
bit-exact equality between the mapped `TargetRestAreaRatio` and
`TargetArea / RestArea` on all 29,899 corrected triangles.

The material heuristic begins with `log(r_raw)`, applies a symmetric soft
deadband and separate weighted caps, and diffuses the signed field over the
finite `IsFace` component at a 5 mm length scale. The second-row scalar is only
a ratio-like visualization of that processed log field:

```text
u[i] = LogAreaDiffused[i]
r_processed[i] = exp(u[i])
```

Positive `u` is subsequently decoded into `ExpansionWeight` for Young's-modulus
softening. Negative `u` is decoded into `ContractionSeverityLogCapped` for
pre-strain. Exponentiating `u` makes the two rows comparable on one linear
ratio scale; it does not turn the processed field back into target geometry.

## Command

Working directory:

```text
/home/liblaf/Projects/liblaf/apple/exp/2026/08/19/human-face-smile-heterogeneous-material-visualization
```

Exact command:

```bash
DEBUG=1 \
MPLBACKEND=Agg \
PYVISTA_OFF_SCREEN=true \
CHERRIES_NAME='Smile target area change visualization' \
CHERRIES_TAGS='human-face,smile,skin,target-area,visualization,debug' \
uv run --frozen python src/30-visualize-target-area-change.py
```

This was a local Cherries debug run. It created a local snapshot and log but no
remote Comet experiment. The final log entry was at 2.447 seconds. The snapshot
is:

```text
/home/liblaf/Projects/liblaf/apple/.cherries/runs/2026/08/19/human-face-smile-heterogeneous-material-visualization/30-visualize-target-area-change/2026-08-19T030451-Smile-target-area-change-visualization
```

## Outputs

- `data/30-target-rest-area-ratio.png`
  - 2200 x 1450 RGB PNG, 1,592,073 bytes
  - SHA-256:
    `95a0285d66a6d3e309876ab26d5d43fef998948d82edd0bd4152710aed36a159`
- `data/30-target-rest-area-ratio-stats.json`
  - strict JSON, 12,026 bytes
  - SHA-256:
    `437bd35158f6dbeb359123ad16dff6d8840787e5369406728db5614090399e94`
- `logs/30-visualize-target-area-change.log`
  - 968 bytes
  - SHA-256:
    `c5327529289662cd80732c8ee82c1b555aeab312f082e8308d83aa97e76b8527`
- `src/30-visualize-target-area-change.py`
  - 37,823 bytes
  - SHA-256:
    `0008a876cd670d900d3c9b2704c98d385170add9af56a43c10356c19de0fe10a`

The live script, PNG, and JSON are byte-identical to their Cherries snapshot
copies.

![Raw and processed Smile target-area-change fields](../data/30-target-rest-area-ratio.png)

## Results

Both rows use the same `RdBu_r` linear color range `[0.6, 1.4]`, centered at
the area-preserving value `1`. Blue is the contraction side and red is the
expansion side. The columns show front, 30-degree, and mouth views. No log-scale
field is rendered.

| Quantity | Raw target geometry | Processed heuristic driver |
| --- | ---: | ---: |
| Ratio definition | `TargetArea / RestArea` | `exp(LogAreaDiffused)` |
| Minimum | 0.045416 | 0.658714 |
| Maximum | 16.513034 | 1.323384 |
| Rest-area-weighted mean | 0.995904 | 0.989191 |
| Rest-area-weighted q1 | 0.639938 | 0.703264 |
| Rest-area-weighted median | 1.002322 | 1.000058 |
| Rest-area-weighted q99 | 1.370439 | 1.239818 |
| Expansion-side triangles | 16,723 | 16,770 |
| Expansion-side rest-area fraction | 54.5531% | 55.1175% |
| Contraction-side triangles | 13,159 | 13,129 |
| Contraction-side rest-area fraction | 45.4428% | 44.8825% |
| Exactly neutral triangles | 17 | 0 |

The complete corrected `IsFace` patch has rest area `0.0428799806 m^2` and
target area `0.0427043283 m^2`. Their ratio is `0.9959036302`, or a net area
change of `-0.409637%`. This near-one total does not imply small local changes:
the raw triangle ratios have a much wider range.

The common display range intentionally saturates only the colors, not the
stored values:

| Display saturation | Triangles | Rest-area fraction |
| --- | ---: | ---: |
| Raw ratio below 0.6 | 140 | 0.372784% |
| Raw ratio above 1.4 | 607 | 0.775073% |
| Raw ratio total | 747 | 1.147856% |
| Processed ratio outside `[0.6, 1.4]` | 0 | 0% |

The JSON retains every unsaturated value. The shared display range therefore
shows the bulk pattern directly while making the raw and processed rows
comparable; the exact raw extrema must be read from the table or JSON.

## Interpretation

The raw row contains sharp, triangle-scale variation, especially around the
mouth, lip boundary, nostrils, and adjacent facial folds. Its broad pattern is
contraction across much of the lateral midface and expansion around the mouth,
but sparse extreme triangles extend far beyond the display range.

The processed row removes most of that high-frequency structure and produces
coherent spatial lobes. It retains the broad contraction/expansion organization
while shifting the sign boundary slightly, as shown by the small change in
support fractions. In the planned heterogeneous-material construction, the red
processed regions provide the expansion-side softening driver, while the blue
regions provide the contraction-side pre-strain driver.

The visual comparison therefore explains why the material maps are smoother
than the raw target-area map: they are based on the deadbanded, capped, and
diffused signed field, not directly on each raw triangle ratio. It does not by
itself demonstrate that either material modification improves an inverse
solution.

## Limitations

- This run was visualization-only. It started no forward solve, adjoint, or
  inverse solve, as recorded both in the strict JSON and Cherries metrics.
- The figure cannot establish target fit, bumpiness, convergence, or
  inversion/folding behavior.
- The processed ratio is a visualization of a target-derived heuristic input.
  Its area-weighted mean is not a physical total-area ratio and must not be
  interpreted as a simulated deformation.
- The processed field depends on the selected deadband, caps, diffusion length,
  boundary conditions, and this one `Smile` target. Other choices or targets
  can produce different material regions.
- The linear display saturates 1.147856% of raw rest area, so it shows the
  location and sign but not the magnitude ordering of those extreme triangles.
- The extreme raw ratios are real values in the pinned target field, but this
  visualization alone cannot determine whether each extreme is anatomically
  meaningful or a local target/triangle artifact.
- Front and 30-degree renderings cannot expose every occluded surface detail;
  the mouth close-up is included to reduce that limitation in the main region
  of interest.

## Reproducibility

- Corrected `IsFace` skin input SHA-256:
  `4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`
- Pinned target/heuristic driver skin SHA-256:
  `ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f`
- The 29,899 corrected triangles mapped one-to-one to 29,899 unique triangles
  in the 128,172-triangle driver using sorted `GlobalPointId` keys, with exact
  readback.
- Corrected triangle-key hash (`le-i8`):
  `dca8d77662f49b54250657424cfc29a3b438437081f83d65f7e108f312da2310`
- Mapped driver-index hash (`le-i8`):
  `13458107ceef23ecc144340101574f3fb4f2e157f90a9251434e7b09a86a66c3`
- `RestArea` field hash (`le-f8`):
  `5a7b8eb9861fa509212afd610c60183f894b80db8ded53d22f3f9045bc6889de`
- `TargetArea` field hash (`le-f8`):
  `b50b815618e75ecd7b99619dc5a11492ea21dcde240dbd3a283030ac36dea580`
- Raw `TargetRestAreaRatio` field hash (`le-f8`):
  `da98a6f48694eed30b1683ccf5a1f02fd67c87393b30003d07c52a7fa25bc606`
- Mapped `LogAreaDiffused` field hash (`le-f8`):
  `df8d57c95f18f63bda06a52eb4abbcd76e86eff9b259a53d6cd15d328bd566df`
- Processed `exp(LogAreaDiffused)` field hash (`le-f8`):
  `08f1c02973f8798bbbb3950d071e1a3b1316e3ae242899881d52fc72dc1e22b5`
- Runtime: Python 3.14.6, NumPy 2.4.6, PyVista 0.48.4, with
  `uv run --frozen` at Git HEAD
  `837bbb31f9c152d412f4b72ec45a42f32df14c4a`.
- At report time the experiment group was untracked, so the input, field,
  script, and output hashes above are the authoritative run identities.
