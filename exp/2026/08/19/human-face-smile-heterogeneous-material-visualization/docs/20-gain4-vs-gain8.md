# Gain-4 versus gain-8 skin Young's modulus

## Purpose

This visualization compares the proposed gain-4 heterogeneous skin Young's
modulus against the earlier gain-8 field on the corrected `IsFace` membrane.
Both cases use the same expansion-weight driver, nominal range, Poisson's ratio,
thickness, and plane-stress Lamé conversion:

```text
s = clip(gain * ExpansionWeight, 0, 1)
E = 0.2 * exp(log(0.0003 / 0.2) * s) MPa
lambda = E * nu / (1 - nu^2), mu = E / (2 * (1 + nu))
nu = 0.49, thickness = 0.001 m
```

The primary figure uses one shared **linear** color range of `[0, 0.2] MPa` so
gain-4 and gain-8 can be compared without the low-modulus differences being
amplified by a logarithmic scale.

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
CHERRIES_NAME='Gain-4 vs gain-8 linear Young modulus visualization' \
CHERRIES_TAGS='human-face,skin,material,heterogeneous,'\
'gain4,gain8,visualization,debug' \
uv run --frozen python src/20-visualize-gain4-comparison.py
```

This was a local Cherries debug run, so it produced a local snapshot and log
without a remote Comet experiment. The log completed at 2.859 seconds.

## Outputs

- `data/20-gain4-vs-gain8-linear-young-modulus.png`
  - 2200 x 1450 RGB PNG, 1,299,467 bytes
  - SHA-256:
    `77c175e7c0a75cb40379de38fd6d880d23d27de61df132e9d6800fa47af0ac9d`
- `data/20-gain4-vs-gain8-stats.json`
  - strict JSON, 9,736 bytes
  - SHA-256:
    `de9f7d73df359faad11b38c4160f11c7af2bfab1b8b8c5d263121e9e5b8bd980`
- `logs/20-visualize-gain4-comparison.log`
  - SHA-256:
    `b89848e437be2a6f779685fea15abe33c35ca2850dc5d3ef0d942549d83dfe0f`

![Gain-4 versus gain-8 on a shared linear Young's-modulus scale](../data/20-gain4-vs-gain8-linear-young-modulus.png)

## Results

| Quantity | gain-4 | gain-8 |
| --- | ---: | ---: |
| Nominal modulus range (MPa) | 0.0003--0.2 | 0.0003--0.2 |
| Area-weighted mean E (MPa) | 0.141652 | 0.131075 |
| Area at the minimum E | 3.827% | 8.737% |
| Area below fat E = 0.003 MPa | 6.004% | 18.096% |
| Interior-edge E-jump RMS (MPa) | 0.010929 | 0.013305 |
| Interior-edge E-jump q99 (MPa) | 0.050002 | 0.061785 |
| Maximum interior-edge E jump (MPa) | 0.180721 | 0.196539 |

Relative to gain-8, gain-4 raises the area-weighted mean by `0.010577 MPa`,
reduces minimum-modulus coverage by `4.910` percentage points, and reduces the
area softer than fat by `12.091` percentage points. It also reduces all three
reported interior-edge jump measures.

On the shared linear scale, both cases retain the same anatomical softening
pattern around the forehead, nose, cheeks, mouth, and lower face. Gain-4 shows
more intermediate green/blue transition and a smaller dark minimum-modulus
plateau, especially around the mouth and lower face. Gain-8 expands the dark
soft regions and changes more abruptly into the yellow baseline. The numerical
edge-jump statistics agree with that visual reading, so gain-4 is the less sharp
candidate while retaining strong localized softening.

## Limitations

This run only constructed and rendered prescribed material fields. It started
no forward solve, adjoint, or inverse solve. Therefore it does not show whether
gain-4 improves target fit, reduces bumpiness, changes convergence, or avoids
inversion/folding. Those questions require at least a forward probe and, only if
that probe is acceptable, an inverse run. The figure also isolates Young's
modulus; it does not visualize or change p200 pre-strain.

## Reproducibility

- Corrected `IsFace` skin input:
  `4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`
- Expansion-driver skin input:
  `ffd586e8e1625facc89e87be803fbac16b374ae3d64b34916fd280cd05104c5f`
- The 29,899 corrected triangles mapped one-to-one to 29,899 unique driver
  triangles using sorted `GlobalPointId` keys, with exact readback.
- Gain-4 Young's-modulus field hash:
  `84c15003461f3f69e212be9e60d2e99dc8747072f963a48398ab3fb9d16ecc9c`
- Gain-8 Young's-modulus field hash:
  `d42fa66ea54c47890e184fd8b670a7e80cd5621e337ee81b03f658a8b9821a44`
- Script SHA-256:
  `4f481e85aba5a602bda1d657e60f4dcdf27be281ae17cadd16f4be481beaea02`;
  the local Cherries snapshot is byte-identical to the live script.
- Runtime: Python 3.14.6, NumPy 2.4.6, PyVista 0.48.4, with
  `uv run --frozen` at Git HEAD
  `837bbb31f9c152d412f4b72ec45a42f32df14c4a`.
- At report time the experiment group was untracked, so the input, field, script,
  and output hashes above are the authoritative run identities.
