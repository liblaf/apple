# Uniform 5% IsFace Skin And Volume Prestrain Rest Recovery

## Purpose

Test a uniform 5% shrink prestrain on all `IsFace` skin triangles, then estimate
and apply a compensating volume prestrain from the skin-only forward deformation.
The stored skin value is `ActivationInv_diag = 1 / (1 - 0.05) - 1 = 0.052632`,
which gives stress-free area ratio `0.95^2 = 0.9025` on active triangles.

## Command

Run directory:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-skin-prestrain
```

Command:

```bash
DEBUG=1 CHERRIES_NAME="Uniform 5pct IsFace skin-volume rest recovery" CHERRIES_TAGS="human-face,skin-prestrain,volume-prestrain,uniform-5pct,IsFace,rest-recovery" uv run python src/30-estimate-volume-prestrain.py --skin-prestrain-mode uniform --uniform-skin-prestrain 0.05 --output-stem 31-uniform-5pct-isface-volume-prestrain-rest-recovery
```

`DEBUG=1` kept the run local and avoided Comet/git-patch cleanup. The solver path
is otherwise the same two-stage recovery test as the target-derived run.

## Outputs

- `data/31-uniform-5pct-isface-volume-prestrain-rest-recovery-summary.json`
- `data/31-uniform-5pct-isface-volume-prestrain-rest-recovery.vtu`
- `data/31-uniform-5pct-isface-volume-prestrain-rest-recovery-skin.vtp`
- `data/31-uniform-5pct-isface-volume-prestrain-rest-recovery-skin-inspect.vtp`
- `data/31-uniform-5pct-isface-volume-prestrain-rest-recovery-target.vtu`

The result VTU stores `SkinOnlyDisplacement`, `SkinOnlyPoint`,
`CompensatedDisplacement`, `CompensatedPoint`, `TargetDisplacement`, and
`EstimatedActivationInvVol`. The skin inspect VTP stores `SkinActivationInvDiag`,
`StressFreeAreaRatio`, `IsActivePrestrainCell`, and `UniformSkinPrestrain`.

## Results

- Target reference field: `LipsCornersDown`.
- Simulation mesh: 228660 points, 1146517 tets.
- Active skin triangles: 29899 `IsFace` triangles.
- Skin `ActivationInv_diag`: max `0.052632`, RMS `0.025420`.
- Stress-free area ratio: min `0.9025`, max `1.0`.
- Estimated volume `ActivationInv_vol`: component range
  `[-0.815067, 1.434490]`, norm RMS `0.144120`.

Forward convergence:

| solve | steps | relative grad norm |
|---|---:|---:|
| skin only | 3814 | 4.980e-4 |
| compensated | 1635 | 4.936e-4 |

Rest-shape displacement improved, but only modestly:

| metric | skin only | compensated |
|---|---:|---:|
| free displacement RMS | 1.3814e-3 | 1.2022e-3 |
| surface displacement RMS | 1.5851e-3 | 1.3849e-3 |
| target-mask displacement RMS | 2.4073e-3 | 2.1075e-3 |

The free RMS ratio is `0.87030`, so the compensation recovers about 13% of the
skin-only rest-shape drift.

Surface area moves slightly back toward rest:

| metric | skin only | compensated |
|---|---:|---:|
| all surface area ratio | 0.973699 | 0.976278 |
| `IsFace` area ratio | 0.913890 | 0.922100 |
| active prestrain area ratio | 0.913890 | 0.922100 |

Local tet quality worsened significantly:

| metric | skin only | compensated |
|---|---:|---:|
| inverted tets | 1 | 29 |
| `detF < 0.5` tets | 50 | 113 |
| min `detF` | -0.141225 | -0.740954 |

## Analysis

The uniform 5% skin prestrain creates a much stronger global contraction than
the target-derived `LipsCornersDown` prestrain. The volume compensation again
has the right global sign: it reduces displacement RMS and partially restores
surface area. But the raw per-tet polar-stretch estimate is too aggressive
locally. It introduces many more inverted or near-collapsed tetrahedra.

This run is a useful stress test, not a usable compensation. Before using this
kind of estimate downstream, cap the derived volume stretch, drop invalid
skin-only tets from the estimate, and smooth `ActivationInv_vol` spatially.
