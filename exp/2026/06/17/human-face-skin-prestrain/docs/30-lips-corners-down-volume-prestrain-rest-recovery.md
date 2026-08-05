# LipsCornersDown Skin And Volume Prestrain Rest Recovery

## Purpose

Test whether the `LipsCornersDown` stretch-only `IsFace` skin prestrain can be
balanced by an estimated volume prestrain. The volume estimate is derived from
the skin-only forward deformation: compute each tet polar stretch `U`, store
`ActivationInv_vol = U - I`, and interpret the implied volume prestrain as
`U^{-1}`.

## Command

Run directory:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-skin-prestrain
```

Command:

```bash
CHERRIES_NAME="LipsCornersDown skin-volume rest recovery corrected" CHERRIES_TAGS="human-face,skin-prestrain,volume-prestrain,LipsCornersDown,rest-recovery,corrected-activation-inv" uv run python src/30-estimate-volume-prestrain.py
```

Comet printed:

```text
https://www.comet.com/liblaf/apple/81c515d6a0c64f9587db9e2abfe47c14
```

The solve outputs were written before Cherries/Comet hit the known local asset
copy and git-patch logging warnings.

## Outputs

- `data/30-lips-corners-down-volume-prestrain-rest-recovery-summary.json`
- `data/30-lips-corners-down-volume-prestrain-rest-recovery.vtu`
- `data/30-lips-corners-down-volume-prestrain-rest-recovery-skin.vtp`
- `data/30-lips-corners-down-volume-prestrain-rest-recovery-skin-inspect.vtp`
- `data/30-lips-corners-down-volume-prestrain-rest-recovery-target.vtu`
- `logs/30-estimate-volume-prestrain.log`

The result VTU stores `SkinOnlyDisplacement`, `SkinOnlyPoint`,
`CompensatedDisplacement`, `CompensatedPoint`, `TargetDisplacement`, and
`EstimatedActivationInvVol`. The skin inspect VTP stores the same surface
displacements plus `SkinActivationInvDiag`, `StressFreeAreaRatio`,
`IsFacePrestrainCell`, and `IsStretchedPrestrainCell`.

## Results

- Target: `LipsCornersDown` from
  `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu`.
- Simulation mesh: 228660 points, 1146517 tets.
- Skin prestrain: 7952 stretched `IsFace` triangles, max diagonal
  `ActivationInv = 1.719229`, RMS `0.017549`.
- Estimated volume `ActivationInv_vol`: component range
  `[-0.695095, 1.149319]`, norm RMS `0.036679`.
- Skin-only forward converged: 1116 steps, relative grad norm `4.996e-4`.
- Compensated forward converged: 1043 steps, relative grad norm `4.962e-4`.

Rest-shape displacement improved modestly:

| metric | skin only | compensated |
|---|---:|---:|
| free displacement RMS | 2.5835e-4 | 2.2186e-4 |
| surface displacement RMS | 3.0677e-4 | 2.6585e-4 |
| target-mask displacement RMS | 4.3319e-4 | 3.8539e-4 |

The free RMS ratio is `0.85876`, so the estimated volume prestrain recovers
about 14% of the skin-only rest-shape drift.

Surface area did not move much back toward the rest area:

| metric | skin only | compensated |
|---|---:|---:|
| `IsFace` area ratio | 0.988566 | 0.989211 |
| active prestrain triangle area ratio | 0.951060 | 0.952263 |

Local tet quality worsened:

| metric | skin only | compensated |
|---|---:|---:|
| inverted tets | 1 | 3 |
| `detF < 0.5` tets | 25 | 47 |
| min `detF` | -0.096376 | -0.568362 |

## Analysis

The convention is easy to get backward. The volume kernel minimizes energy near
`F @ ActivationInv = I`; therefore if the actual volume prestrain should be the
inverse of the skin-only deformation, the stored `ActivationInv_vol` should be
the skin-only polar stretch `U`, not `U^{-1}`.

With that corrected convention, compensation helps displacement RMS but not
enough to call this a clean recovery of the rest shape. The estimate also
amplifies local bad elements, probably because the per-tet inverse estimate is
unsmoothed and inherits the skin-only deformation outliers.

## Next Step

Keep this as evidence that the idea has the right global direction, but add
regularization before using it as a production prestrain estimate: cap the
derived tet stretch, ignore inverted/near-singular skin-only tets, and smooth
`ActivationInv_vol` over neighboring tets.
