# Uniform 5% IsFace Boundary-Pullback Volume Prestrain Recovery

## Purpose

Test the three-step recovery idea:

1. Apply uniform 5% `IsFace` skin prestrain and run skin-only forward.
2. Use the skin-only result as the volume rest shape, fix skull plus the skin
   surface back to the original rest shape, and solve a volume-only pullback.
3. Estimate volume prestrain from the pullback deformation, then run the original
   mesh with both skin and volume prestrain.

The skin prestrain is the same as the previous uniform run:
`ActivationInv_diag = 1 / 0.95 - 1 = 0.052632`, stress-free area ratio `0.9025`.

## Command

Run directory:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/17/human-face-skin-prestrain
```

Command:

```bash
DEBUG=1 CHERRIES_NAME="Uniform 5pct IsFace boundary pullback recovery oriented" CHERRIES_TAGS="human-face,skin-prestrain,volume-prestrain,uniform-5pct,IsFace,boundary-pullback,rest-recovery,oriented-contracted" uv run python src/30-estimate-volume-prestrain.py --skin-prestrain-mode uniform --uniform-skin-prestrain 0.05 --volume-estimation-mode boundary-pullback --output-stem 32-uniform-5pct-isface-boundary-pullback-rest-recovery
```

`DEBUG=1` kept the run local. The first un-oriented contracted-rest attempt
failed during eigen-decomposition because the boundary solve produced NaNs after
`dV <= 0` warnings. The successful rerun reoriented the contracted rest mesh;
one tet was flipped.

## Outputs

- `data/32-uniform-5pct-isface-boundary-pullback-rest-recovery-summary.json`
- `data/32-uniform-5pct-isface-boundary-pullback-rest-recovery.vtu`
- `data/32-uniform-5pct-isface-boundary-pullback-rest-recovery-skin.vtp`
- `data/32-uniform-5pct-isface-boundary-pullback-rest-recovery-skin-inspect.vtp`
- `data/32-uniform-5pct-isface-boundary-pullback-rest-recovery-target.vtu`

The result VTU stores `SkinOnlyPoint`, `BoundaryPullbackPoint`,
`BoundaryPullbackTotalDisplacement`, `CompensatedPoint`, and
`EstimatedActivationInvVol`.

## Results

- Active skin triangles: 29899 `IsFace` triangles.
- Boundary-pullback fixed points: 64042 surface/skull points.
- Boundary-pullback solve converged: 193 steps, relative grad norm `4.541e-4`.
- Final compensated solve did not converge: 5000 steps, relative grad norm
  `1.270e-1`.
- Estimated volume `ActivationInv_vol`: component range
  `[-759.589, 1600.270]`, norm RMS `2.6653`.

Displacement improved more than the direct estimate, but the solve is not valid:

| metric | skin only | boundary pullback | compensated |
|---|---:|---:|---:|
| non-skull free displacement RMS | 1.3814e-3 | 5.4838e-5 | 9.6780e-4 |
| non-boundary interior RMS | 1.1661e-3 | 6.0689e-5 | 7.8507e-4 |
| surface/boundary RMS | 1.5850e-3 | ~0 | 1.1681e-3 |
| target-mask displacement RMS | 2.4074e-3 | n/a | 1.9079e-3 |
| `IsFace` area ratio | 0.913889 | 1.000000 | 0.923414 |

Local tet quality is the blocker:

| metric | skin only | boundary pullback | compensated |
|---|---:|---:|---:|
| inverted tets | 1 | 149 | 277 |
| `detF < 0.5` tets | 50 | 458 | 947 |
| min `detF` | -0.195268 | -20.717782 | -15.549032 |
| max `detF` | 1.644237 | 9.938329 | 6.245296 |

## Analysis

This method enforces the boundary target very strongly in step 2, so the
boundary-pullback shape is nearly the original rest shape on the surface. But
the interior deformation required to satisfy that boundary from the contracted
rest shape is highly singular. Turning that pullback directly into
`ActivationInv_vol` gives enormous per-tet values and a final compensated solve
that fails to converge.

As a recovery estimator, this is not usable without regularization. The useful
piece is conceptual: the boundary-pullback solve can define a better constrained
target for the volume, but the derived volume prestrain must be capped,
smoothed, and probably solved as an optimization with a distortion penalty
instead of directly copying per-tet polar stretch.
