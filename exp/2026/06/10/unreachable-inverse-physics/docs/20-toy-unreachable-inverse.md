# Toy Unreachable Inverse Physics

## Purpose

This experiment builds a controlled toy problem to test whether a nearly incompressible forward model (`nu = 0.49`) produces bumpy inverse behavior when the target displacement is not reachable by the available active muscle region.

The geometry is a thin box:

- full body: `box(0, 1, 0, 0.1, 0, 1)`
- SMAS layer: `box(0, 1, 0.04, 0.06, 0, 1)`
- muscle region: `box(0, 0.5, 0.04, 0.06, 0.4, 0.6)`

Boundary conditions fix the bottom surface and the four side surfaces. The target surface is the free interior of the top surface, excluding fixed side points. Two target modes were run:

- stretch: top target moves by `+0.02` in `y`
- squash: top target moves by `-0.02` in `y`

Material fractions per tetrahedron follow the requested split:

- muscle fraction: active muscle support
- aponeurosis fraction: `max(0, smas - muscle)`
- fat fraction: `1 - aponeurosis - muscle`

Material properties:

- muscle: `E = 1e2`, `nu = 0.49`, with activation
- aponeurosis: `E = 1e2`, `nu = 0.49`, no activation
- fat: `E = 1`, `nu = 0.49`

## Command

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="unreachable inverse toy stretch squash signed volume sweep" \
CHERRIES_TAGS="unreachable-inverse,toy,stretch,squash,resolution-sweep,nu049,signed-volume" \
uv run python src/20-toy-unreachable-inverse.py
```

Comet run: <https://www.comet.com/liblaf/apple/9cc568edab8340fabbba76355f4535ea>

## Outputs

- `data/20-toy-unreachable-inverse-summary.json`
- `data/20-toy-unreachable-inverse-cases.csv`
- `data/20-toy-unreachable-inverse-table.md`
- one `input.vtu`, `target.vtu`, inverse result `.vtu`, and `.vtu.series` for each stretch/squash and resolution case
- each inverse series has `36` frames, from step `0` through step `35`

## Results

| case | points | tets | active tets | target signed volume change | inverse signed volume change | target inverted tets | inverse inverted tets | error RMS | error / target RMS | top y std | top edge RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20-toy-stretch-coarse` | 567 | 2304 | 96 | 15.3125% | 3.0031% | 0.0000% | 0.5208% | 0.017551 | 87.7572% | 0.005518 | 0.005204 |
| `20-toy-squash-coarse` | 567 | 2304 | 96 | -15.3125% | -1.4087% | 12.7604% | 0.0000% | 0.018663 | 93.3135% | 0.003505 | 0.002644 |
| `20-toy-stretch-medium` | 2475 | 11760 | 224 | 17.2449% | 2.1948% | 0.0000% | 0.3061% | 0.018373 | 91.8658% | 0.004639 | 0.004044 |
| `20-toy-squash-medium` | 2475 | 11760 | 224 | -17.2449% | -1.8284% | 8.6224% | 0.2381% | 0.018559 | 92.7929% | 0.004613 | 0.003452 |
| `20-toy-stretch-fine` | 4851 | 24000 | 480 | 18.0500% | 3.1501% | 0.0000% | 0.1250% | 0.017966 | 89.8302% | 0.005667 | 0.004061 |
| `20-toy-squash-fine` | 4851 | 24000 | 480 | -18.0500% | -1.8727% | 9.0250% | 0.1792% | 0.018551 | 92.7528% | 0.004332 | 0.002912 |

## Interpretation

The toy targets deliberately prescribe large signed-volume changes on a body whose sides and bottom are fixed. Stretch demands a `+15%` to `+18%` signed-volume increase; squash demands a `-15%` to `-18%` signed-volume decrease. The squash target also creates local inversions in `8.6%` to `12.8%` of tetrahedra.

The inverse solutions do not reproduce those volume changes. Across resolutions, stretch reaches only about `+2.2%` to `+3.2%` signed volume change, and squash reaches only about `-1.4%` to `-1.9%`. The remaining point error is large: the final RMS error is `87.8%` to `93.3%` of the target displacement RMS. This is the expected unreachable-inverse signature.

The behavior persists from the coarse mesh through the fine mesh, so it is not just a coarse-discretization artifact. The recovered top surface is also spatially uneven: top `y` standard deviation is roughly `0.0035` to `0.0057`, and top-edge RMS roughness is roughly `0.0026` to `0.0052`. Those values are a direct quantitative proxy for the bumpy inverse surface.

The signed-volume rerun matters for squash. Absolute volume can hide orientation loss because inverted tetrahedra still contribute positive absolute volume. Signed volume shows that the squash target is strongly compressive and locally inverted, while the inverse solution stays much closer to the physically reachable branch.

## Conclusion

This toy setup reproduces the intended failure mode: with `nu = 0.49`, fixed sides/bottom, and a small active muscle patch, the target top motion is largely unreachable. The inverse optimizer responds with high residual error and spatially uneven displacement rather than a smooth exact match. That makes the high-Poisson, forward-unreachable explanation plausible for the 3152k real-mesh case, especially where the target itself contains signed-volume and local-inversion pathologies.
