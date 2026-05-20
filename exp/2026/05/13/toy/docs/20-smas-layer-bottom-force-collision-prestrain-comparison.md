# SMAS Prestrain Comparison

Experiment: `20-smas-layer-bottom-force-collision`

## Common Setup

| quantity | value |
|---|---:|
| body bounds | `(0, 1, 0, 0.1, 0, 1)` |
| SMAS layer bounds | `(0, 1, 0.04, 0.06, 0, 1)` |
| `E_fat` | `1.0` |
| `E_smas / E_fat` | `100` |
| `E_smas` | `100.0` |
| `nu` | `0.49` |
| TetWild `lr` | `0.02` |
| bottom pressure | `0.3` |
| collision stiffness | `0.1` |
| optimizer max steps | `3000` |
| optimizer `rtol_primary` | `5e-4` |
| optimizer `rtol_secondary` | `5e-4` |
| mesh points | `9699` |
| mesh cells | `45769` |
| mesh generation wall time | `10.609466 s` |

Wall times were measured with `time.perf_counter()` in one paired run. Both cases used the same generated tetrahedral mesh.

## Saved Meshes

| case | solution mesh |
|---|---|
| with prestrain | `exp/2026/05/13/toy/data/20-smas-layer-bottom-force-collision-with-prestrain.vtu` |
| without prestrain | `exp/2026/05/13/toy/data/20-smas-layer-bottom-force-collision-without-prestrain.vtu` |

## Displacement Summary

| case | SMAS prestrain | SMAS activation | result | steps | bottom max `u_y` | bottom mean `u_y` | max displacement |
|---|---|---|---:|---:|---:|---:|---:|
| with prestrain | `(0.8, 1, 0.8, 0, 0, 0)` | `(-0.2, 0, -0.2, 0, 0, 0)` | `PRIMARY_SUCCESS` | `302` | `0.0923113` | `0.0524899` | `0.0923158` |
| without prestrain | `(1, 1, 1, 0, 0, 0)` | `(0, 0, 0, 0, 0, 0)` | `PRIMARY_SUCCESS` | `1807` | `0.262284` | `0.132438` | `0.262286` |

Removing prestrain increases bottom max `u_y` by about `2.84x` and bottom mean `u_y` by about `2.52x` under the same load.

## Performance

| case | solve wall time | save wall time |
|---|---:|---:|
| with prestrain | `43.295485 s` | `0.106771 s` |
| without prestrain | `187.185539 s` | `0.088319 s` |

The solve wall time includes forward model construction, initial/final energy evaluation, optimization, and result mesh construction. It does not include the shared mesh generation time or the VTU save time.
