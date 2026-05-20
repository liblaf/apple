# 20 Activated Boxes Collision Rtol Sweep

Scene: two activated boxes from `exp/2026/05/13/toy/src/20-activated-boxes-collision.py`.

Fixed parameters:

- `E = 1`
- `nu = 0.49`
- `collision_stiffness = 0.1 * E`
- `lr = 0.05`
- `max_steps = 1500`

Outputs:

- Logs: `logs/rtol-sweep/20-activated-boxes-collision-*.log`
- VTUs: `exp/2026/05/13/toy/data/rtol-sweep/20-activated-boxes-collision-*`

`elapsed` is the timestamp of the case's `Solution(...)` log line, measured from the start of that script run. `rel_grad` is computed from the rounded tensor values printed in the logs.

| rtol | case | result | steps | elapsed | rel_grad | final_energy | final_gap | overlap | max_displacement |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `1e-3` | no collision | `PRIMARY_SUCCESS` | 237 | `9.246s` | `9.369e-04` | `0.137353` | `-0.453856` | `0.453856` | `0.487421` |
| `1e-3` | collision | `PRIMARY_SUCCESS` | 173 | `37.773s` | `1.022e-03` | `0.179569` | `0.043286` | `0` | `0.329971` |
| `1e-4` | no collision | `PRIMARY_SUCCESS` | 484 | `9.712s` | `8.517e-05` | `0.137341` | `-0.455345` | `0.455345` | `0.488127` |
| `1e-4` | collision | `MAX_STEPS_REACHED` | 1500 | `13m 33.803s` | `2.555e-04` | `0.179484` | `0.047742` | `0` | `0.328479` |
| `1e-5` | no collision | `PRIMARY_SUCCESS` | 662 | `10.324s` | `9.928e-06` | `0.137341` | `-0.455664` | `0.455664` | `0.488147` |
| `1e-5` | collision | `MAX_STEPS_REACHED` | 1500 | `10m 42.737s` | `1.703e-04` | `0.179476` | `0.046589` | `0` | `0.328545` |
| `1e-6` | no collision | `PRIMARY_SUCCESS` | 957 | `12.237s` | `9.840e-07` | `0.137341` | `-0.455775` | `0.455775` | `0.488174` |
| `1e-6` | collision | `MAX_STEPS_REACHED` | 1500 | `8m 13.670s` | `2.555e-04` | `0.179370` | `0.020612` | `0` | `0.331687` |

Total elapsed per run:

| rtol | total elapsed |
|---:|---:|
| `1e-3` | `37.810s` |
| `1e-4` | `13m 33.842s` |
| `1e-5` | `10m 42.796s` |
| `1e-6` | `8m 13.707s` |

Summary:

- No-collision cases converged for all tested tolerances.
- The collision case converged only at `rtol = 1e-3`.
- For `rtol <= 1e-4`, the collision case reached `max_steps = 1500` before satisfying the tolerance, although the final overlap stayed zero in all collision runs.
