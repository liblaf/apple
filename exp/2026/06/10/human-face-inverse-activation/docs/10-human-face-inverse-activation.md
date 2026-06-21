# Human Face Inverse Activation

## Scope

This experiment solves inverse activation for the `Smile` displacement field on
the human-face tetrahedral mesh:

- source mesh:
  `/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu`
- loss: point-to-point L2 on finite `Smile` vectors within `IsFace`
- fixed boundary: `IsFixed`
- optimized variables: per-muscle-tet `ActivationInv`, 6 values per active tet,
  no range clamping

The prompt also mentioned a target where the top surface moves in `+y` by
`0.1`. I did not run that as a separate inverse case in this batch because the
mesh has no top-surface label. A coordinate-derived top band is ambiguous, and
small exterior `+y` top bands have zero `IsFace` overlap, which conflicts with
the requested `IsFace` loss mask.

## Model

Volume tets use three fraction-weighted materials:

| material | E (MPa) | nu | activation |
| --- | ---: | ---: | --- |
| aponeurosis | 0.100 | 0.35 | no |
| fat | 0.003 | 0.49 | no |
| muscle | 0.030 | 0.49 | yes |

Skin is an extracted surface triangle mesh using Koiter energy with
`E = 0.200 MPa`, `nu = 0.49`, thickness `0.001`, and no prestrain.

Forward solves use PNCG with `rtol = 5e-4`, `atol = 1e-10`, and
`max_steps = 5000`. Adjoint solves use the fallback linear solver
`CupyCG -> CupyMinRes` with `rtol = 5e-4` and `atol = 0`.

## Mesh Preparation

Preparation wrote `data/10-human-face-prepared.vtu` and
`data/10-human-face-prepared-summary.json`.

| quantity | value |
| --- | ---: |
| points | 599998 |
| tetrahedra | 3190515 |
| surface points | 113786 |
| surface triangles | 227720 |
| fixed points | 42791 |
| finite Smile loss points | 15302 |
| active muscle tets | 900138 |
| activation parameters | 5400828 |
| fraction sum min/mean/max | 1.0 / 1.0 / 1.0 |
| oriented flipped tets | 0 |

## Runs

The initial 80-step `lr03` run used `lr = 0.3`. It did not satisfy the plateau
stop before the step limit, but it produced a valid warm-start state. A
continuation from that state used `lr = 0.3` and
`inverse_loss_min_delta = 2e-8`, which is small relative to a `~2e-6` loss while
still allowing the 20-step stop rule to fire in the noisy tail.

| case | stop | best step | best loss | error RMS | error / target |
| --- | --- | ---: | ---: | ---: | ---: |
| `20-human-face-smile-lr03` | `step_limit` | 80 | 2.379969e-6 | 0.00267206 | 0.5032 |
| `20-human-face-smile-lr03-cont1-d2e8` | `loss_plateau_20_steps` | 33 | 2.074415e-6 | 0.00249464 | 0.469789 |

The continuation is the completed result. It ran 34 inverse evaluations, saved
34 frames in one VTKHDF history, and the best step is the true lowest-loss step:
step 33, equal to the final step.

## Bumpiness

The continuation reduces the residual bumpiness while making the displacement
field slightly less smooth than the 80-step warm start.

| case | disp edge RMS | residual edge RMS | disp lap RMS | residual lap RMS |
| --- | ---: | ---: | ---: | ---: |
| `20-human-face-smile-lr03` | 0.000327856 | 0.000377386 | 0.000137474 | 0.000154838 |
| `20-human-face-smile-lr03-cont1-d2e8` | 0.000336670 | 0.000369736 | 0.000138517 | 0.000154641 |

Relative to the warm start, the continuation lowers residual edge RMS by about
2.0% and residual Laplacian RMS by about 0.13%. Displacement edge RMS rises by
about 2.7% and displacement Laplacian RMS by about 0.76%.

## Outputs

- implementation: `src/_human_face_inverse.py`
- preparation entrypoint: `src/10-prepare-human-face.py`
- inverse entrypoint: `src/20-inverse-human-face.py`
- plot entrypoint: `src/30-plot-loss-curves.py`
- final mesh: `data/20-human-face-smile-lr03-cont1-d2e8.vtu`
- inverse history: `data/20-human-face-smile-lr03-cont1-d2e8-steps.vtkhdf`
- final summary: `data/20-human-face-smile-lr03-cont1-d2e8-summary.json`
- aggregate table: `data/20-inverse-table.md`
- log-y loss plot:
  `figs/30-loss-curves/20-human-face-smile-lr03-cont1-d2e8-log-loss.png`

The Cherries local plugin reported post-run `log_asset` errors while copying
assets into `.cherries/runs`, but the experiment artifacts, VTKHDF history,
summary, table, and plot were all written.
