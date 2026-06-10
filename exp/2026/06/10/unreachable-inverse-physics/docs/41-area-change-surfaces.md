# Area Change Surface Artifacts

## Purpose

This run creates inspectable per-triangle surface-area change artifacts for the real human-face cases and the toy-geometry cases. The earlier diagnostics already computed aggregate area changes, and this pass writes `.vtp` surfaces with cell arrays so the area-change distribution can be inspected directly in ParaView.

The run covers:

- `3152k-expression001`
- `515k-nosmas`
- the six primary `20-toy-*` stretch/squash inverse cases
- the three `40-toy-forward-activation-*` forward cases
- the sixteen `30-toy-*` target-magnitude and Poisson-ratio sweep cases

## Command

Working directory:

```bash
/home/liblaf/github/liblaf/apple/exp/2026/06/10/unreachable-inverse-physics
```

Command:

```bash
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_GIT_METADATA=false \
CHERRIES_NAME="area change surface artifacts" \
CHERRIES_TAGS="unreachable-inverse,area-change,surface-vtp,real-mesh,toy" \
uv run python src/41-area-change-surfaces.py
```

Comet run: <https://www.comet.com/liblaf/apple/676d536b762e40fc86b365c038e38f7e>

Cherries summary fields from the run log:

- `cherries/entrypoint`: `exp/2026/06/10/unreachable-inverse-physics/src/41-area-change-surfaces.py`
- `cherries/exp_dir`: `exp/2026/06/10/unreachable-inverse-physics`
- `cherries/git/sha`: `80edb7c84287b045a5bfb15a8bfac892fddaaec9`
- `cherries/start_time`: `2026-06-10 15:22:07.339316+08:00`
- `cherries/end_time`: `2026-06-10 15:22:12.960552+08:00`

## Outputs

- `data/41-area-change-surfaces-summary.json`
- `data/41-area-change-surfaces-cases.csv`
- `data/41-area-change-surfaces-table.md`
- `data/41-area-change-surfaces/*.vtp`
- `logs/41-area-change-surfaces.log`

Each `.vtp` stores the extracted triangulated surface with these core cell arrays:

- `RestArea`
- `TargetValidPointCount`
- `TargetValidTriangleAny`
- `TargetValidTriangleAll`
- `TargetRawArea`, `TargetRawAreaDelta`, `TargetRawAreaRelChange`
- `TargetMaskedArea`, `TargetMaskedAreaDelta`, `TargetMaskedAreaRelChange`
- `SolutionArea`, `SolutionAreaDelta`, `SolutionAreaRelChange`

For real human-face cases, the target mask is `IsFace`. For toy cases, the target mask is `TargetSurfaceMask`.

## Results

Primary real-mesh rows:

| case | state | triangles | mask triangles | surface area change | mask area change | mask p50 | mask p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3152k-expression001` | target-masked diagnostic | 126402 | 34372 | 0.3057% | -0.3666% | -0.2947% | 9.4788% |
| `3152k-expression001` | solution | 126402 | 34372 | 0.6933% | 0.8127% | -0.1436% | 14.6104% |
| `515k-nosmas` | target | 59068 | 13079 | 0.0186% | 0.0474% | -0.0015% | 3.5244% |
| `515k-nosmas` | solution | 59068 | 13079 | 0.0120% | 0.0556% | -0.0010% | 3.0200% |

Primary toy inverse rows:

| case | solution surface area change | solution target-area change | target-area p50 | target-area p95 |
| --- | ---: | ---: | ---: | ---: |
| `20-toy-stretch-coarse` | 0.0794% | 1.0167% | 0.6229% | 5.2993% |
| `20-toy-stretch-medium` | 0.1950% | 0.7625% | -0.5943% | 7.9368% |
| `20-toy-stretch-fine` | 0.3546% | 1.0604% | -0.9167% | 12.2693% |
| `20-toy-squash-coarse` | 0.0405% | 0.1853% | 0.6298% | 2.9239% |
| `20-toy-squash-medium` | 0.1192% | 0.2651% | 0.4550% | 2.7630% |
| `20-toy-squash-fine` | 0.1940% | 0.4009% | 0.4884% | 3.2143% |

Forward activation rows:

| case | solution surface area change | solution target-area change | target-area p50 | target-area p95 |
| --- | ---: | ---: | ---: | ---: |
| `40-toy-forward-activation-coarse` | 0.1121% | -0.8641% | 2.8039% | 13.4438% |
| `40-toy-forward-activation-medium` | 0.3328% | 0.7053% | 2.6902% | 11.5066% |
| `40-toy-forward-activation-fine` | 0.1489% | 0.1150% | 3.0784% | 10.5400% |

## Analysis

The `3152k-expression001` target and solution differ in the `IsFace` triangle-area distribution. The target-masked diagnostic has aggregate `IsFace` area change `-0.3666%`, while the inverse solution has `+0.8127%`. The p95 `IsFace` triangle-area change rises from `9.4788%` in the masked target diagnostic to `14.6104%` in the solution. This gives a concrete surface metric to inspect alongside the volume fields when looking for bumpy regions.

The 515k no-SMAS case remains much closer: target and solution mask-area changes are both near `0.05%`, and p95 area changes are roughly `3%`. This matches the earlier volume conclusion that the 515k no-SMAS forward target does not show the same strong global area/volume mismatch as the 3152k expression target.

For toy stretch/squash, the prescribed uniform top displacement leaves the top target triangles unchanged in area, but the inverse solution does not. The recovered top-area distributions are nonuniform, especially for stretch. The fine stretch solution has target-area p95 `12.2693%`, while the fine squash p95 is `3.2143%`. This is another quantitative signature of the bumpy inverse response.

The forward activation surfaces are useful as a baseline: they show that a reachable forward contraction from the small muscle patch also produces heterogeneous surface area changes. The difference is that these rows are valid forward equilibrium states, while the inverse stretch/squash rows are failed attempts to match a prescribed top motion with much larger volume demands.

## Reproducibility

The script is CPU-only and post-processes already generated VTU artifacts. It uses PyVista surface extraction with `vtkOriginalPointIds`, then computes triangle areas from rest points and displaced points. `uv run python -m py_compile` and `uv run ruff check` passed for the script before the report-worthy run.
