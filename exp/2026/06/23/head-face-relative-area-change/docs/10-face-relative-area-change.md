# Head IsFace Relative Area Change

## Purpose

Compute the per-triangle minimum and maximum relative area change across all
expression blendshapes stored in:

```text
/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu
```

For each selected surface triangle and expression, the script evaluates:

```text
relative_area_change = area(points + expression_displacement) / area(points) - 1
```

The selected face set is the triangulated exterior surface where all three
original mesh point ids have `IsFace=True`. This matches the all-vertices mask
rule used by the existing human-face inverse workflow for surface triangles.

## Command

Normal Cherries/Comet run:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/23/head-face-relative-area-change
CHERRIES_NAME="Head IsFace relative area change" \
CHERRIES_TAGS="head,IsFace,area-change,blendshape" \
uv run python src/10-compute-face-relative-area-change.py
```

The normal run wrote all data outputs and logged metrics, but Comet finalization
was interrupted after about 3 minutes while it attempted to package a very large
dirty git patch containing unrelated experiment artifacts.

Clean local verification run:

```bash
cd /home/liblaf/github/liblaf/apple/exp/2026/06/23/head-face-relative-area-change
DEBUG=1 \
CHERRIES_NAME="Head IsFace relative area change" \
CHERRIES_TAGS="head,IsFace,area-change,blendshape" \
uv run python src/10-compute-face-relative-area-change.py
```

This debug run exited with code 0 and regenerated the same outputs.

## Summary

- Input mesh: 599,998 points, 3,190,515 tetrahedra.
- Extracted surface: 113,786 points, 227,720 triangles.
- Selected `IsFace` surface triangles: 29,899.
- Expression arrays used: 36 numeric point-data arrays with shape
  `(n_points, 3)`.
- Base selected face area sum: `0.04287998059707303`.
- Global minimum relative area change: `-0.9545843344570981` on triangle
  `7813`, expression `Smile`.
- Global maximum relative area change: `34.32827593391882` on triangle `7678`,
  expression `Scream`.

## Outputs

- `data/10-face-relative-area-change.csv`: one row per selected face triangle,
  including point ids, base area, min/max relative area change, and the
  expression names that attained those extrema.
- `data/10-face-relative-area-change.npz`: numeric arrays for downstream
  analysis, including all expression-by-triangle relative changes.
- `data/10-face-relative-area-change.vtp`: selected surface triangles for
  ParaView, with cell arrays `BaseArea`, `MinRelativeAreaChange`,
  `MaxRelativeAreaChange`, `MinExpressionIndex`, and `MaxExpressionIndex`.
- `data/10-face-relative-area-change-summary.json`: run summary and global
  extrema.
- `data/10-expression-area-change-summary.csv`: per-expression min, mean, and
  max relative area change over the selected face triangles.
- `logs/10-compute-face-relative-area-change.log`: Cherries run log.

## Verification

The following checks passed after the clean debug run:

- `ruff check` passed on the experiment script.
- Python bytecode compilation passed.
- `relative_changes.min(axis=0)` equals `min_relative_area_change`.
- `relative_changes.max(axis=0)` equals `max_relative_area_change`.
- `relative_changes.argmin(axis=0)` equals `min_expression_index`.
- `relative_changes.argmax(axis=0)` equals `max_expression_index`.
- All stored relative changes and extrema are finite.
- All selected base triangle areas are positive.
- The VTP output loads as `PolyData` with 29,899 selected cells and the expected
  cell-data arrays.

## Notes

The `.vtp` stores expression extrema as integer expression indices because VTK
cell-data string arrays are less convenient for ParaView and downstream numeric
processing. The CSV and summary JSON provide the corresponding expression names.
