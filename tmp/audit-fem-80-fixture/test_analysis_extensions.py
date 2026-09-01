"""Isolated contract checks for the folding-analysis extensions."""

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np

SOURCE = Path(
    "/home/liblaf/Projects/liblaf/apple/exp/2026/08/31/"
    "unreachable-pork-factor-study/src/80-analyze-pork-folding.py"
)
SPEC = importlib.util.spec_from_file_location("folding_analysis", SOURCE)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


row = {
    "loss": "1.25",
    "target/rms": "0.2",
    "activation/rms": "0.3",
    "activation/neighbor_jump_rms": "0.4",
    "detG/min": "0.5",
    "detAinv/min": "0.6",
}
assert MODULE.trace_finite(row, ("objective", "loss"), "fixture") == 1.25
try:
    MODULE.trace_finite({"target/rms": "nan"}, ("target/rms",), "fixture")
except ValueError:
    pass
else:
    message = "non-finite trace metric must be rejected"
    raise AssertionError(message)

rows = []
for levels in itertools.product(("0", "1"), repeat=4):
    coded = [(-1.0 if level == "0" else 1.0) for level in levels]
    value = 3.0 + 2.0 * coded[0] + 5.0 * coded[1] * coded[3] - 7.0 * math.prod(coded)
    row = {"dimension": "2d"}
    row.update(dict(zip(MODULE.FACTORS, levels, strict=True)))
    row.update(dict.fromkeys(MODULE.METRICS, value))
    row["peak_inverted_cell_fraction"] = 0.0
    row["case_name"] = "-".join(levels)
    rows.append(row)

coefficients = {
    (item["factor_subset"], item["metric"]): item["signed_nested_contrast"]
    for item in MODULE.factorial_coefficients(rows)
}
metric = MODULE.METRICS[0]
assert coefficients[("geometry", metric)] == 4.0
assert coefficients[("muscle_extent:poisson", metric)] == 20.0
assert coefficients[(":".join(MODULE.FACTORS), metric)] == -112.0
assert len(coefficients) == (2 ** len(MODULE.FACTORS) - 1) * len(MODULE.METRICS)

plot = Path(__file__).with_name("fit-vs-jump.png")
MODULE.plot_fit_roughness(rows, plot)
assert plot.is_file()
series_plot = Path(__file__).with_name("one-panel.png")
MODULE.plot(rows, series_plot, MODULE.METRICS[0], "fixture")
assert series_plot.is_file()

points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
cells = np.array([3, 0, 1, 2, 3, 1, 3, 2])
grid = MODULE.pv.UnstructuredGrid(
    cells, np.full(2, MODULE.pv.CellType.TRIANGLE), points
)
grid.cell_data["DetF"] = np.array([-1.0, -1.0])
grid.cell_data["DetAinv"] = np.array([-1.0, 1.0])
grid.cell_data["DetG"] = np.array([1.0, -1.0])
determinants = Path(__file__).with_name("determinants.vtu")
grid.save(determinants)
folds = MODULE.fold_metrics(determinants, np.array([1.0, 3.0]))
assert (
    folds["detf_negative_detainv_negative_detg_positive_rest_measure_fraction"] == 0.25
)
assert (
    folds["detf_negative_detainv_positive_detg_negative_rest_measure_fraction"] == 0.75
)

grid.cell_data["DetG"] = np.array([-1.0, -1.0])
grid.save(determinants)
try:
    MODULE.fold_metrics(determinants, np.array([1.0, 3.0]))
except ValueError:
    pass
else:
    message = "inconsistent determinant signs must be rejected"
    raise AssertionError(message)
