"""Create a disposable strict 2-D 2^4 folding fixture."""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

root = Path(sys.argv[1])
if root.exists():
    shutil.rmtree(root)
root.mkdir(parents=True)
levels = (
    ("geometry", ("slab", "tall")),
    ("muscle_extent", ("thin", "wide")),
    ("activation_sharing", ("per_cell", "shared")),
    ("poisson", (0.30, 0.49)),
)
for index, values in enumerate(itertools.product(*(pair[1] for pair in levels))):
    factors = dict(zip((pair[0] for pair in levels), values, strict=True))
    name = f"case-{index:02d}"
    folder = root / name
    folder.mkdir()
    points = np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.1, 0.0)), float)
    cells = np.array((3, 0, 1, 2))
    grid = pv.UnstructuredGrid(cells, np.array((pv.CellType.TRIANGLE,)), points)
    grid.point_data["Displacement"] = np.zeros((3, 3))
    grid.point_data["TargetDisplacement"] = np.zeros((3, 3))
    grid.cell_data["MuscleMask"] = np.array((1,), np.uint8)
    grid.cell_data["DetF"] = np.array((-0.2 if index % 2 else 0.8,))
    grid.save(folder / "final.vtu")
    initial = grid.copy()
    initial.cell_data["DetF"] = np.array((1.0,))
    initial.save(folder / "frame-000.vtu")
    grid.save(folder / "frame-001.vtu")
    (folder / "history.vtu.series").write_text(
        json.dumps(
            {
                "files": [
                    {"name": "frame-000.vtu", "time": 0.0},
                    {"name": "frame-001.vtu", "time": 1.0},
                ]
            }
        )
    )
    inverted = float(index % 2)
    negative = 0.2 if index % 2 else 0.0
    with (folder / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "inverted_cell_fraction",
                "inverted_rest_measure_fraction",
                "negative_det_f_mean",
                "step",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "inverted_cell_fraction": 0,
                "inverted_rest_measure_fraction": 0,
                "negative_det_f_mean": 0,
                "step": 0,
            }
        )
        writer.writerow(
            {
                "inverted_cell_fraction": inverted,
                "inverted_rest_measure_fraction": inverted,
                "negative_det_f_mean": negative,
                "step": 1,
            }
        )
    summary = {
        "case": {"name": name},
        "geometry": {
            "domain": [1.0, 0.1],
            "geometry_id": factors["geometry"],
            "muscle_extent_id": factors["muscle_extent"],
        },
        "materials": {
            "fat": None
            if factors["geometry"] == "tall"
            else {"nu": factors["poisson"]},
            "muscle": {"nu": factors["poisson"]},
        },
        "activation": {"sharing_id": factors["activation_sharing"]},
        "counts": {},
        "inverse": {
            "tail": {"inverse_converged_1pct_tail_gate": index % 2 == 0},
            "failures": {"forward": 0, "inverse": 0, "adjoint": 0},
        },
        "metrics": {
            "final": {
                "detF/min": -0.2 if index % 2 else 0.8,
                "target/rms": 0.01 + index,
                "activation/rms": 0.05 + index,
                "activation/neighbor_jump_rms": 0.1 + index,
            }
        },
    }
    (folder / "summary.json").write_text(json.dumps(summary))
