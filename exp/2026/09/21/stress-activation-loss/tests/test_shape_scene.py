"""Regression checks for static-coordinate displacement transfer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv

SOURCE = (
    Path(__file__).resolve().parents[6] / "exp/2026/09/21/stress-activation-loss/src"
)
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from shape_scene import ShapeScene, deformed_exterior  # noqa: E402


def test_deformed_exterior_uses_rest_coordinates_after_shallow_copy_mutation() -> None:
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    volume = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA]),
        rest,
    )
    exterior = volume.extract_surface(
        algorithm=None, pass_pointid=True, pass_cellid=False
    )
    point_ids = np.asarray(exterior.point_data.pop("vtkOriginalPointIds"))
    exterior.point_data["GlobalPointId"] = point_ids
    scene = ShapeScene(
        volume=volume,
        exterior_rest=exterior,
        exterior_point_ids=point_ids,
        bones={},
        eyes=pv.PolyData(),
        rest_points=rest.copy(),
    )

    first = np.full_like(rest, (0.2, -0.1, 0.05))
    second = np.full_like(rest, (-0.03, 0.07, 0.11))
    shallow_state_volume = scene.volume.copy(deep=False)
    shallow_state_volume.points = rest + first

    # VTK's shared vtkPoints means a shallow state copy mutates scene.volume.
    np.testing.assert_array_equal(scene.volume.points, rest + first)

    transferred = deformed_exterior(scene, second)
    expected = rest[point_ids] + second[point_ids]
    np.testing.assert_allclose(transferred.points, expected, rtol=0.0, atol=0.0)
