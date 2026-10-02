"""CPU behavior check for registry-only FEM rest-edge scaling."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

source = Path(__file__).with_name("mesh_step_scale.py")
spec = importlib.util.spec_from_file_location("mesh_step_scale", source)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def model(*cells: np.ndarray):
    potentials = {
        f"p{index}": SimpleNamespace(cells=value) for index, value in enumerate(cells)
    }
    return SimpleNamespace(
        warp_model=SimpleNamespace(__wrapped__=SimpleNamespace(potentials=potentials))
    )


def main() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [
                1000.0,
                1000.0,
                1000.0,
            ],  # obstacle-like unused point: must not affect scale
        ]
    )
    tet = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    tri = np.array([[0, 1, 2]], dtype=np.int64)  # repeats tet edges; unique edges only
    expected = (1 + 2 + 3 + np.sqrt(5) + np.sqrt(10) + np.sqrt(13)) / 6
    actual = module.mean_rest_edge_length(model(tet, tri), points)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    try:
        module.mean_rest_edge_length(model(np.array([[0, 1]], dtype=np.int64)), points)
    except TypeError:
        pass
    else:
        message = "non-tri/tet potential must fail visibly"
        raise AssertionError(message)
    print("mesh_step_scale CPU checks passed")


if __name__ == "__main__":
    main()
