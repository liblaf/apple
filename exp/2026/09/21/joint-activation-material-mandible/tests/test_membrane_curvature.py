"""Per-triangle PNCG clamping must preserve the signed physical Hessian."""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import pyvista as pv
import torch
import warp as wp

from liblaf.apple.common import FRACTION, GLOBAL_POINT_ID, LAMBDA, MU


@pytest.fixture(scope="module")
def material_module() -> Any:
    path = Path(__file__).parents[1] / "src/joint_materials.py"
    spec = importlib.util.spec_from_file_location("curvature_test_materials", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def cpu_float64() -> Iterator[None]:
    old_dtype, old_device = torch.get_default_dtype(), torch.get_default_device()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")
    wp.init()
    try:
        with wp.ScopedDevice("cpu"):
            yield
    finally:
        torch.set_default_device(old_device)
        torch.set_default_dtype(old_dtype)


def vec3(value: torch.Tensor) -> wp.array:
    return wp.from_torch(value, dtype=wp.types.vector(3, wp.float64))


def gradient(potential: Any, displacement: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(displacement)
    potential.grad(vec3(displacement), vec3(result))
    wp.synchronize()
    return result


@pytest.mark.parametrize(
    "stresses", [(2.0, -6.0), (2.0, 4.0), (-2.0, -4.0), (0.0, 0.0)]
)
def test_clamp_each_triangle_before_summing(
    material_module: Any, stresses: tuple[float, float]
) -> None:
    # Two disconnected rest triangles have area 1/2. An out-of-plane edge
    # direction has exact q = area * S_xx for each triangle at this state.
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
        dtype=np.float64,
    )
    mesh = pv.PolyData(points, np.array([3, 0, 1, 2, 3, 3, 4, 5]))
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(6, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.full(2, 2.3)
    mesh.cell_data[MU.vtk] = np.full(2, 1.1)
    mesh.cell_data[FRACTION.vtk] = np.ones(2)
    stress = np.zeros((2, 2, 2))
    stress[:, 0, 0] = stresses
    mesh.cell_data["BaselineStress"] = stress
    potential = material_module.StableNeoHookeanMembrane.from_pyvista(
        mesh, thickness=0.001
    )
    displacement = torch.zeros((6, 3))
    direction = torch.zeros_like(displacement)
    direction[[1, 4], 2] = 1.0

    product = torch.zeros_like(displacement)
    potential.hess_prod(vec3(displacement), vec3(direction), vec3(product))
    quadratic = torch.zeros(1)
    potential.hess_quad(
        vec3(displacement), vec3(direction), wp.from_torch(quadratic, dtype=wp.float64)
    )
    wp.synchronize()

    raw_cells = 0.5 * torch.tensor(stresses)
    torch.testing.assert_close(
        (product * direction).sum(), raw_cells.sum(), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        quadratic[0], raw_cells.clamp_min(0).sum(), atol=1e-12, rtol=1e-12
    )
    if stresses == (2.0, -6.0):
        assert quadratic[0] > 0
        assert (product * direction).sum() < 0

    step = 1e-5
    finite_difference = (
        gradient(potential, displacement + step * direction)
        - gradient(potential, displacement - step * direction)
    ) / (2 * step)
    torch.testing.assert_close(product, finite_difference, atol=1e-10, rtol=1e-9)
