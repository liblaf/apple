# ruff: noqa: EM101, PT017, TRY003
"""CPU checks for the frozen exact FEM Hessian product cache."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import torch
import warp as wp


def _joint_source() -> Path:
    return (
        Path(__file__).parents[4]
        / "09"
        / "21"
        / "joint-activation-material-mandible"
        / "src"
    )


sys.path.insert(0, str(_joint_source()))
_MODULE = Path(__file__).with_name("cached_fem_hvp.py")
_SPEC = importlib.util.spec_from_file_location("cached_fem_hvp", _MODULE)
assert _SPEC is not None
assert _SPEC.loader is not None
cached_fem_hvp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cached_fem_hvp
_SPEC.loader.exec_module(cached_fem_hvp)
sys.path.remove(str(_joint_source()))


def _vec3(value: torch.Tensor) -> wp.array:
    return wp.from_torch(
        value, dtype=wp.types.vector(3, wp.dtype_from_torch(value.dtype))
    )


def _make_model() -> tuple[object, SimpleNamespace, dict[str, object]]:
    source = _joint_source()
    sys.path.insert(0, str(source))
    try:
        from joint_materials import StableNeoHookeanMembrane, StableNeoHookeanStress

        from liblaf.apple.common import FRACTION, GLOBAL_POINT_ID, LAMBDA, MU

        points = np.vstack((np.zeros(3), np.eye(3))).astype(np.float64)
        bulk_mesh = pv.UnstructuredGrid(
            np.array([4, 0, 1, 2, 3]),
            np.array([pv.CellType.TETRA], dtype=np.uint8),
            points,
        )
        skin_mesh = pv.PolyData(points, np.array([3, 0, 1, 2]))
        for mesh in (bulk_mesh, skin_mesh):
            mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
            mesh.cell_data[LAMBDA.vtk] = np.array([2.3])
            mesh.cell_data[MU.vtk] = np.array([1.1])
            mesh.cell_data[FRACTION.vtk] = np.array([1.0])
        potentials = {
            "bulk": StableNeoHookeanStress.from_pyvista(bulk_mesh),
            "skin": StableNeoHookeanMembrane.from_pyvista(skin_mesh, thickness=0.001),
        }
        state = SimpleNamespace(
            u=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.07, 0.01, -0.02],
                    [0.01, -0.03, 0.04],
                    [0.0, -0.01, 0.05],
                ],
                dtype=torch.float64,
            )
        )
        model = SimpleNamespace(
            warp_model=SimpleNamespace(
                __wrapped__=SimpleNamespace(potentials=potentials)
            )
        )
        return model, state, potentials
    finally:
        sys.path.remove(str(source))


def _direct(
    potentials: dict[str, object],
    state: SimpleNamespace,
    direction: torch.Tensor,
    names: tuple[str, ...],
) -> torch.Tensor:
    result = torch.zeros_like(direction)
    for name in names:
        potentials[name].hess_prod(_vec3(state.u), _vec3(direction), _vec3(result))
    wp.synchronize()
    return result


def check_bulk_and_membrane_match_direct_hvp() -> None:
    previous_dtype, previous_device = (
        torch.get_default_dtype(),
        torch.get_default_device(),
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")
    try:
        wp.init()
        wp.set_device("cpu")
        model, state, potentials = _make_model()
        cache = cached_fem_hvp.CachedFemHvp(model, state)
        direction = torch.tensor(
            [
                [0.1, -0.2, 0.05],
                [-0.1, 0.03, 0.07],
                [0.02, 0.06, -0.04],
                [0.03, -0.05, 0.08],
            ],
            dtype=torch.float64,
        )
        cached = cache.apply(direction)
        wp.synchronize()
        expected = _direct(potentials, state, direction, ("bulk", "skin"))
        torch.testing.assert_close(cached, expected, rtol=2.0e-12, atol=2.0e-12)
        for name in ("bulk", "skin"):
            isolated_model = SimpleNamespace(
                warp_model=SimpleNamespace(
                    __wrapped__=SimpleNamespace(potentials={name: potentials[name]})
                )
            )
            isolated = cached_fem_hvp.CachedFemHvp(isolated_model, state).apply(
                direction
            )
            wp.synchronize()
            torch.testing.assert_close(
                isolated,
                _direct(potentials, state, direction, (name,)),
                rtol=2.0e-12,
                atol=2.0e-12,
            )
        assert cache.metadata["cached_bulk"] == ("F", "J", "cofactor")
        assert cache.metadata["cached_membrane"] == ("a", "b", "w", "H", "weight")
        assert cache.persistent_bytes > 0
        state.u.add_(0.001)
        try:
            cache.apply(direction)
        except RuntimeError as error:
            assert "stale" in str(error)
        else:  # pragma: no cover
            raise AssertionError("changed displacement did not invalidate cache")
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def check_unsupported_potential_fails() -> None:
    state = SimpleNamespace(u=torch.zeros((1, 3), dtype=torch.float64))
    model = SimpleNamespace(
        warp_model=SimpleNamespace(
            __wrapped__=SimpleNamespace(potentials={"bad": object()})
        )
    )
    try:
        cached_fem_hvp.CachedFemHvp(model, state)
    except TypeError as error:
        assert "unsupported FEM potentials" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsupported potential unexpectedly accepted")


if __name__ == "__main__":
    check_bulk_and_membrane_match_direct_hvp()
    check_unsupported_potential_fails()
    print("cached FEM HVP CPU checks passed")
