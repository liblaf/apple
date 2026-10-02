# ruff: noqa: EM101, PT017, PT018, TRY003
"""CPU-only checks for ``vertex_blocks``; run directly, not as an experiment runner."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import scipy.sparse
import torch

_MODULE = Path(__file__).with_name("vertex_blocks.py")
_SPEC = importlib.util.spec_from_file_location("vertex_blocks", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
vertex_blocks = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vertex_blocks
_SPEC.loader.exec_module(vertex_blocks)


class StableNeoHookeanStress:
    def __init__(self, cells: torch.Tensor) -> None:
        self.cells = cells


class _DofMap:
    def __init__(self, free_indices: torch.Tensor) -> None:
        self.free_indices = free_indices


class _Model:
    def __init__(
        self,
        hessian: torch.Tensor,
        free_indices: torch.Tensor,
        *,
        cells: torch.Tensor | None = None,
        collision: object | None = None,
    ) -> None:
        self.hessian = hessian
        self.n_points = hessian.shape[0] // 3
        self.dim = 3
        self.dof_map = _DofMap(free_indices)
        self.collision = collision
        self.warp_model = SimpleNamespace(
            __wrapped__=SimpleNamespace(
                potentials={
                    "bulk": StableNeoHookeanStress(
                        torch.tensor([[0, 1, 2, 3]]) if cells is None else cells
                    )
                }
            )
        )

    def hess_prod(self, _state: object, direction: torch.Tensor) -> torch.Tensor:
        return (self.hessian @ direction.flatten()).reshape(-1, 3)


def _expected_inverse(
    block: torch.Tensor, relative: float, absolute: float
) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (block + block.T))
    floor = max(absolute, relative * float(values.abs().max()))
    return (vectors * values.clamp_min(floor).reciprocal()) @ vectors.T


def check_exact_reduced_blocks_and_regularization() -> None:
    torch.manual_seed(17)
    matrix = torch.randn(12, 12, dtype=torch.float64)
    hessian = matrix + matrix.T
    # Give the first vertex a deliberately negative direction: flooring belongs
    # only in the inverse preconditioner and must not mutate this Hessian.
    hessian[:3, :3] -= 10.0 * torch.eye(3, dtype=torch.float64)
    free = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
    model = _Model(hessian.clone(), free)
    state = SimpleNamespace(u=torch.zeros((4, 3), dtype=torch.float64))
    relative, absolute = 1.0e-6, 1.0e-9
    preconditioner = vertex_blocks.build_vertex_preconditioner(
        model,
        state,
        relative_eigenvalue_floor=relative,
        absolute_eigenvalue_floor=absolute,
    )
    probe = torch.randn(len(free), dtype=torch.float64)
    actual = preconditioner.apply(probe)
    expected = torch.empty_like(probe)
    for vertex in range(4):
        full = torch.arange(3 * vertex, 3 * vertex + 3)
        local = torch.tensor(
            [
                index
                for index, value in enumerate(free.tolist())
                if value in full.tolist()
            ]
        )
        if not len(local):
            continue
        selected = free[local]
        expected[local] = (
            _expected_inverse(hessian[selected[:, None], selected], relative, absolute)
            @ probe[local]
        )
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(model.hessian, hessian, rtol=0.0, atol=0.0)
    assert preconditioner.setup_metadata["exact_hessian_unchanged"] is True
    assert preconditioner.setup_metadata["exact_hessian_product_calls"] == 11
    assert preconditioner.setup_metadata["regularization"]["floored_modes"] > 0


def check_unsupported_potential_fails() -> None:
    hessian = torch.eye(3, dtype=torch.float64)
    model = _Model(hessian, torch.tensor([0, 1, 2]))
    model.warp_model.__wrapped__.potentials = {"bad": object()}
    state = SimpleNamespace(u=torch.zeros((1, 3), dtype=torch.float64))
    try:
        vertex_blocks.build_vertex_preconditioner(model, state)
    except TypeError as error:
        assert "unsupported potentials" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsupported potential unexpectedly accepted")


def check_shift_reuses_eigendecomposition() -> None:
    torch.manual_seed(19)
    matrix = torch.randn(12, 12, dtype=torch.float64)
    hessian = matrix + matrix.T + 5.0 * torch.eye(12, dtype=torch.float64)
    model = _Model(hessian, torch.arange(12))
    state = SimpleNamespace(u=torch.zeros((4, 3), dtype=torch.float64))
    preconditioner = vertex_blocks.build_vertex_preconditioner(model, state)
    shifted = preconditioner.with_shift(0.25)
    vector = torch.randn(12, dtype=torch.float64)
    expected = torch.empty_like(vector)
    for vertex in range(4):
        positions = torch.arange(3 * vertex, 3 * vertex + 3)
        expected[positions] = (
            _expected_inverse(
                hessian[positions[:, None], positions] + 0.25 * torch.eye(3),
                1.0e-8,
                1.0e-12,
            )
            @ vector[positions]
        )
    torch.testing.assert_close(
        shifted.apply(vector), expected, rtol=1.0e-12, atol=1.0e-12
    )
    assert (
        shifted.setup_metadata["shift"]["additional_exact_hessian_product_calls"] == 0
    )
    assert shifted.setup_metadata["shift"]["additional_eigendecompositions"] == 0


def check_ipc_graph_is_not_reused_or_omitted() -> None:
    """An IPC edge must split otherwise same-colour disconnected vertices."""
    torch.manual_seed(23)
    hessian = torch.randn(12, 12, dtype=torch.float64)
    hessian = hessian + hessian.T + 20.0 * torch.eye(12, dtype=torch.float64)
    permitted = {(0, 1), (0, 2), (1, 2), (0, 3)}
    for left in range(4):
        for right in range(left + 1, 4):
            if (left, right) not in permitted:
                hessian[3 * left : 3 * left + 3, 3 * right : 3 * right + 3] = 0
                hessian[3 * right : 3 * right + 3, 3 * left : 3 * left + 3] = 0
    local = torch.tensor([0, 1, 2, 9, 10, 11])
    ipc_hessian = scipy.sparse.csr_matrix(hessian[local[:, None], local].numpy())

    class _ContactPotential:
        def hessian(self, **_kwargs: object) -> scipy.sparse.csr_matrix:
            return ipc_hessian

    collision = SimpleNamespace(
        indices=torch.tensor([0, 3]),
        vertices=torch.zeros((2, 3), dtype=torch.float64),
        potential=_ContactPotential(),
        collision_mesh=object(),
    )
    model = _Model(
        hessian,
        torch.arange(12),
        cells=torch.tensor([[0, 1, 2]]),
        collision=collision,
    )
    state = SimpleNamespace(
        u=torch.zeros((4, 3), dtype=torch.float64),
        collision=SimpleNamespace(hess=None, collisions=object()),
    )
    preconditioner = vertex_blocks.build_vertex_preconditioner(model, state)
    vector = torch.arange(1.0, 13.0, dtype=torch.float64)
    expected = torch.empty_like(vector)
    for vertex in range(4):
        positions = torch.arange(3 * vertex, 3 * vertex + 3)
        expected[positions] = (
            _expected_inverse(hessian[positions[:, None], positions], 1.0e-8, 1.0e-12)
            @ vector[positions]
        )
    torch.testing.assert_close(preconditioner.apply(vector), expected)
    assert preconditioner.setup_metadata["ipc_graph_edges"] > 0
    assert preconditioner.setup_metadata["ipc_hessian_nnz"] == ipc_hessian.nnz


def check_tiny_actual_warp_bulk_and_membrane() -> None:
    """Exercise current experiment potential classes on Warp CPU, without adapter."""
    import numpy as np
    import pyvista as pv
    import warp as wp

    group = (
        Path(__file__).parents[4]
        / "09"
        / "21"
        / "joint-activation-material-mandible"
        / "src"
    )
    sys.path.insert(0, str(group))
    try:
        from joint_materials import StableNeoHookeanMembrane, StableNeoHookeanStress

        from liblaf.apple.common import FRACTION, GLOBAL_POINT_ID, LAMBDA, MU

        previous_dtype = torch.get_default_dtype()
        previous_device = torch.get_default_device()
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("cpu")
        wp.init()
        wp.set_device("cpu")
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

        class _ActualModel:
            n_points, dim, collision = 4, 3, None
            dof_map = _DofMap(torch.arange(12))
            warp_model = SimpleNamespace(
                __wrapped__=SimpleNamespace(potentials=potentials)
            )

            def hess_prod(
                self, _state: object, direction: torch.Tensor
            ) -> torch.Tensor:
                output = torch.zeros_like(direction)
                dtype = wp.types.vector(3, wp.dtype_from_torch(direction.dtype))
                displacement = wp.from_torch(state.u, dtype=dtype)
                vector = wp.from_torch(direction, dtype=dtype)
                result = wp.from_torch(output, dtype=dtype)
                for potential in potentials.values():
                    potential.hess_prod(displacement, vector, result)
                wp.synchronize()
                return output

        state = SimpleNamespace(u=torch.zeros((4, 3), dtype=torch.float64))
        preconditioner = vertex_blocks.build_vertex_preconditioner(
            _ActualModel(), state
        )
        assert torch.isfinite(
            preconditioner.apply(torch.ones(12, dtype=torch.float64))
        ).all()
    finally:
        sys.path.remove(str(group))
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    check_exact_reduced_blocks_and_regularization()
    check_unsupported_potential_fails()
    check_shift_reuses_eigendecomposition()
    check_ipc_graph_is_not_reused_or_omitted()
    check_tiny_actual_warp_bulk_and_membrane()
    print("vertex-block CPU checks passed")
