from typing import Any

import numpy as np
import pyvista as pv
import torch
import warp as wp


def _make_mesh(*, use_activation: bool = False) -> pv.PolyData:
    from liblaf.apple.common import (
        ACTIVATION,
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 0.1, 0.0],
            [0.2, 0.9, 0.15],
            [1.2, 1.0, 0.25],
        ],
        dtype=np.float64,
    )
    faces = np.array([3, 0, 1, 2, 3, 1, 3, 2], dtype=np.int64)
    mesh = pv.PolyData(points, faces)
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(mesh.n_points, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.array([1.7, 2.1], dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.array([0.6, 0.9], dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.array([0.75, 1.25], dtype=np.float64)
    activation = np.array(
        [
            [0.08, -0.04, 0.015],
            [-0.03, 0.05, -0.02],
        ],
        dtype=np.float64,
    )
    if use_activation:
        mesh.cell_data[ACTIVATION.vtk] = activation
    else:
        A = np.zeros((mesh.n_cells, 2, 2), dtype=np.float64)
        A[:, 0, 0] = 1.0 + activation[:, 0]
        A[:, 1, 1] = 1.0 + activation[:, 1]
        A[:, 0, 1] = A[:, 1, 0] = activation[:, 2]
        A_inv = np.linalg.inv(A)
        activation_inv = np.zeros((mesh.n_cells, 3), dtype=np.float64)
        activation_inv[:, 0] = A_inv[:, 0, 0] - 1.0
        activation_inv[:, 1] = A_inv[:, 1, 1] - 1.0
        activation_inv[:, 2] = A_inv[:, 0, 1]
        mesh.cell_data[ACTIVATION_INV.vtk] = activation_inv
    return mesh


def _from_torch_vec3(x: torch.Tensor) -> wp.array:
    floating = wp.dtype_from_torch(x.dtype)
    return wp.from_torch(x, dtype=wp.types.vector(3, floating))


def _from_torch_float(x: torch.Tensor) -> wp.array:
    return wp.from_torch(x, dtype=wp.dtype_from_torch(x.dtype))


def _fun(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.fun(_from_torch_vec3(u), _from_torch_float(output))
    wp.synchronize()
    return output[0]


def _grad(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.grad(_from_torch_vec3(u), _from_torch_vec3(output))
    wp.synchronize()
    return output


def _hess_diag(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_diag(_from_torch_vec3(u), _from_torch_vec3(output))
    wp.synchronize()
    return output


def _hess_prod(potential: Any, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_prod(
        _from_torch_vec3(u), _from_torch_vec3(p), _from_torch_vec3(output)
    )
    wp.synchronize()
    return output


def _hess_quad(potential: Any, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.hess_quad(
        _from_torch_vec3(u), _from_torch_vec3(p), _from_torch_float(output)
    )
    wp.synchronize()
    return output[0]


def test_koiter_activation_conversion_matches_activation_inv() -> None:
    from liblaf.apple.common import ACTIVATION_INV
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    direct = Koiter.from_pyvista(_make_mesh(use_activation=False), thickness=0.7)
    converted = Koiter.from_pyvista(_make_mesh(use_activation=True), thickness=0.7)

    np.testing.assert_allclose(
        wp.to_torch(converted.get_materials()[ACTIVATION_INV.value]).cpu().numpy(),
        wp.to_torch(direct.get_materials()[ACTIVATION_INV.value]).cpu().numpy(),
        rtol=1e-14,
        atol=1e-14,
    )


def test_koiter_zero_activation_has_zero_rest_energy() -> None:
    from liblaf.apple.common import ACTIVATION_INV
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    mesh = _make_mesh()
    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((mesh.n_cells, 3), dtype=np.float64)
    potential = Koiter.from_pyvista(mesh, thickness=0.7)
    u = torch.zeros((mesh.n_points, 3), dtype=torch.float64)

    assert torch.allclose(
        _fun(potential, u), torch.zeros((), dtype=u.dtype), atol=1e-14
    )
    torch.testing.assert_close(
        _grad(potential, u), torch.zeros_like(u), atol=1e-14, rtol=0.0
    )


def test_koiter_derivatives_match_finite_difference() -> None:
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    mesh = _make_mesh()
    potential = Koiter.from_pyvista(mesh, thickness=0.7)
    u = torch.tensor(
        [
            [0.03, -0.02, 0.01],
            [-0.01, 0.04, -0.02],
            [0.02, 0.01, 0.03],
            [-0.04, 0.015, 0.02],
        ],
        dtype=torch.float64,
    )
    p = torch.tensor(
        [
            [0.04, -0.03, 0.02],
            [0.01, 0.05, -0.04],
            [-0.02, 0.03, 0.01],
            [0.03, -0.01, 0.05],
        ],
        dtype=torch.float64,
    )
    eps = 1.0e-5

    grad = _grad(potential, u)
    hess_prod = _hess_prod(potential, u, p)
    hess_quad = _hess_quad(potential, u, p)
    hess_diag = _hess_diag(potential, u)

    fd_grad_dot_p = (_fun(potential, u + eps * p) - _fun(potential, u - eps * p)) / (
        2.0 * eps
    )
    torch.testing.assert_close(
        torch.sum(grad * p), fd_grad_dot_p, rtol=2e-8, atol=2e-10
    )

    fd_hess_prod = (_grad(potential, u + eps * p) - _grad(potential, u - eps * p)) / (
        2.0 * eps
    )
    torch.testing.assert_close(hess_prod, fd_hess_prod, rtol=2e-7, atol=2e-8)

    fd_hess_quad = (
        _fun(potential, u + eps * p)
        - 2.0 * _fun(potential, u)
        + _fun(potential, u - eps * p)
    ) / (eps * eps)
    torch.testing.assert_close(
        hess_quad, torch.sum(hess_prod * p), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(hess_quad, fd_hess_quad, rtol=3e-5, atol=3e-7)

    fd_diag = torch.empty_like(u)
    flat = fd_diag.reshape(-1)
    for i in range(u.numel()):
        basis = torch.zeros_like(u).reshape(-1)
        basis[i] = 1.0
        direction = basis.reshape_as(u)
        fd_grad = (
            _grad(potential, u + eps * direction)
            - _grad(potential, u - eps * direction)
        ) / (2.0 * eps)
        flat[i] = fd_grad.reshape(-1)[i]
    torch.testing.assert_close(hess_diag, fd_diag, rtol=2e-7, atol=2e-8)
