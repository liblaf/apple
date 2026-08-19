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


def _make_unit_triangle_mesh(
    *,
    lambda_: float,
    mu: float,
    fraction: float = 1.0,
    activation_inv: tuple[float, float, float] = (0.0, 0.0, 0.0),
    global_point_ids: tuple[int, int, int] = (0, 1, 2),
) -> pv.PolyData:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    mesh = pv.PolyData(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        np.array([3, 0, 1, 2], dtype=np.int64),
    )
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.asarray(global_point_ids, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.array([lambda_], dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.array([mu], dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.array([fraction], dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.asarray([activation_inv], dtype=np.float64)
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


def test_koiter_homogeneous_patch_matches_analytic_membrane_energy() -> None:
    from liblaf.apple import common
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    lambda_t, mu_t = common.lame_converter_plane_stress(0.2, 0.49)
    lambda_ = float(lambda_t)
    mu = float(mu_t)
    thickness = 0.001
    fraction = 0.7
    stretch_x = 1.08
    stretch_y = 0.96
    mesh = _make_unit_triangle_mesh(lambda_=lambda_, mu=mu, fraction=fraction)
    potential = Koiter.from_pyvista(mesh, thickness=thickness)
    u = torch.tensor(
        [[0.0, 0.0, 0.0], [stretch_x - 1.0, 0.0, 0.0], [0.0, stretch_y - 1.0, 0.0]],
        dtype=torch.float64,
    )

    m_x = stretch_x**2 - 1.0
    m_y = stretch_y**2 - 1.0
    density = 0.5 * lambda_ * (m_x + m_y) ** 2 + mu * (m_x**2 + m_y**2)
    expected = thickness * fraction * density / 8.0

    torch.testing.assert_close(
        _fun(potential, u),
        torch.tensor(expected, dtype=u.dtype),
        rtol=1.0e-7,
        atol=1.0e-15,
    )


def test_koiter_prestrain_patch_has_zero_energy_at_natural_metric() -> None:
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    inverse_stretch = 1.3
    mesh = _make_unit_triangle_mesh(
        lambda_=0.4,
        mu=0.2,
        activation_inv=(inverse_stretch - 1.0, inverse_stretch - 1.0, 0.0),
    )
    potential = Koiter.from_pyvista(mesh, thickness=0.001)
    natural_stretch = 1.0 / inverse_stretch
    u = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [natural_stretch - 1.0, 0.0, 0.0],
            [0.0, natural_stretch - 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        _fun(potential, u), torch.zeros((), dtype=u.dtype), rtol=0.0, atol=1.0e-15
    )
    torch.testing.assert_close(
        _grad(potential, u), torch.zeros_like(u), rtol=0.0, atol=1.0e-14
    )


def test_koiter_prestrain_keeps_original_reference_area_weight() -> None:
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    inverse_stretch = 1.3
    elastic_stretch = 1.08
    thickness = 0.001
    baseline = Koiter.from_pyvista(
        _make_unit_triangle_mesh(lambda_=0.4, mu=0.2), thickness=thickness
    )
    prestrained = Koiter.from_pyvista(
        _make_unit_triangle_mesh(
            lambda_=0.4,
            mu=0.2,
            activation_inv=(
                inverse_stretch - 1.0,
                inverse_stretch - 1.0,
                0.0,
            ),
        ),
        thickness=thickness,
    )
    baseline_u = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [elastic_stretch - 1.0, 0.0, 0.0],
            [0.0, elastic_stretch - 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    prestrained_stretch = elastic_stretch / inverse_stretch
    prestrained_u = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [prestrained_stretch - 1.0, 0.0, 0.0],
            [0.0, prestrained_stretch - 1.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        _fun(prestrained, prestrained_u),
        _fun(baseline, baseline_u),
        rtol=1.0e-12,
        atol=1.0e-15,
    )


def test_koiter_filtered_surface_uses_global_point_ids() -> None:
    from liblaf.apple.warp.fem import Koiter

    torch.set_default_dtype(torch.float64)
    wp.init()

    local_mesh = _make_unit_triangle_mesh(lambda_=0.4, mu=0.2)
    mapped_mesh = _make_unit_triangle_mesh(
        lambda_=0.4, mu=0.2, global_point_ids=(2, 5, 8)
    )
    local = Koiter.from_pyvista(local_mesh, thickness=0.001)
    mapped = Koiter.from_pyvista(mapped_mesh, thickness=0.001)
    local_u = torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, -0.04, 0.0]],
        dtype=torch.float64,
    )
    mapped_u = torch.zeros((9, 3), dtype=torch.float64)
    mapped_u[[2, 5, 8]] = local_u

    torch.testing.assert_close(
        _fun(mapped, mapped_u), _fun(local, local_u), rtol=0.0, atol=0.0
    )
    mapped_grad = _grad(mapped, mapped_u)
    torch.testing.assert_close(
        mapped_grad[[2, 5, 8]], _grad(local, local_u), rtol=0.0, atol=0.0
    )
    inactive = torch.ones(9, dtype=torch.bool)
    inactive[[2, 5, 8]] = False
    torch.testing.assert_close(
        mapped_grad[inactive],
        torch.zeros_like(mapped_grad[inactive]),
        rtol=0.0,
        atol=0.0,
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
