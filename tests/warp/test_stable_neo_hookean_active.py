# Copyright (c) 2026 liblaf
from typing import Any

import numpy as np
import pytest
import pyvista as pv
import torch
import warp as wp


@pytest.fixture(autouse=True)
def _cpu_float64_defaults() -> None:
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def _make_tetra_mesh(
    *,
    lambda_: float = 2.3,
    mu: float = 1.1,
    activation_inv: np.ndarray | None = None,
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    mesh = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3]),
        np.array([pv.CellType.TETRA], dtype=np.uint8),
        np.vstack((np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))),
    )
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.array([lambda_], dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.array([mu], dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.ones(1, dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.asarray(
        [np.zeros(6) if activation_inv is None else activation_inv], dtype=np.float64
    )
    return mesh


def _from_torch_vec3(x: torch.Tensor) -> wp.array:
    return wp.from_torch(x, dtype=wp.types.vector(3, wp.dtype_from_torch(x.dtype)))


def _from_torch_float(x: torch.Tensor) -> wp.array:
    return wp.from_torch(x, dtype=wp.dtype_from_torch(x.dtype))


def _fun(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, dtype=u.dtype, device=u.device)
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
    output = torch.zeros(1, dtype=u.dtype, device=u.device)
    potential.hess_quad(
        _from_torch_vec3(u), _from_torch_vec3(p), _from_torch_float(output)
    )
    wp.synchronize()
    return output[0]


def _displacement() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.0, 0.0], [0.08, 0.02, -0.01], [0.01, -0.04, 0.03], [0.0, -0.02, 0.06]]
    )


def _activation_matrix(
    activation_inv: np.ndarray, *, dtype: torch.dtype
) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0 + activation_inv[0], activation_inv[3], activation_inv[5]],
            [activation_inv[3], 1.0 + activation_inv[1], activation_inv[4]],
            [activation_inv[5], activation_inv[4], 1.0 + activation_inv[2]],
        ],
        dtype=dtype,
    )


def _activated_norm_energy(
    u: torch.Tensor, activation_inv: np.ndarray, *, mu: float
) -> torch.Tensor:
    """The activated isochoric-norm term, including this tetrahedron's volume."""
    F = torch.eye(3, dtype=u.dtype) + torch.stack(
        (u[1] - u[0], u[2] - u[0], u[3] - u[0]), dim=1
    )
    B = _activation_matrix(activation_inv, dtype=u.dtype)
    return 0.5 * mu * ((F @ B).square().sum() - 3.0) / 6.0


def test_active_stable_neo_hookean_uses_activated_norm_and_physical_volume() -> None:
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    wp.init()
    lambda_, mu = 2.3, 1.1
    activation_inv = np.array([0.2, -0.1, 0.05, 0.03, -0.02, 0.04])
    potential = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(lambda_=lambda_, mu=mu, activation_inv=activation_inv)
    )
    u = _displacement()
    F = torch.eye(3) + torch.stack((u[1], u[2], u[3]), dim=1)
    B = torch.eye(3)
    B[0, 0] += activation_inv[0]
    B[1, 1] += activation_inv[1]
    B[2, 2] += activation_inv[2]
    B[0, 1] = B[1, 0] = activation_inv[3]
    B[1, 2] = B[2, 1] = activation_inv[4]
    B[0, 2] = B[2, 0] = activation_inv[5]
    J = torch.linalg.det(F)
    expected_density = (
        0.5 * mu * (torch.sum((F @ B) ** 2) - 3.0)
        - mu * (J - 1.0)
        + 0.5 * lambda_ * (J - 1.0) ** 2
    )

    torch.testing.assert_close(_fun(potential, u), expected_density / 6.0)


def test_active_strain_leaves_both_physical_volume_terms_outside_activation() -> None:
    """Changing B only changes the activated norm, never either det(F) term."""
    from liblaf.apple.common import ACTIVATION_INV
    from liblaf.apple.warp.fem import StableNeoHookeanActive
    from liblaf.apple.warp.model import WarpModel

    wp.init()
    lambda_, mu = 2.3, 1.1
    activation_inv = np.array([0.24, -0.13, 0.08, 0.05, -0.03, 0.07])
    identity = np.zeros(6)
    u = _displacement()
    p = torch.tensor(
        [
            [0.02, -0.03, 0.01],
            [-0.01, 0.04, -0.02],
            [0.03, 0.01, 0.02],
            [-0.02, 0.02, -0.03],
        ]
    )
    F = torch.eye(3) + torch.stack((u[1], u[2], u[3]), dim=1)
    assert not torch.isclose(torch.linalg.det(F), torch.tensor(1.0))
    assert not torch.isclose(
        torch.linalg.det(_activation_matrix(activation_inv, dtype=u.dtype)),
        torch.tensor(1.0),
    )

    active = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(lambda_=lambda_, mu=mu, activation_inv=activation_inv)
    )
    passive_activation = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(lambda_=lambda_, mu=mu, activation_inv=identity)
    )

    def norm_difference(candidate: torch.Tensor) -> torch.Tensor:
        return _activated_norm_energy(
            candidate, activation_inv, mu=mu
        ) - _activated_norm_energy(candidate, identity, mu=mu)

    reference_u = u.detach().clone().requires_grad_()
    reference_energy = norm_difference(reference_u)
    reference_grad = torch.autograd.grad(
        reference_energy, reference_u, create_graph=True
    )[0]
    reference_hvp = torch.autograd.grad((reference_grad * p).sum(), reference_u)[0]
    reference_diag = torch.empty_like(u)
    for index in range(u.numel()):
        candidate = u.detach().clone().requires_grad_()
        gradient = torch.autograd.grad(
            norm_difference(candidate), candidate, create_graph=True
        )[0]
        reference_diag.reshape(-1)[index] = torch.autograd.grad(
            gradient.reshape(-1)[index], candidate
        )[0].reshape(-1)[index]

    torch.testing.assert_close(
        _fun(active, u) - _fun(passive_activation, u), reference_energy
    )
    torch.testing.assert_close(
        _grad(active, u) - _grad(passive_activation, u), reference_grad
    )
    torch.testing.assert_close(
        _hess_prod(active, u, p) - _hess_prod(passive_activation, u, p),
        reference_hvp,
    )
    torch.testing.assert_close(
        _hess_diag(active, u) - _hess_diag(passive_activation, u), reference_diag
    )

    mixed = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(lambda_=lambda_, mu=mu, activation_inv=activation_inv),
        requires_grad=(ACTIVATION_INV.value,),
    )
    WarpModel(potentials={"muscle": mixed}).mixed_derivative_prod(
        _from_torch_vec3(u), _from_torch_vec3(p)
    )
    mixed_activation = wp.to_torch(
        mixed.get_materials()[ACTIVATION_INV.value].grad
    ).cpu()[0]
    eps = 1.0e-6
    expected_mixed = torch.empty(6)
    for index in range(6):
        plus, minus = activation_inv.copy(), activation_inv.copy()
        plus[index] += eps
        minus[index] -= eps
        # Evaluate the two norm-only displacement gradients on independent
        # autograd leaves, so no physical determinant term can enter the FD.
        plus_u = u.detach().clone().requires_grad_()
        minus_u = u.detach().clone().requires_grad_()
        plus_gradient = torch.autograd.grad(
            _activated_norm_energy(plus_u, plus, mu=mu), plus_u
        )[0]
        minus_gradient = torch.autograd.grad(
            _activated_norm_energy(minus_u, minus, mu=mu), minus_u
        )[0]
        expected_mixed[index] = torch.sum((plus_gradient - minus_gradient) * p) / (
            2.0 * eps
        )
    torch.testing.assert_close(
        mixed_activation, expected_mixed, rtol=2.0e-8, atol=2.0e-10
    )


def test_active_stable_neo_hookean_plane_strain_reduction() -> None:
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    wp.init()
    lambda_, mu, stretch_x, stretch_y, inverse_fiber_stretch = 2.3, 1.1, 1.08, 0.96, 1.2
    potential = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(
            lambda_=lambda_,
            mu=mu,
            activation_inv=np.array([inverse_fiber_stretch - 1.0, 0, 0, 0, 0, 0]),
        )
    )
    u = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [stretch_x - 1.0, 0.0, 0.0],
            [0.0, stretch_y - 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    J = stretch_x * stretch_y
    expected_density = (
        0.5 * mu * ((stretch_x * inverse_fiber_stretch) ** 2 + stretch_y**2 - 2.0)
        - mu * (J - 1.0)
        + 0.5 * lambda_ * (J - 1.0) ** 2
    )

    torch.testing.assert_close(_fun(potential, u), torch.tensor(expected_density / 6.0))


def test_active_stable_neo_hookean_identity_activation_matches_passive() -> None:
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    wp.init()
    mesh = _make_tetra_mesh()
    active = StableNeoHookeanActive.from_pyvista(mesh)
    passive = StableNeoHookean.from_pyvista(mesh)
    u = _displacement()
    p = torch.tensor(
        [
            [0.02, -0.03, 0.01],
            [-0.01, 0.04, -0.02],
            [0.03, 0.01, 0.02],
            [-0.02, 0.02, -0.03],
        ]
    )

    torch.testing.assert_close(_fun(active, u), _fun(passive, u))
    torch.testing.assert_close(_grad(active, u), _grad(passive, u))
    torch.testing.assert_close(_hess_diag(active, u), _hess_diag(passive, u))
    torch.testing.assert_close(_hess_prod(active, u, p), _hess_prod(passive, u, p))
    torch.testing.assert_close(_hess_quad(active, u, p), _hess_quad(passive, u, p))


def test_active_stable_neo_hookean_derivatives_match_finite_difference() -> None:
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    wp.init()
    potential = StableNeoHookeanActive.from_pyvista(
        _make_tetra_mesh(
            activation_inv=np.array([0.1, -0.04, 0.03, 0.02, -0.01, 0.015])
        )
    )
    u = _displacement()
    p = torch.tensor(
        [
            [0.02, -0.03, 0.01],
            [-0.01, 0.04, -0.02],
            [0.03, 0.01, 0.02],
            [-0.02, 0.02, -0.03],
        ]
    )
    eps = 1.0e-5
    fd_energy_direction = (
        _fun(potential, u + eps * p) - _fun(potential, u - eps * p)
    ) / (2.0 * eps)
    torch.testing.assert_close(
        torch.sum(_grad(potential, u) * p),
        fd_energy_direction,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    hess_prod = _hess_prod(potential, u, p)
    hess_quad = _hess_quad(potential, u, p)
    fd_hess_prod = (_grad(potential, u + eps * p) - _grad(potential, u - eps * p)) / (
        2.0 * eps
    )
    torch.testing.assert_close(hess_prod, fd_hess_prod, rtol=2.0e-7, atol=2.0e-8)
    torch.testing.assert_close(
        hess_quad, torch.sum(hess_prod * p), rtol=1.0e-12, atol=1.0e-12
    )

    fd_diag = torch.empty_like(u)
    for index in range(u.numel()):
        axis = torch.zeros_like(u).reshape(-1)
        axis[index] = 1.0
        axis = axis.reshape_as(u)
        fd_column = (
            _grad(potential, u + eps * axis) - _grad(potential, u - eps * axis)
        ) / (2.0 * eps)
        fd_diag.reshape(-1)[index] = fd_column.reshape(-1)[index]
    torch.testing.assert_close(
        _hess_diag(potential, u), fd_diag, rtol=2.0e-7, atol=2.0e-8
    )


def test_active_stable_neo_hookean_material_gradients_match_finite_difference() -> None:
    from liblaf.apple.common import ACTIVATION_INV, LAMBDA
    from liblaf.apple.warp.fem import StableNeoHookeanActive
    from liblaf.apple.warp.model import WarpModel

    wp.init()
    activation_inv = np.array([0.1, -0.04, 0.03, 0.02, -0.01, 0.015])
    lambda_ = 2.3
    mesh = _make_tetra_mesh(lambda_=lambda_, activation_inv=activation_inv)
    potential = StableNeoHookeanActive.from_pyvista(
        mesh, requires_grad=(ACTIVATION_INV.value, LAMBDA.value)
    )
    assert potential.cells.device.is_cpu
    assert all(
        value.device == potential.cells.device
        for value in potential.get_materials().values()
    )
    u = _displacement()
    output = wp.zeros(1, dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        potential.fun(_from_torch_vec3(u), output)
    tape.backward(loss=output)
    materials = potential.get_materials()
    activation_gradient = wp.to_torch(materials[ACTIVATION_INV.value].grad).cpu()[0]
    lambda_gradient = wp.to_torch(materials[LAMBDA.value].grad).cpu()[0]
    assert torch.linalg.vector_norm(activation_gradient) > 0.0
    assert lambda_gradient > 0.0

    eps = 1.0e-6
    fd_potential = StableNeoHookeanActive.from_pyvista(mesh)

    def energy_at(candidate: np.ndarray) -> torch.Tensor:
        fd_potential.set_materials(
            {
                ACTIVATION_INV.value: torch.from_numpy(
                    np.ascontiguousarray(candidate[None])
                ).to(dtype=u.dtype)
            }
        )
        return _fun(fd_potential, u)

    def gradient_at(candidate: np.ndarray) -> torch.Tensor:
        fd_potential.set_materials(
            {
                ACTIVATION_INV.value: torch.from_numpy(
                    np.ascontiguousarray(candidate[None])
                ).to(dtype=u.dtype)
            }
        )
        return _grad(fd_potential, u)

    activation_fd = torch.empty(6)
    for index in range(6):
        plus = activation_inv.copy()
        minus = activation_inv.copy()
        plus[index] += eps
        minus[index] -= eps
        activation_fd[index] = (energy_at(plus) - energy_at(minus)) / (2.0 * eps)
    F = torch.eye(3) + torch.stack((u[1], u[2], u[3]), dim=1)
    lambda_fd = 0.5 * (torch.linalg.det(F) - 1.0) ** 2 / 6.0
    torch.testing.assert_close(
        activation_gradient, activation_fd, rtol=2.0e-8, atol=2.0e-10
    )
    torch.testing.assert_close(lambda_gradient, lambda_fd, rtol=2.0e-8, atol=2.0e-10)

    direction = torch.tensor(
        [
            [0.02, -0.03, 0.01],
            [-0.01, 0.04, -0.02],
            [0.03, 0.01, 0.02],
            [-0.02, 0.02, -0.03],
        ]
    )
    mixed_potential = StableNeoHookeanActive.from_pyvista(
        mesh, requires_grad=(ACTIVATION_INV.value,)
    )
    WarpModel(potentials={"muscle": mixed_potential}).mixed_derivative_prod(
        _from_torch_vec3(u), _from_torch_vec3(direction)
    )
    mixed_activation = wp.to_torch(
        mixed_potential.get_materials()[ACTIVATION_INV.value].grad
    ).cpu()[0]
    plus = activation_inv.copy()
    minus = activation_inv.copy()
    plus[0] += eps
    minus[0] -= eps
    mixed_fd = torch.sum((gradient_at(plus) - gradient_at(minus)) * direction) / (
        2.0 * eps
    )
    torch.testing.assert_close(mixed_activation[0], mixed_fd, rtol=2.0e-8, atol=2.0e-10)
