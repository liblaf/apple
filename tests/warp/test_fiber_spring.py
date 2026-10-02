from typing import Any

import numpy as np
import pytest
import torch
import warp as wp


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


def _hess_prod(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_prod(
        _from_torch_vec3(u),
        _from_torch_vec3(direction),
        _from_torch_vec3(output),
    )
    wp.synchronize()
    return output


def _hess_quad(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.hess_quad(
        _from_torch_vec3(u),
        _from_torch_vec3(direction),
        _from_torch_float(output),
    )
    wp.synchronize()
    return output[0]


def _single_spring(
    *, rest_length: float = 1.0, stiffness: float = 4.0, tension_only: bool = False
) -> Any:
    from liblaf.apple.warp.potential import FiberSpring

    return FiberSpring.from_arrays(
        np.array([[0, 1]], dtype=np.int32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([stiffness], dtype=np.float64),
        rest_lengths=np.array([rest_length], dtype=np.float64),
        tension_only=tension_only,
        device="cpu",
    )


def test_fiber_spring_matches_simple_axial_traction() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    potential = _single_spring()
    u = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
    direction = torch.tensor([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    hess_r = torch.diag(torch.tensor([4.0, 0.8, 0.8]))
    relative_direction = direction[1] - direction[0]
    expected_product = hess_r @ relative_direction

    torch.testing.assert_close(_fun(potential, u), torch.tensor(0.125))
    torch.testing.assert_close(
        _grad(potential, u),
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        _hess_diag(potential, u),
        torch.tensor([[4.0, 0.8, 0.8], [4.0, 0.8, 0.8]]),
    )
    torch.testing.assert_close(
        _hess_prod(potential, u, direction),
        torch.stack([-expected_product, expected_product]),
    )
    torch.testing.assert_close(
        _hess_quad(potential, u, direction),
        relative_direction @ expected_product,
    )


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA device")
def test_fiber_spring_launches_on_its_array_device() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    with wp.ScopedDevice("cuda:0"):
        potential = _single_spring()
        u = torch.tensor([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]], device="cpu")

        assert potential.endpoints.device.is_cpu
        assert all(array.device.is_cpu for array in potential.get_materials().values())
        torch.testing.assert_close(_fun(potential, u), torch.tensor(0.125))


def test_fiber_spring_is_invariant_to_rigid_motion() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    from liblaf.apple.warp.potential import FiberSpring

    points = torch.tensor([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])
    rest_vector = points[1] - points[0]
    potential = FiberSpring.from_arrays(
        np.array([[0, 1]], dtype=np.int32),
        rest_vector.numpy()[None, :],
        np.array([3.2]),
        rest_lengths=np.array([0.9 * torch.linalg.vector_norm(rest_vector).item()]),
        device="cpu",
    )
    angle = torch.tensor(0.63)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([-0.3, 0.8, 0.25])
    transformed = points @ rotation.T + translation
    rigid_u = transformed - points

    torch.testing.assert_close(
        _fun(potential, rigid_u),
        _fun(potential, torch.zeros_like(points)),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_fiber_spring_internal_gradient_is_force_balanced() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    from liblaf.apple.warp.potential import FiberSpring

    potential = FiberSpring.from_arrays(
        np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32),
        np.array(
            [[1.0, 0.2, 0.0], [-0.1, 0.8, 0.3], [0.9, 1.0, 0.3]],
            dtype=np.float64,
        ),
        np.array([1.2, 2.3, 0.7]),
        rest_lengths=np.array([0.9, 0.7, 1.1]),
        device="cpu",
    )
    u = torch.tensor([[0.03, -0.02, 0.01], [-0.01, 0.04, 0.02], [0.02, -0.03, 0.05]])

    torch.testing.assert_close(
        _grad(potential, u).sum(dim=0),
        torch.zeros(3),
        rtol=0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(("tension_only", "rest_length"), [(False, 1.2), (True, 0.8)])
def test_fiber_spring_derivatives_match_finite_difference(
    tension_only: Any, rest_length: float
) -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    from liblaf.apple.warp.potential import FiberSpring

    potential = FiberSpring.from_arrays(
        np.array([[1, 3], [0, 2]], dtype=np.int32),
        np.array([[1.1, 0.2, -0.1], [0.3, 0.9, 0.4]]),
        np.array([1.7, 0.6]),
        rest_lengths=np.array([rest_length, 0.75 * rest_length]),
        tension_only=tension_only,
        device="cpu",
    )
    u = torch.tensor(
        [
            [0.03, -0.02, 0.01],
            [-0.01, 0.04, -0.02],
            [0.02, 0.01, 0.03],
            [-0.04, 0.015, 0.02],
        ]
    )
    direction = torch.tensor(
        [
            [0.04, -0.03, 0.02],
            [0.01, 0.05, -0.04],
            [-0.02, 0.03, 0.01],
            [0.03, -0.01, 0.05],
        ]
    )
    eps = 1.0e-5

    grad = _grad(potential, u)
    hess_prod = _hess_prod(potential, u, direction)
    hess_quad = _hess_quad(potential, u, direction)
    hess_diag = _hess_diag(potential, u)
    fd_grad_dot_direction = (
        _fun(potential, u + eps * direction) - _fun(potential, u - eps * direction)
    ) / (2.0 * eps)
    torch.testing.assert_close(
        torch.sum(grad * direction),
        fd_grad_dot_direction,
        rtol=2.0e-9,
        atol=2.0e-11,
    )
    fd_hess_prod = (
        _grad(potential, u + eps * direction) - _grad(potential, u - eps * direction)
    ) / (2.0 * eps)
    torch.testing.assert_close(hess_prod, fd_hess_prod, rtol=2.0e-9, atol=2.0e-10)
    torch.testing.assert_close(
        hess_quad,
        torch.sum(hess_prod * direction),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    fd_diag = torch.empty_like(u)
    for index in range(u.numel()):
        basis = torch.zeros_like(u).reshape(-1)
        basis[index] = 1.0
        axis = basis.reshape_as(u)
        fd_column = (
            _grad(potential, u + eps * axis) - _grad(potential, u - eps * axis)
        ) / (2.0 * eps)
        fd_diag.reshape(-1)[index] = fd_column.reshape(-1)[index]
    torch.testing.assert_close(hess_diag, fd_diag, rtol=2.0e-9, atol=2.0e-10)


def test_tension_only_fiber_spring_has_zero_slack_response() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    potential = _single_spring(rest_length=1.2, tension_only=True)
    u = torch.tensor([[0.0, 0.0, 0.0], [-0.1, 0.2, 0.0]])
    direction = torch.tensor([[0.1, 0.3, -0.2], [-0.2, 0.4, 0.1]])

    torch.testing.assert_close(_fun(potential, u), torch.tensor(0.0))
    torch.testing.assert_close(_grad(potential, u), torch.zeros_like(u))
    torch.testing.assert_close(_hess_diag(potential, u), torch.zeros_like(u))
    torch.testing.assert_close(_hess_prod(potential, u, direction), torch.zeros_like(u))
    torch.testing.assert_close(_hess_quad(potential, u, direction), torch.tensor(0.0))


def test_fiber_spring_material_gradients_and_mixed_derivatives() -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    from liblaf.apple.warp.model import WarpModel
    from liblaf.apple.warp.potential import FiberSpring

    stiffness = 3.5
    rest_length = 0.8
    potential = FiberSpring.from_arrays(
        np.array([[0, 1]], dtype=np.int32),
        np.array([[1.0, 0.2, -0.1]]),
        np.array([stiffness]),
        rest_lengths=np.array([rest_length]),
        requires_grad=("rest_length", "stiffness"),
        device="cpu",
    )
    u_torch = torch.tensor([[0.03, -0.02, 0.01], [0.09, 0.04, -0.03]])
    u = _from_torch_vec3(u_torch)
    output = wp.zeros((1,), dtype=wp.float64, device="cpu", requires_grad=True)
    tape = wp.Tape()
    with tape:
        potential.fun(u, output)
    tape.backward(loss=output)

    r = torch.tensor([1.0, 0.2, -0.1]) + u_torch[1] - u_torch[0]
    length = torch.linalg.vector_norm(r)
    extension = length - rest_length
    materials = potential.get_materials()
    torch.testing.assert_close(
        wp.to_torch(materials["stiffness"].grad).cpu(),
        (0.5 * extension.square()).reshape(1),
    )
    torch.testing.assert_close(
        wp.to_torch(materials["rest_length"].grad).cpu(),
        (-stiffness * extension).reshape(1),
    )

    potential = FiberSpring.from_arrays(
        np.array([[0, 1]], dtype=np.int32),
        np.array([[1.0, 0.2, -0.1]]),
        np.array([stiffness]),
        rest_lengths=np.array([rest_length]),
        requires_grad=("rest_length", "stiffness"),
        device="cpu",
    )
    direction_torch = torch.tensor([[0.1, -0.3, 0.2], [-0.2, 0.4, 0.05]])
    model = WarpModel(potentials={"fiber": potential})
    model.mixed_derivative_prod(u, _from_torch_vec3(direction_torch))

    unit = r / length
    relative_direction = direction_torch[1] - direction_torch[0]
    unit_dot_direction = unit @ relative_direction
    materials = potential.get_materials()
    torch.testing.assert_close(
        wp.to_torch(materials["stiffness"].grad).cpu(),
        (extension * unit_dot_direction).reshape(1),
    )
    torch.testing.assert_close(
        wp.to_torch(materials["rest_length"].grad).cpu(),
        (-stiffness * unit_dot_direction).reshape(1),
    )


@pytest.mark.parametrize(
    ("endpoints", "rest_vectors", "rest_lengths", "message"),
    [
        ([[2, 2]], [[1.0, 0.0, 0.0]], [1.0], "distinct endpoints"),
        ([[0, 1]], [[0.0, 0.0, 0.0]], [1.0], "positive length"),
        ([[0, 1]], [[1.0, 0.0, 0.0]], [0.0], "finite and positive"),
    ],
)
def test_fiber_spring_rejects_degenerate_reference_data(
    endpoints: list[list[int]],
    rest_vectors: list[list[float]],
    rest_lengths: list[float],
    message: str,
) -> None:
    torch.set_default_dtype(torch.float64)
    wp.init()

    from liblaf.apple.warp.potential import FiberSpring

    with pytest.raises(ValueError, match=message):
        FiberSpring.from_arrays(
            np.asarray(endpoints),
            np.asarray(rest_vectors),
            np.array([1.0]),
            rest_lengths=np.asarray(rest_lengths),
            device="cpu",
        )
