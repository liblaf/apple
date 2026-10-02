import numpy as np
import pytest
import pyvista as pv
import torch
import warp as wp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Forward and CuPy adjoint regression requires CUDA",
)


def test_fiber_spring_equilibrium_and_material_adjoint() -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE, FORCE
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.inverse import DifferentiableForward
    from liblaf.apple.solvers.linalg import CupyCG
    from liblaf.apple.warp.potential import ExternalForce, FiberSpring

    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_default_device("cuda")
        wp.init()
        with wp.ScopedDevice("cuda:0"):
            points = pv.PolyData(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
            )
            builder = ModelBuilder()
            builder.add_vertices(points)
            fixed_mask = np.array([[True, True, True], [False, True, True]], dtype=bool)
            points.point_data[FIXED_MASK.vtk] = fixed_mask
            points.point_data[FIXED_VALUE.vtk] = np.zeros((2, 3), dtype=np.float64)
            builder.add_fixed(points)
            spring = FiberSpring.from_arrays(
                np.array([[0, 1]], dtype=np.int32),
                np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                np.array([4.0], dtype=np.float64),
                rest_lengths=np.array([1.0], dtype=np.float64),
                name="fiber",
            )
            builder.add_potential(spring)
            force = np.zeros((2, 3), dtype=np.float64)
            force[1, 0] = 0.4
            points.point_data[FORCE.vtk] = force
            builder.add_potential(ExternalForce.from_pyvista(points, name="load"))
            differentiable = DifferentiableForward(
                Forward(builder.finalize()),
                adjoint_solver=CupyCG(maxiter=100, rtol=1.0e-12, atol=1.0e-14),
            )
            materials = differentiable.model.get_materials()["fiber"]
            stiffness = materials["stiffness"].detach().clone().requires_grad_()
            rest_length = materials["rest_length"].detach().clone().requires_grad_()

            displacement = differentiable.forward(
                {"fiber": {"stiffness": stiffness, "rest_length": rest_length}}
            )
            displacement.retain_grad()
            displacement[1, 0].backward()

            assert differentiable.last_solution is not None
            assert differentiable.last_solution.success
            assert differentiable.last_adjoint_solution is not None
            assert differentiable.last_adjoint_solution.success
            assert stiffness.grad is not None
            assert rest_length.grad is not None
            torch.testing.assert_close(
                displacement.detach().cpu(),
                torch.tensor(
                    [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
                    device="cpu",
                ),
                rtol=1.0e-9,
                atol=1.0e-11,
            )
            torch.testing.assert_close(
                stiffness.grad.detach().cpu(),
                torch.tensor([-0.025], device="cpu"),
                rtol=1.0e-8,
                atol=1.0e-10,
            )
            torch.testing.assert_close(
                rest_length.grad.detach().cpu(),
                torch.tensor([1.0], device="cpu"),
                rtol=1.0e-9,
                atol=1.0e-11,
            )
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
