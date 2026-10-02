from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse
import torch

from liblaf.apple.forward.hessian._contact import GpuContactHessian
from liblaf.apple.forward.hessian._problem import HessianProblem


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_contact_reuses_and_invalidates_exact_hessian() -> None:
    indices = torch.tensor([0, 2], device="cuda")
    matrix = np.diag([2.0, -1.0, 3.0, 4.0, 5.0, 6.0])
    matrix[1, 4] = matrix[4, 1] = 0.75
    collision = SimpleNamespace(indices=indices)
    adapter = GpuContactHessian(collision)
    state = SimpleNamespace(hess=scipy.sparse.csc_matrix(matrix))
    u = torch.zeros((3, 3), dtype=torch.float64, device="cuda")
    direction = torch.arange(9, dtype=u.dtype, device=u.device).reshape(3, 3)
    expected = torch.zeros_like(u)
    expected[indices] = (
        torch.as_tensor(matrix, device=u.device) @ direction[indices].flatten()
    ).reshape(-1, 3)
    for _ in range(2):
        output = torch.zeros_like(u)
        adapter.hess_prod(state, u, direction, output)
        torch.testing.assert_close(output, expected, rtol=1e-13, atol=1e-13)
    assert adapter.uploads == 1
    state.hess = scipy.sparse.csc_matrix(2 * matrix)
    output = torch.zeros_like(u)
    adapter.hess_prod(state, u, direction, output)
    torch.testing.assert_close(output, 2 * expected)
    assert adapter.uploads == 2
    adapter.invalidate()
    output.zero_()
    adapter.hess_prod(state, u, direction, output)
    assert adapter.uploads == 3
    assert adapter.persistent_bytes > 0


def test_reference_backend_delegates_without_changing_mechanics() -> None:
    class Problem:
        model = SimpleNamespace(collision=None)

        @staticmethod
        def hess_prod(state: SimpleNamespace, direction: torch.Tensor) -> torch.Tensor:
            return state.diagonal * direction

        @staticmethod
        def hess_diag(state: SimpleNamespace) -> torch.Tensor:
            return state.diagonal

        @staticmethod
        def update(state: SimpleNamespace, free: torch.Tensor) -> None:
            state.u.copy_(free)

    state = SimpleNamespace(u=torch.zeros(3), diagonal=torch.tensor([2.0, -1.0, 3.0]))
    problem = HessianProblem(Problem(), "matrix_free")
    direction = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(
        problem.hess_prod(state, direction), state.diagonal * direction
    )
    torch.testing.assert_close(problem.hess_diag(state), state.diagonal)
    torch.testing.assert_close(
        problem.hess_quad(state, direction), (state.diagonal * direction.square()).sum()
    )
    problem.update(state, direction)
    torch.testing.assert_close(state.u, direction)
