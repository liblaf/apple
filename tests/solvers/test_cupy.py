import pytest
import torch

cp = pytest.importorskip("cupy")

from liblaf.apple.solvers.linalg import CupyCG, CupyMinRes  # noqa: E402


def _cuda_available() -> bool:
    try:
        return torch.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")


class MatrixProblem:
    def __init__(self, matrix: torch.Tensor, rhs: torch.Tensor) -> None:
        self.matrix = matrix
        self._rhs = rhs

    @property
    def b(self) -> torch.Tensor:
        return self._rhs

    def matvec(self, vector: torch.Tensor, /) -> torch.Tensor:
        return self.matrix @ vector


def test_cupy_cg_solves_spd_system_in_float64() -> None:
    with torch.device("cuda"):
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64, device="cuda"
        )
        rhs = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")
        problem = MatrixProblem(matrix, rhs)

        solution = CupyCG(rtol=1e-12, atol=0.0).solve(problem, torch.zeros_like(rhs))

    assert solution.success
    assert solution.params.dtype is torch.float64
    assert solution.params.device.type == "cuda"
    torch.testing.assert_close(
        solution.params, torch.linalg.solve(matrix, rhs), rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(matrix @ solution.params, rhs, rtol=1e-11, atol=1e-12)


def test_cupy_minres_solves_symmetric_indefinite_system() -> None:
    with torch.device("cuda"):
        matrix = torch.tensor(
            [[1.0, 2.0], [2.0, -1.0]], dtype=torch.float64, device="cuda"
        )
        rhs = torch.tensor([3.0, -1.0], dtype=torch.float64, device="cuda")
        problem = MatrixProblem(matrix, rhs)

        solution = CupyMinRes(tol=1e-12).solve(problem, torch.zeros_like(rhs))

    assert solution.success
    torch.testing.assert_close(
        solution.params, torch.linalg.solve(matrix, rhs), rtol=1e-11, atol=1e-12
    )
    torch.testing.assert_close(matrix @ solution.params, rhs, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(
        solution.stats.absolute_residual,
        torch.tensor(0.0, dtype=torch.float64, device="cuda"),
        rtol=0.0,
        atol=1e-11,
    )
