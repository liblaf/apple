import torch

from liblaf.apple.solvers.optim import Pncg


class QuadraticProblem:
    def __init__(self, matrix: torch.Tensor, rhs: torch.Tensor) -> None:
        self.matrix = matrix
        self.rhs = rhs

    def update(self, state: torch.Tensor, params: torch.Tensor, /) -> None:
        state.copy_(params)

    def fun(self, state: torch.Tensor, /) -> torch.Tensor:
        return 0.5 * torch.dot(state, self.matrix @ state) - torch.dot(self.rhs, state)

    def grad(self, state: torch.Tensor, /) -> torch.Tensor:
        return self.matrix @ state - self.rhs

    def hess_diag(self, state: torch.Tensor, /) -> torch.Tensor:
        del state
        return torch.diagonal(self.matrix)

    def hess_quad(
        self, state: torch.Tensor, direction: torch.Tensor, /
    ) -> torch.Tensor:
        del state
        return torch.dot(direction, self.matrix @ direction)


def test_pncg_converges_to_quadratic_minimizer() -> None:
    matrix = torch.tensor([[6.0, 2.0], [2.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([2.0, -1.0], dtype=torch.float64)
    initial = torch.tensor([-3.0, 4.0], dtype=torch.float64)
    model_state = initial.clone()
    problem = QuadraticProblem(matrix, rhs)

    solution = Pncg().minimize(problem, model_state, initial)
    expected = torch.linalg.solve(matrix, rhs)

    assert solution.success
    torch.testing.assert_close(solution.params, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(model_state, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        problem.grad(model_state), torch.zeros_like(rhs), rtol=0.0, atol=1e-10
    )
    assert solution.state.step <= matrix.shape[0] + 1


def test_pncg_reports_accepted_force_at_iteration_limit() -> None:
    problem = QuadraticProblem(
        torch.eye(1, dtype=torch.float64), torch.tensor([3.0], dtype=torch.float64)
    )
    initial = torch.zeros(1, dtype=torch.float64)
    state = initial.clone()
    optimizer = Pncg(
        criteria=Pncg.ConvergenceCriteria(
            max_steps=1,
            atol_primary=1e-12,
            rtol_primary=0.0,
            atol_secondary=1e-12,
            rtol_secondary=0.0,
        )
    )

    solution = optimizer.minimize(problem, state, initial)

    assert solution.success
    torch.testing.assert_close(state, problem.rhs)
    torch.testing.assert_close(
        solution.state.convergence_state.grad_norm,
        torch.linalg.vector_norm(problem.grad(state)),
    )
    torch.testing.assert_close(
        solution.state.convergence_state.grad_norm_first, torch.tensor(3.0).double()
    )
