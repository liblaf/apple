# ruff: noqa: EM101, PT017, TRY003
"""CPU contracts for PCG and cached Newton-Hessian optimization plumbing.

The linear checks deliberately use small dense operators.  They verify the
observable numerical contract while leaving implementation choices such as
buffer reuse and reciprocal caching unconstrained.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
JOINT_SOURCE = HERE.parents[2] / "21/joint-activation-material-mandible/src"
sys.path[:0] = [str(HERE), str(JOINT_SOURCE)]
spec = importlib.util.spec_from_file_location(
    "accelerated_solvers", HERE / "accelerated_solvers.py"
)
assert spec is not None
assert spec.loader is not None
accelerated_solvers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = accelerated_solvers
spec.loader.exec_module(accelerated_solvers)


def check_spd_pcg_preserves_rhs_and_returns_independent_solution() -> None:
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([2.0, -1.0], dtype=torch.float64)
    before = rhs.clone()
    solution, receipt = accelerated_solvers.pcg(
        lambda vector: matrix @ vector,
        lambda vector: vector / torch.diagonal(matrix),
        rhs,
        rtol=1.0e-12,
        max_steps=4,
    )
    assert solution.data_ptr() != rhs.data_ptr()
    torch.testing.assert_close(rhs, before)
    relative = torch.linalg.vector_norm(
        rhs - matrix @ solution
    ) / torch.linalg.vector_norm(rhs)
    assert receipt["relative_residual"] <= 1.0e-12
    assert float(relative) <= 1.0e-12


def check_residual_replacement_enforces_true_spd_residual() -> None:
    """A falsely small recursive norm must not terminate before true residuals."""
    diagonal = torch.tensor([1.0, 2.0], dtype=torch.float64)
    rhs = torch.ones(2, dtype=torch.float64)
    original = torch.linalg.vector_norm
    calls = 0
    rhs_norm: torch.Tensor | None = None

    def understate_recursive_norm(
        values: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        nonlocal calls, rhs_norm
        calls += 1
        measured = original(values, *args, **kwargs)
        if calls == 1:
            rhs_norm = measured
        if calls == 2:
            assert rhs_norm is not None
            return rhs_norm * 0.30
        return measured

    torch.linalg.vector_norm = understate_recursive_norm
    try:
        solution, receipt = accelerated_solvers.pcg(
            lambda vector: diagonal * vector,
            lambda vector: vector,
            rhs,
            rtol=0.32,
            max_steps=2,
        )
    finally:
        torch.linalg.vector_norm = original
    # The actual first residual is 1/3: inside the old 1.05 allowance but
    # outside rtol.  Replacement forces a second iteration.
    assert receipt["steps"] == 2
    relative = torch.linalg.vector_norm(
        rhs - diagonal * solution
    ) / torch.linalg.vector_norm(rhs)
    assert receipt["relative_residual"] <= 0.32
    assert float(relative) <= 0.32


def check_nonpositive_and_nonfinite_curvature_rejects() -> None:
    rhs = torch.ones(2, dtype=torch.float64)
    for matvec in (
        lambda vector: -vector,
        lambda vector: torch.full_like(vector, torch.nan),
    ):
        try:
            accelerated_solvers.pcg(
                matvec,
                lambda vector: vector,
                rhs,
                rtol=1.0e-3,
                max_steps=2,
            )
        except accelerated_solvers.LinearRejection as error:
            assert str(error) == "nonpositive or nonfinite CG curvature"
        else:  # pragma: no cover
            raise AssertionError("invalid curvature was accepted")


def check_linear_iteration_budget_rejects() -> None:
    matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    try:
        accelerated_solvers.pcg(
            lambda vector: matrix @ vector,
            lambda vector: vector,
            torch.ones(2, dtype=torch.float64),
            rtol=1.0e-12,
            max_steps=1,
        )
    except accelerated_solvers.LinearRejection as error:
        assert str(error) == "CG iteration budget exhausted"
    else:  # pragma: no cover
        raise AssertionError("linear iteration budget was silently relaxed")


if __name__ == "__main__":
    check_spd_pcg_preserves_rhs_and_returns_independent_solution()
    check_residual_replacement_enforces_true_spd_residual()
    check_nonpositive_and_nonfinite_curvature_rejects()
    check_linear_iteration_budget_rejects()
    print("Hybrid optimization CPU checks passed")
