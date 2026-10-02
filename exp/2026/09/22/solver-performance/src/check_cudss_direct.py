"""Tiny CUDA ABI and numerical check for :mod:`cudss_direct`.

Run only after setting ``CUDSS_LIBRARY`` to the staged libcudss shared object.
It exercises a 4x4 SPD solve, a same-pattern numeric refactorization, and a
4x4 symmetric-indefinite solve with an inertia receipt.
"""

from __future__ import annotations

import json

import torch
from cudss_direct import CudssDirect


def _lower(dense: torch.Tensor) -> torch.Tensor:
    return torch.tril(dense).to_sparse_csr()


def _residual(matrix: torch.Tensor, solution: torch.Tensor, rhs: torch.Tensor) -> float:
    value = matrix @ solution - rhs
    return float(torch.linalg.vector_norm(value) / torch.linalg.vector_norm(rhs))


def check_spd_and_refactorization() -> dict:
    matrix = torch.tensor(
        (
            (6.0, 1.0, 0.0, 0.0),
            (1.0, 5.0, 1.0, 0.0),
            (0.0, 1.0, 4.0, 1.0),
            (0.0, 0.0, 1.0, 3.0),
        ),
        device="cuda",
        dtype=torch.float64,
    )
    rhs = torch.tensor((1.0, -2.0, 3.0, 0.5), device="cuda", dtype=torch.float64)
    with CudssDirect(_lower(matrix), mode="spd") as direct:
        analysis = direct.analyze()
        factor = direct.factorize()
        assert factor["info"] == 0, factor
        solution = direct.solve(rhs)
        torch.testing.assert_close(
            solution, torch.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-12
        )
        assert _residual(matrix, solution, rhs) < 2e-12
        assert analysis["config"]["pivot_epsilon"] == 0.0
        scaled = 1.5 * matrix
        refactor = direct.refactorize(_lower(scaled).values().contiguous())
        assert refactor["info"] == 0, refactor
        scaled_solution = direct.solve(rhs)
        torch.testing.assert_close(
            scaled_solution, torch.linalg.solve(scaled, rhs), rtol=2e-12, atol=2e-12
        )
        assert _residual(scaled, scaled_solution, rhs) < 2e-12
    return {"analysis": analysis, "factor": factor, "refactor": refactor}


def check_symmetric_inertia() -> dict:
    matrix = torch.tensor(
        (
            (2.0, 1.0, 0.0, 0.0),
            (1.0, -3.0, 1.0, 0.0),
            (0.0, 1.0, 4.0, 1.0),
            (0.0, 0.0, 1.0, -2.0),
        ),
        device="cuda",
        dtype=torch.float64,
    )
    rhs = torch.tensor((1.0, 2.0, -1.0, 0.5), device="cuda", dtype=torch.float64)
    with CudssDirect(_lower(matrix), mode="symmetric") as direct:
        direct.analyze()
        factor = direct.factorize()
        assert factor["info"] == 0, factor
        assert factor["inertia_status"] == 0, factor
        assert factor["inertia"] is not None, factor
        expected = torch.linalg.eigvalsh(matrix)
        expected_inertia = [int((expected > 0).sum()), int((expected < 0).sum())]
        assert factor["inertia"] == expected_inertia, (factor, expected_inertia)
        solution = direct.solve(rhs)
        torch.testing.assert_close(
            solution, torch.linalg.solve(matrix, rhs), rtol=2e-12, atol=2e-12
        )
        assert _residual(matrix, solution, rhs) < 2e-12
    return {"factor": factor}


def main() -> None:
    assert torch.cuda.is_available(), "this check requires CUDA"
    torch.set_default_dtype(torch.float64)
    torch.cuda.synchronize()
    receipt = {
        "spd": check_spd_and_refactorization(),
        "symmetric": check_symmetric_inertia(),
    }
    torch.cuda.synchronize()
    print(json.dumps(receipt, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
