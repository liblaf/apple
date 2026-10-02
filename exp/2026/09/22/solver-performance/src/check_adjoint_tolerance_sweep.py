"""CPU contracts for fixed-state adjoint replay helpers."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import optree
import torch
from adjoint_tolerance_common import FixedStateImplicit


def load_sweep() -> Any:
    path = Path(__file__).with_name("30-adjoint-tolerance-sweep.py")
    spec = spec_from_file_location("adjoint_tolerance_sweep_check", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DofMap:
    fixed_indices = torch.empty(0, dtype=torch.long)
    fixed_values = torch.empty(0)

    def to_free_grad(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def to_full_grad(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def to_free_hess_diag(self, value: torch.Tensor) -> torch.Tensor:
        return value


class Model:
    State = SimpleNamespace

    def __init__(self) -> None:
        self.dof_map = DofMap()
        self.collision = None
        self.materials: dict[str, dict[str, torch.Tensor]] = {}

    def get_materials(self):
        return self.materials

    def set_materials(self, materials: dict[str, dict[str, torch.Tensor]]) -> None:
        self.materials = materials

    def hess_diag(self, state: Any) -> torch.Tensor:
        return torch.ones_like(state.u)

    def hess_prod(self, state: Any, vector: torch.Tensor) -> torch.Tensor:
        del state
        return vector

    def mixed_derivative_prod(self, state: Any, p: torch.Tensor) -> None:
        del state
        self.materials["active"]["q"].grad = p.sum().reshape(1)


class Solver:
    def solve(self, problem: Any, initial: torch.Tensor) -> SimpleNamespace:
        del initial
        # H=I, so the exact solution is b. Deliberately call one HVP.
        assert torch.equal(problem.matvec(problem.b), problem.b)
        return SimpleNamespace(success=True, params=problem.b, result="CPU exact")


def test_fixed_state_backward_without_primal() -> None:
    runtime = SimpleNamespace(
        forward=SimpleNamespace(model=Model()),
        warm_adjoints={},
        tolerances={"adjoint_rtol": 1e-8},
        solver=Solver(),
    )
    q = torch.tensor([2.0], requires_grad=True)
    fixed = torch.empty(0)
    saved = torch.tensor([3.0])
    spec = optree.tree_flatten({"active": {"q": q}})[1]
    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None
    try:
        output = FixedStateImplicit.apply(runtime, "cpu", spec, fixed, saved, q)
        (gradient,) = torch.autograd.grad(output.sum(), q)
    finally:
        torch.cuda.synchronize = original_sync
    assert float(gradient) == -1.0
    assert runtime.last_adjoint["hvp_matvec_calls"] == 1
    assert runtime.last_adjoint["operator"] == "owned_unshifted_model.hess_prod"


def test_vector_comparison_zero_norm_contract() -> None:
    sweep = load_sweep()
    exact = sweep.vector_comparison(torch.zeros(2), torch.zeros(2))
    assert exact["relative_l2"] == 0.0
    assert exact["cosine"] == 1.0
    nonzero = sweep.vector_comparison(torch.ones(2), torch.zeros(2))
    assert nonzero["relative_l2"] is None
    assert nonzero["cosine"] is None


if __name__ == "__main__":
    test_fixed_state_backward_without_primal()
    test_vector_comparison_zero_norm_contract()
    print("fixed-state adjoint tolerance checks passed")
