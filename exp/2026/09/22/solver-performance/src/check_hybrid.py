# ruff: noqa: EM101, TRY003
"""CPU-only contract check for coarse PNCG followed by exact Newton."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import scipy.sparse
import torch

from liblaf.apple.solvers.optim import Pncg

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

spec = importlib.util.spec_from_file_location(
    "hybrid_hessian", HERE / "hybrid_hessian.py"
)
assert spec is not None
assert spec.loader is not None
hybrid_hessian = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hybrid_hessian
spec.loader.exec_module(hybrid_hessian)


class DofMap:
    def to_free(self, values: torch.Tensor) -> torch.Tensor:
        return values


class Model:
    dof_map = DofMap()


class QuarticProblem:
    """A smooth nonquadratic scalar problem that leaves work for Newton."""

    model = Model()

    def update(self, state: SimpleNamespace, values: torch.Tensor) -> None:
        state.u.copy_(values)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        value = state.u[0]
        return 0.5 * (value - 0.7).square() + 0.05 * value.pow(4)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        value = state.u[0]
        return torch.stack((value - 0.7 + 0.2 * value.pow(3),))

    def hess_diag(self, state: SimpleNamespace) -> torch.Tensor:
        value = state.u[0]
        return torch.stack((1.0 + 0.6 * value.square(),))

    def hess_prod(
        self, state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.hess_diag(state) * direction

    def hess_quad(
        self, state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.dot(direction, self.hess_prod(state, direction))

    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones((), dtype=torch.float64)


def default_optimizer(atol: float) -> Pncg:
    return Pncg(
        criteria=Pncg.ConvergenceCriteria(
            max_steps=100,
            atol_primary=atol,
            rtol_primary=0.0,
            atol_secondary=atol,
            rtol_secondary=0.0,
        )
    )


def check_hybrid_reaches_exact_force_after_coarse_transition() -> None:
    problem = accelerated_solvers.CachedProblem(QuarticProblem())
    state = SimpleNamespace(u=torch.tensor([4.0], dtype=torch.float64))
    initial_force = float(torch.linalg.vector_norm(problem.grad(state)))
    solved, receipt = accelerated_solvers.hybrid_pncg_newton(
        problem,
        state,
        initial_force=initial_force,
        atol=1e-12,
        make_default_optimizer=default_optimizer,
        hessian_damping_initial=0.001,
        line_search_armijo=0.25,
        max_step_norm=10.0,
        newton_max_step_norm=10.0,
        pncg_restart_interval=200,
        linear_rtol=1e-12,
        max_newton_steps=20,
        preconditioner="diag",
        newton_switch_atol=0.0,
    )
    assert solved is state
    assert receipt["coarse_threshold"] == initial_force * 1e-3
    assert receipt["coarse_steps"] > 0
    assert receipt["newton_steps"] > 0
    assert receipt["coarse_terminal_force"] <= receipt["coarse_threshold"]
    assert float(torch.linalg.vector_norm(problem.grad(state))) <= 1e-12


def check_hybrid_can_skip_warmup_but_not_final_force_gate() -> None:
    """A switching floor only skips PNCG; Newton still solves to ``atol``."""
    problem = accelerated_solvers.CachedProblem(QuarticProblem())
    state = SimpleNamespace(u=torch.tensor([0.72], dtype=torch.float64))
    initial_force = float(torch.linalg.vector_norm(problem.grad(state)))

    def pncg_must_not_run(_atol: float) -> Pncg:
        raise AssertionError("switch floor should skip PNCG warm-up")

    solved, receipt = accelerated_solvers.hybrid_pncg_newton(
        problem,
        state,
        initial_force=initial_force,
        atol=1e-12,
        make_default_optimizer=pncg_must_not_run,
        hessian_damping_initial=0.001,
        line_search_armijo=0.25,
        max_step_norm=10.0,
        newton_max_step_norm=10.0,
        pncg_restart_interval=200,
        linear_rtol=1e-12,
        max_newton_steps=20,
        preconditioner="diag",
        newton_switch_atol=1e-1,
    )
    assert solved is state
    assert receipt["coarse_steps"] == 0
    assert receipt["newton_switch_atol"] == 1e-1
    assert receipt["coarse_threshold"] == 1e-1
    assert receipt["coarse_terminal_force"] == initial_force
    assert receipt["newton_steps"] > 0
    assert float(torch.linalg.vector_norm(problem.grad(state))) <= 1e-12


def check_hybrid_hessian_accepts_exact_empty_contact() -> None:
    """No collision bypasses only IPC preparation; FEM/sparse setup remains."""

    class FakeFem:
        calls: ClassVar[int] = 0

        def __init__(self, _model: object, _state: object) -> None:
            type(self).calls += 1
            self.persistent_bytes = 7
            self.metadata = {
                "topology_cache_hit": True,
                "numeric_setup_seconds": 0.0,
            }

    class FakeSparse:
        calls: ClassVar[int] = 0

        def __init__(
            self, _model: object, state: object, _fem: object, *, shift: float
        ) -> None:
            assert state.collision is None
            assert shift == 0.0
            type(self).calls += 1
            self.persistent_bytes = 11
            self.metadata = {
                "shift": 0.0,
                "contact_pattern_changed": False,
                "union_reused_for_contact": False,
                "symbolic_cache_hit": True,
                "pattern_hash": "empty-contact",
                "free_nnz": 1,
                "lower_nnz": 1,
            }

    original_fem = hybrid_hessian.AssembledFemHvp
    original_sparse = hybrid_hessian.GpuFreeSparseHessian
    hybrid_hessian.AssembledFemHvp = FakeFem
    hybrid_hessian.GpuFreeSparseHessian = FakeSparse
    try:
        state = SimpleNamespace(
            u=torch.zeros((1, 3), dtype=torch.float64), collision=None
        )
        cache = hybrid_hessian.HybridHessian(SimpleNamespace(collision=None))
        cache._require_gpu_state = lambda _state: None  # noqa: SLF001
        cache.prepare(state)
        cache.prepare(state)
        assert FakeFem.calls == 1
        assert FakeSparse.calls == 1
        assert cache.metadata["numeric_refreshes"] == 1
        assert cache.metadata["prepare_cache_hits"] == 1
        assert cache.metadata["persistent_bytes"] == 18
    finally:
        hybrid_hessian.AssembledFemHvp = original_fem
        hybrid_hessian.GpuFreeSparseHessian = original_sparse


def check_hybrid_hessian_still_prepares_owned_contact() -> None:
    """The no-contact branch does not bypass existing owned-contact assembly."""

    class FakeFem:
        def __init__(self, _model: object, _state: object) -> None:
            self.persistent_bytes = 0
            self.metadata = {
                "topology_cache_hit": False,
                "numeric_setup_seconds": 0.0,
            }

    class FakeSparse:
        def __init__(
            self, _model: object, state: object, _fem: object, *, shift: float
        ) -> None:
            assert state.collision.hess is not None
            assert shift == 0.0
            self.persistent_bytes = 0
            self.metadata = {
                "shift": 0.0,
                "contact_pattern_changed": True,
                "union_reused_for_contact": False,
                "symbolic_cache_hit": False,
                "pattern_hash": "contact",
                "free_nnz": 1,
                "lower_nnz": 1,
            }

    class Potential:
        def __init__(self) -> None:
            self.calls = 0

        def hessian(self, **_kwargs: object) -> scipy.sparse.csr_matrix:
            self.calls += 1
            return scipy.sparse.eye(3, format="csr")

    class Collision:
        def __init__(self) -> None:
            self.vertices = torch.zeros((1, 3), dtype=torch.float64)
            self.indices = torch.tensor([0])
            self.potential = Potential()
            self.collision_mesh = object()

        def state_at(self, _u: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(collisions=object(), hess=None)

    original_fem = hybrid_hessian.AssembledFemHvp
    original_sparse = hybrid_hessian.GpuFreeSparseHessian
    hybrid_hessian.AssembledFemHvp = FakeFem
    hybrid_hessian.GpuFreeSparseHessian = FakeSparse
    try:
        collision = Collision()
        state = SimpleNamespace(
            u=torch.zeros((1, 3), dtype=torch.float64), collision=None
        )
        cache = hybrid_hessian.HybridHessian(SimpleNamespace(collision=collision))
        cache._require_gpu_state = lambda _state: None  # noqa: SLF001
        cache.prepare(state)
        assert collision.potential.calls == 1
        assert state.collision.hess.shape == (3, 3)
    finally:
        hybrid_hessian.AssembledFemHvp = original_fem
        hybrid_hessian.GpuFreeSparseHessian = original_sparse


if __name__ == "__main__":
    check_hybrid_reaches_exact_force_after_coarse_transition()
    check_hybrid_can_skip_warmup_but_not_final_force_gate()
    check_hybrid_hessian_accepts_exact_empty_contact()
    check_hybrid_hessian_still_prepares_owned_contact()
    print("hybrid CPU check passed")
