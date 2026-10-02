# ruff: noqa: EM101, PT017, TRY003
"""CPU contracts for the adaptive PNCG/Newton controller.

The fixtures deliberately use tiny smooth objectives.  They check controller
state ownership and trigger semantics before any CUDA/contact benchmark.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
JOINT_SOURCE = HERE.parents[2] / "21" / "joint-activation-material-mandible" / "src"
sys.path[:0] = [str(HERE), str(JOINT_SOURCE)]


def load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


accelerated_solvers = load_module(
    "accelerated_solvers", HERE / "accelerated_solvers.py"
)
Pncg = accelerated_solvers.AcceptedForcePncg
adaptive_pncg = load_module("adaptive_pncg", HERE / "adaptive_pncg.py")
check_hybrid = load_module("check_hybrid_fixture", HERE / "check_hybrid.py")


def strict_optimizer(atol: float) -> object:
    criteria = Pncg.ConvergenceCriteria(
        max_steps=100,
        atol_primary=atol,
        rtol_primary=0.0,
        atol_secondary=atol,
        rtol_secondary=0.0,
    )
    return accelerated_solvers.AcceptedForcePncg(
        criteria=criteria,
        hess_damping=accelerated_solvers.AcceptedForcePncg.HessianDamping(
            initial=0.001
        ),
        line_search=accelerated_solvers.StrictLineSearch(
            armijo=0.25, max_steps=60, max_step_norm=10.0
        ),
    )


def check_no_trigger_matches_ordinary_quartic_pncg() -> None:
    """The controller must not perturb a trajectory when no trigger is allowed."""
    plain_problem = accelerated_solvers.CachedProblem(check_hybrid.QuarticProblem())
    plain_state = SimpleNamespace(u=torch.tensor([4.0], dtype=torch.float64))
    plain_optimizer = strict_optimizer(1e-12)
    plain_optimizer.restart_interval = 200
    plain = plain_optimizer.minimize(plain_problem, plain_state, plain_state.u)
    assert plain.success

    adaptive_problem = accelerated_solvers.CachedProblem(check_hybrid.QuarticProblem())
    adaptive_state = SimpleNamespace(u=torch.tensor([4.0], dtype=torch.float64))
    solved, receipt = adaptive_pncg.adaptive_pncg_newton(
        adaptive_problem,
        adaptive_state,
        atol=1e-12,
        make_default_optimizer=strict_optimizer,
        hessian_damping_initial=0.001,
        line_search_armijo=0.25,
        max_step_norm=10.0,
        newton_max_step_norm=10.0,
        pncg_restart_interval=200,
        linear_rtol=1e-12,
        max_newton_steps=100,
        max_pncg_steps=100,
        window_steps=1,
        minimum_reduction=0.1,
        required_poor_windows=101,
    )
    assert solved is adaptive_state
    assert receipt["newton_steps"] == 0
    assert receipt["pncg_steps"] == plain.state.step
    assert len(receipt["trace"]) == receipt["pncg_steps"] + 1
    assert all(row["kind"] in {"initial", "pncg"} for row in receipt["trace"])
    torch.testing.assert_close(adaptive_state.u, plain_state.u, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(
        torch.linalg.vector_norm(adaptive_problem.grad(adaptive_state)),
        torch.linalg.vector_norm(plain_problem.grad(plain_state)),
        rtol=1e-12,
        atol=1e-13,
    )


def check_force_windows_trigger_exactly_and_reset() -> None:
    detector = adaptive_pncg.ForceWindows(
        10.0, window_steps=2, minimum_reduction=0.1, required_poor_windows=2
    )
    assert detector.observe(10.0) == (False, None)
    trigger, first = detector.observe(10.0)
    assert not trigger
    assert first is not None
    assert first["segment_pncg_steps"] == 2
    assert first["fractional_improvement"] == 0.0
    assert first["consecutive_poor_windows"] == 1
    assert detector.observe(10.0) == (False, None)
    trigger, second = detector.observe(10.0)
    assert trigger
    assert second is not None
    assert second["segment_pncg_steps"] == 4
    assert second["consecutive_poor_windows"] == 2

    # This is the controller's post-Newton reset: earlier poor windows must
    # not contribute to the next trigger.
    reset = adaptive_pncg.ForceWindows(
        8.0, window_steps=2, minimum_reduction=0.1, required_poor_windows=2
    )
    assert reset.observe(8.0) == (False, None)
    trigger, receipt = reset.observe(8.0)
    assert not trigger
    assert receipt is not None
    assert receipt["consecutive_poor_windows"] == 1
    assert receipt["start_best_force"] == 8.0


class DofMap:
    def to_free(self, values: torch.Tensor) -> torch.Tensor:
        return values


class Model:
    dof_map = DofMap()


class CappedQuadratic:
    """A real Newton/PNCG fixture whose shared step cap forces corrections."""

    model = Model()

    def update(self, state: SimpleNamespace, values: torch.Tensor) -> None:
        state.u.copy_(values)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        return 0.5 * torch.dot(state.u, state.u)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        return state.u.clone()

    def hess_diag(self, state: SimpleNamespace) -> torch.Tensor:
        return torch.ones_like(state.u)

    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return direction

    def hess_quad(
        self, state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.dot(direction, self.hess_prod(state, direction))

    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones((), dtype=torch.float64)


def capped_optimizer(atol: float) -> object:
    return Pncg(
        criteria=Pncg.ConvergenceCriteria(
            max_steps=100,
            atol_primary=atol,
            rtol_primary=0.0,
            atol_secondary=atol,
            rtol_secondary=0.0,
        )
    )


def check_real_newton_corrections_restart_pncg_and_global_cap_is_visible() -> None:
    state = SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    problem = accelerated_solvers.CachedProblem(CappedQuadratic())
    try:
        adaptive_pncg.adaptive_pncg_newton(
            problem,
            state,
            atol=1e-12,
            make_default_optimizer=capped_optimizer,
            hessian_damping_initial=0.001,
            line_search_armijo=0.25,
            max_step_norm=0.01,
            newton_max_step_norm=0.01,
            pncg_restart_interval=200,
            linear_rtol=1e-12,
            max_newton_steps=100,
            max_pncg_steps=5,
            window_steps=1,
            minimum_reduction=0.1,
            required_poor_windows=2,
        )
    except accelerated_solvers.ForwardConvergenceError as error:
        assert str(error) == "global PNCG iteration budget exhausted"
        receipt = error.receipt
    else:  # pragma: no cover
        raise AssertionError("forced-stall fixture unexpectedly converged")

    assert receipt["pncg_steps"] == 5
    assert receipt["newton_steps"] == 2
    assert len(receipt["corrections"]) == 2
    assert all("linear" in item for item in receipt["corrections"])
    kinds = [item["kind"] for item in receipt["trace"]]
    first_newton = kinds.index("newton")
    assert "pncg" in kinds[first_newton + 1 :]
    assert receipt["interrupted_operation"] is None
    assert float(state.u[0]) < 1.0


if __name__ == "__main__":
    check_no_trigger_matches_ordinary_quartic_pncg()
    check_force_windows_trigger_exactly_and_reset()
    check_real_newton_corrections_restart_pncg_and_global_cap_is_visible()
    print("adaptive PNCG CPU checks passed")
