"""CPU behavior checks for one-way hybrid PNCG handoff routing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path.insert(0, str(SOURCE_GROUP / "src"))

source = HERE / "hybrid_handoff.py"
spec = importlib.util.spec_from_file_location("hybrid_handoff", source)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class FakePncg:
    class HessianDamping:
        def __init__(self, *, initial: float) -> None:
            self.initial = initial

    values: list[float] = []
    init_calls = 0
    step_calls = 0

    def __init__(self, **_: object) -> None:
        pass

    def init(
        self, _: object, state: SimpleNamespace, __: torch.Tensor
    ) -> SimpleNamespace:
        type(self).init_calls += 1
        return SimpleNamespace(fun=torch.tensor(float(state.force)))

    def step(
        self, _: object, state: SimpleNamespace, opt_state: SimpleNamespace
    ) -> None:
        type(self).step_calls += 1
        state.force = type(self).values.pop(0)
        opt_state.fun = torch.tensor(float(state.force))

    def terminate(self, *_: object) -> tuple[bool, str]:
        return False, "continue"


class FakeProblem:
    def __init__(self, force: float) -> None:
        self.model = SimpleNamespace(
            dof_map=SimpleNamespace(to_free=lambda tensor: tensor)
        )
        self.budget_checks = 0
        self.state = SimpleNamespace(u=torch.zeros(1), force=force)

    def check_budget(self) -> None:
        self.budget_checks += 1

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        return torch.tensor([state.force])


def run(
    values: list[float], *, initial: float, **kwargs: object
) -> tuple[SimpleNamespace, dict]:
    FakePncg.values = list(values)
    FakePncg.init_calls = 0
    FakePncg.step_calls = 0
    problem = FakeProblem(initial)
    return module.run_pncg_warmup(
        problem,
        problem.state,
        initial_force=initial,
        atol=1e-6,
        make_default_optimizer=lambda _: SimpleNamespace(criteria=object()),
        hessian_damping_initial=0.0,
        line_search_armijo=1e-4,
        max_step_norm=1.0,
        pncg_restart_interval=10,
        **kwargs,
    )


def main() -> None:
    original = module.AcceptedForcePncg
    module.AcceptedForcePncg = FakePncg
    try:
        _, threshold = run([], initial=5e-7)
        assert threshold["handoff_reason"] == "force_threshold"
        assert threshold["coarse_steps"] == 0
        assert FakePncg.init_calls == FakePncg.step_calls == 0

        _, stalled = run([0.95] * 40, initial=1.0)
        assert stalled["handoff_reason"] == "stalled"
        assert stalled["coarse_steps"] == FakePncg.step_calls == 40
        assert len(stalled["windows"]) == 2
        assert abs(stalled["coarse_terminal_force"] - 0.95) < 1e-7

        _, capped = run([0.99] * 5, initial=1.0, max_pncg_steps=5)
        assert capped["handoff_reason"] == "pncg_step_cap"
        assert capped["coarse_steps"] == FakePncg.step_calls == 5
        assert capped["windows"] == []

        try:
            run([float("nan")], initial=1.0, max_pncg_steps=1)
        except module.ForwardConvergenceError as error:
            assert str(error) == "nonfinite accepted-state PNCG force"
        else:
            raise AssertionError("nonfinite accepted state must fail visibly")
    finally:
        module.AcceptedForcePncg = original
    print("hybrid_handoff CPU checks passed")


if __name__ == "__main__":
    main()
