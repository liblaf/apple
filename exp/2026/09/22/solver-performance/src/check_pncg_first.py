# ruff: noqa: EM101, PT018, TRY003
"""CPU contracts for the undamped PNCG-first handoff phase."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
JOINT_SOURCE = HERE.parents[2] / "21/joint-activation-material-mandible/src"
sys.path[:0] = [str(HERE), str(JOINT_SOURCE)]
spec = importlib.util.spec_from_file_location("pncg_first", HERE / "pncg_first.py")
assert spec and spec.loader
phase = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase
spec.loader.exec_module(phase)


class DofMap:
    def to_free(self, value: torch.Tensor) -> torch.Tensor:
        return value


class Model:
    dof_map = DofMap()


class Problem:
    model = Model()

    def __init__(
        self,
        gradients: list[float] | None = None,
        *,
        energy_sign: float = 1.0,
        nan: bool = False,
    ) -> None:
        self.gradients = gradients
        self.energy_sign = energy_sign
        self.nan = nan
        self.updates: list[torch.Tensor] = []

    def update(self, state: SimpleNamespace, value: torch.Tensor) -> None:
        state.u.copy_(value)
        self.updates.append(value.clone())

    def grad(self, _state: SimpleNamespace) -> torch.Tensor:
        if self.nan:
            return torch.tensor([float("nan")], dtype=torch.float64)
        if self.gradients is None:
            return _state.u.clone()
        index = min(len(self.updates), len(self.gradients) - 1)
        return torch.tensor([self.gradients[index]], dtype=torch.float64)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        return self.energy_sign * state.u.square().sum() / 2

    def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
        return torch.ones(1, dtype=torch.float64)

    def hess_quad(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.dot(direction, direction)

    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(0.5, dtype=torch.float64)


def run(problem: Problem, state: SimpleNamespace, **kwargs: object) -> dict:
    old = phase.newton_ccd_fraction
    calls = []
    phase.newton_ccd_fraction = lambda *_args: calls.append(1) or 0.5
    try:
        _, receipt = phase.run_pncg_phase(
            problem, state, atol=1e-12, max_step_norm=0.25, **kwargs
        )
    finally:
        phase.newton_ccd_fraction = old
    receipt["ccd_calls"] = len(calls)
    return receipt


def check_cap_ccd_once_and_energy_rise() -> None:
    class StopAfterOne(Problem):
        def hess_quad(
            self, _state: SimpleNamespace, direction: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.dot(direction, direction)
                if not self.updates
                else torch.zeros((), dtype=torch.float64)
            )

    state = SimpleNamespace(u=torch.tensor([2.0], dtype=torch.float64))
    problem = StopAfterOne(energy_sign=-1.0)
    energies = []
    receipt = run(problem, state, callback=lambda row: energies.append(row["energy"]))
    assert receipt["reason"] == "nonpositive_curvature" and receipt["ccd_calls"] == 1
    # alpha is cap / |p| = .125, then the one CCD fraction .5: x=1.875.
    assert torch.allclose(state.u, torch.tensor([1.875], dtype=torch.float64))
    assert energies[1] > energies[0], "energy increase must not trigger rejection"
    accepted = receipt["trace"][1]
    assert accepted["alpha_edge"] == 0.125
    assert accepted["ccd_fraction"] == 0.5
    assert accepted["coordinate_displacement"] == 0.125
    assert accepted["limiter_reason"] == "coordinate_cap+ccd"


def check_windows_trigger_at_sixty_and_recover() -> None:
    constant = Problem([1.0])
    receipt = run(constant, SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64)))
    assert receipt["reason"] == "stalled" and receipt["steps"] == 60
    assert [item["end_step"] for item in receipt["windows"]] == [20, 40, 60]
    recovered = Problem(
        [100.0] * 20 + [95.0] * 20 + [50.0] * 20 + [48.0] * 20 + [47.0] * 20
    )
    receipt = run(
        recovered, SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    )
    assert receipt["reason"] == "stalled" and receipt["steps"] == 100
    assert receipt["windows"][2]["poor"] is False


def check_no_coarse_ratio_and_failures() -> None:
    class Negative(Problem):
        def hess_quad(
            self, _state: SimpleNamespace, _direction: torch.Tensor
        ) -> torch.Tensor:
            return torch.tensor(0.0, dtype=torch.float64)

    # A force far below a historical relative 1e-3 threshold still enters PNCG.
    receipt = run(
        Negative([1e-6]), SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    )
    assert receipt["reason"] == "nonpositive_curvature" and receipt["steps"] == 0
    try:
        run(
            Problem(nan=True),
            SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64)),
        )
    except phase.ForwardConvergenceError:
        pass
    else:
        raise AssertionError("nonfinite force must fail visibly")


if __name__ == "__main__":
    check_cap_ccd_once_and_energy_rise()
    check_windows_trigger_at_sixty_and_recover()
    check_no_coarse_ratio_and_failures()
    print("pncg-first CPU checks passed")
