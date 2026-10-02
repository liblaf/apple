"""Acceptance tests for bounded projected-Adam inverse stages."""

from __future__ import annotations

import contextlib
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from activation_models import matrices, project_  # noqa: E402
from run_support import release_zero_axes, run_stage  # noqa: E402
from stress_physics import ForwardConvergenceError  # noqa: E402

from liblaf.apple.inverse import ImplicitNumericalError  # noqa: E402

DTYPE = torch.float64
FORWARD_FAILURE = "scripted forward failure"
ADJOINT_FAILURE = "scripted adjoint failure"
PROGRAMMING_ERROR = "scripted programming error"


class _Physics:
    @contextlib.contextmanager
    def approximate_solves(self):
        yield self

    def check_adjoint(self) -> dict[str, bool]:
        return {"fake": True}


class _Study:
    """Scripted finite objective used without any face or native solve."""

    def __init__(self, outcomes: list[float | str]) -> None:
        self.outcomes = iter(outcomes)
        self.physics = _Physics()

    def evaluate(
        self,
        q: torch.Tensor,
        mode: str,
        axes: torch.Tensor | None,
        seed: np.ndarray,
        normal_weight: float,
        smooth_weight: float,
        *,
        backward: bool,
    ) -> dict:
        del axes, seed, normal_weight, smooth_weight
        outcome = next(self.outcomes)
        if outcome == "forward_failure":
            raise ForwardConvergenceError(FORWARD_FAILURE)
        if outcome == "adjoint_failure":
            raise ImplicitNumericalError(ADJOINT_FAILURE)
        if outcome == "bug":
            raise RuntimeError(PROGRAMMING_ERROR)
        nonfinite = outcome == "nonfinite"
        if nonfinite:
            outcome = 1.0
        assert isinstance(outcome, float)
        q.grad = None
        qhat = matrices(q, mode)
        loss = (qhat[..., 0, 0] - 1.0).square().sum()
        result = {
            "objective": outcome,
            "fit_rms_mm": 0.0,
            "normal_angle_rms_deg": 0.0,
            "activation_smoothness": 0.0,
            "detF_min": 1.0,
            "u": np.array([[outcome]]),
            "forward": {"success": outcome >= 0, "grad_norm": 0.0},
            "adjoint": {"success": outcome >= 0, "absolute_residual": 0.0},
            "solver_valid": outcome >= 0,
            "_loss": loss,
            "_Qhat": qhat,
        }
        if backward:
            self.backward(q, result)
        if nonfinite:
            result["gradient"].fill_(float("nan"))
        return result

    def backward(self, q: torch.Tensor, result: dict) -> None:
        loss, qhat = result.pop("_loss"), result.pop("_Qhat")
        gradient = torch.autograd.grad(loss, qhat)[0].detach()
        q.grad = torch.autograd.grad(qhat, q, grad_outputs=gradient)[0].detach()
        result.update(
            gradient=q.grad.clone(),
            tensor_gradient=(gradient + gradient.mT) / 2,
            gradient_rms=float(q.grad.square().mean().sqrt()),
        )


def _run(tmp_path: Path, name: str, outcomes: list[float | str], *, steps: int):
    return run_stage(
        _Study(outcomes),
        tmp_path / name,
        "symmetric6",
        normal_weight=0.0,
        smooth_weight=0.0,
        Q_initial=torch.zeros((1, 3, 3), dtype=DTYPE),
        seed=np.zeros((1, 1)),
        steps=steps,
        learning_rate=0.1,
    )


def test_release_zero_axes_changes_only_the_chart_and_resets_its_moments() -> None:
    q = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=DTYPE))
    optimizer = torch.optim.Adam([q], lr=0.1)
    q.grad = torch.ones_like(q)
    optimizer.step()
    with torch.no_grad():
        project_(q, "rankone_learned")
        q[0, 0] = 0
    q.grad = None
    result = {
        "tensor_gradient": torch.diag(
            torch.tensor((1.0, -2.0, 1.0), dtype=DTYPE)
        ).unsqueeze(0)
    }

    assert release_zero_axes(q, "rankone_learned", result, optimizer) == 1
    torch.testing.assert_close(
        q[0, 1:].abs(), torch.tensor((0.0, 1.0, 0.0), dtype=DTYPE)
    )
    assert q.grad is not None
    assert q.grad[0, 0] < 0
    for value in optimizer.state[q].values():
        if isinstance(value, torch.Tensor) and value.shape == q.shape:
            torch.testing.assert_close(value, torch.zeros_like(value))


def test_objective_increase_is_an_adam_update_and_best_valid_is_retained(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "increase"
    summary = _run(tmp_path, "increase", [1.0, 0.2, 0.7], steps=2)

    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert summary["optimizer_updates"] == 2
    assert summary["best_valid_step"] == 1
    trace = list(csv.DictReader((folder / "trace.csv").open()))
    assert [float(row["objective"]) for row in trace] == [1.0, 0.2, 0.7]
    assert torch.load(folder / "optimizer-latest.pt", weights_only=False)["optimizer"][
        "state"
    ]
    with (
        np.load(folder / "best-valid.npz") as best,
        np.load(folder / "last.npz") as last,
    ):
        assert int(best["step"]) == 1
        assert int(last["step"]) == 2
        assert not np.array_equal(best["q"], last["q"])


def test_numerical_failures_rollback_moments_then_continue_to_budget(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "recover"
    summary = _run(
        tmp_path,
        "recover",
        [1.0, "forward_failure", "adjoint_failure", 0.5],
        steps=3,
    )

    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert summary["attempted_steps"] == 3
    assert summary["skipped_steps"] == 2
    assert summary["optimizer_updates"] == 1
    latest = torch.load(folder / "optimizer-latest.pt", weights_only=False)
    only_state = next(iter(latest["optimizer"]["state"].values()))
    assert int(only_state["step"]) == 1
    torch.testing.assert_close(
        only_state["exp_avg"][..., 0], torch.tensor([-0.2], dtype=DTYPE)
    )
    assert latest["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.025)


def test_finite_approximate_state_continues_but_never_becomes_best_valid(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "approximate"
    summary = _run(tmp_path, "approximate", [-1.0, -2.0, -3.0], steps=2)

    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert summary["approximate_steps"] == 3
    assert summary["best_valid_step"] is None
    assert (folder / "best-available.npz").is_file()
    assert not (folder / "best-valid.npz").exists()
    with np.load(folder / "last.npz") as state:
        assert not bool(state["solver_valid"])
        assert int(state["step"]) == 2
        assert not np.allclose(state["q"], 0)


def test_initial_numerical_failure_can_recover_on_the_next_attempt(
    tmp_path: Path,
) -> None:
    summary = _run(tmp_path, "initial-recovery", ["forward_failure", 1.0, 0.5], steps=2)

    assert summary["status"] == "completed_budget_not_convergence_certified"
    assert summary["attempted_steps"] == 2
    assert summary["skipped_steps"] == 1
    assert summary["last_step"] == 2
    assert summary["optimizer_updates"] == 1


def test_trailing_numerical_failure_persists_rollback_and_reduced_learning_rate(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "trailing-failure"
    summary = _run(tmp_path, "trailing-failure", [1.0, 0.5, "forward_failure"], steps=2)

    assert summary["attempted_steps"] == 2
    assert summary["last_step"] == 1
    assert summary["skipped_steps"] == 1
    latest = torch.load(folder / "optimizer-latest.pt", weights_only=False)
    assert latest["step"] == 1
    assert int(next(iter(latest["optimizer"]["state"].values()))["step"]) == 1
    assert latest["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.05)


def test_nonfinite_gradient_is_rejected_as_a_numerical_failure(tmp_path: Path) -> None:
    summary = _run(tmp_path, "nonfinite", [1.0, "nonfinite"], steps=1)

    assert summary["skipped_steps"] == 1
    assert summary["last_step"] == 0


def test_programming_errors_write_failure_summary_and_propagate(tmp_path: Path) -> None:
    folder = tmp_path / "bug"

    with pytest.raises(RuntimeError, match=PROGRAMMING_ERROR):
        _run(tmp_path, "bug", ["bug"], steps=0)

    import json

    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["failure"]["type"] == "RuntimeError"


def test_resume_restores_adam_state_without_a_duplicate_update(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    run_stage(
        _Study([1.0, 0.5]),
        parent,
        "symmetric6",
        0.0,
        0.0,
        torch.zeros((1, 3, 3), dtype=DTYPE),
        np.zeros((1, 1)),
        steps=1,
        learning_rate=0.1,
    )
    parent_state = torch.load(parent / "optimizer-latest.pt", weights_only=False)
    resumed = tmp_path / "resumed"
    summary = run_stage(
        _Study([0.4, 0.3]),
        resumed,
        "symmetric6",
        0.0,
        0.0,
        torch.zeros((1, 3, 3), dtype=DTYPE),
        np.zeros((1, 1)),
        steps=2,
        learning_rate=0.5,
        resume_checkpoint=parent / "optimizer-latest.pt",
    )

    assert summary["last_step"] == 2
    assert summary["optimizer_updates"] == 1
    state = torch.load(resumed / "optimizer-latest.pt", weights_only=False)
    assert int(next(iter(state["optimizer"]["state"].values()))["step"]) == 2
    assert state["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.5)
    initialization = json.loads((resumed / "initialization.json").read_text())
    assert initialization["fresh_adam"] is False
    assert initialization["resume_step"] == 1
    assert torch.equal(
        torch.load(parent / "optimizer-latest.pt", weights_only=False)["q"],
        parent_state["q"],
    )
    assert [
        int(row["step"]) for row in csv.DictReader((resumed / "trace.csv").open())
    ] == [1, 2]
