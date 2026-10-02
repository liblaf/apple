"""CPU regression checks for feasible descent and transactional fit updates."""

from __future__ import annotations

import copy
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_fields import symmetric_coordinates, symmetric_matrices

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/projected-descent-check-001"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "projected_descent_fitter", Path(__file__).with_name("93-fit-expressions.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counterexample_gradient() -> torch.Tensor:
    matrix = torch.tensor([[0.1, -0.5, -0.1], [-0.5, 1.0, -0.1], [-0.1, -0.1, 1.0]])
    return symmetric_coordinates(matrix).repeat(2, 1)


def old_projected_adam(module: Any) -> dict:
    gradient = counterexample_gradient()
    q = torch.nn.Parameter(torch.zeros_like(gradient))
    optimizer = torch.optim.Adam((q,), lr=0.003)
    q.grad = gradient.clone()
    optimizer.step()
    module.project_activation_(q, module.CAP)
    slope = float((gradient * q.detach()).sum())
    residual = module.stationarity(
        torch.zeros_like(q),
        torch.zeros(1),
        gradient,
        torch.zeros(1),
        torch.full((2,), 0.5),
    )
    assert slope > 0, slope
    assert not residual["stationary"]
    return {
        "slope": slope,
        "stationarity": residual,
        "projected_eigenvalues": torch.linalg.eigvalsh(
            symmetric_matrices(q.detach())
        ).tolist(),
    }


def fixture(
    module: Any,
    directory: Path,
    *,
    reject: bool = False,
    gradient: torch.Tensor | None = None,
    jaw_gradient: float = 0.0,
) -> tuple[Any, dict, list[str]]:
    """Construct real cached optimizer/checkpoint state, with no physics object."""
    directory.mkdir(parents=True)
    fitter = module.Fitter.__new__(module.Fitter)
    fitter.cfg = SimpleNamespace(
        output_dir=directory,
        pose_collision=True,
        coupled_predictor=False,
        trial_prescreen=False,
        learning_rate=0.003,
        max_backtracks=3,
        armijo=1e-4,
        neighbor_rms_budget=0.05,
        maximum_iterations_per_expression=3,
        wall_cap_seconds=60.0,
    )
    fitter.names = ("A", "B")
    fitter.neutral = torch.zeros((2, 3))
    fitter.neutral_seed = fitter.neutral
    fitter.mass = torch.full((2,), 0.5)
    fitter.graph = SimpleNamespace(
        i=torch.tensor([0]), j=torch.tensor([1]), conductance_m=torch.ones(1)
    )
    fitter.runtime = SimpleNamespace(warm_adjoints={"A": torch.tensor([0.125, 0.25])})
    fitter.started = time.perf_counter()
    fitter.elapsed_offset = 0.0
    fitter.smooth_weight = 1.0
    fitter.status = {
        "running": True,
        "inverse_converged": False,
        "expressions": {
            name: {"status": "queued", "accepted_steps": 0} for name in fitter.names
        },
    }
    gq = counterexample_gradient() if gradient is None else gradient.clone()
    gj = torch.tensor([jaw_gradient])
    q = torch.nn.Parameter(torch.zeros_like(gq))
    jaw = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam((q, jaw), lr=fitter.cfg.learning_rate)
    q.grad = gq.clone()
    jaw.grad = gj.clone()
    # Populate actual Adam moments/step while the fixture's accepted point is zero.
    if bool(torch.isfinite(gq).all() and torch.isfinite(gj).all()):
        optimizer.step()
    with torch.no_grad():
        q.zero_()
        jaw.zero_()
    calls = []

    def metrics(value: float, qvalue: torch.Tensor, jvalue: torch.Tensor) -> dict:
        stationarity = (
            module.stationarity(qvalue, jvalue, gq, gj, fitter.mass)
            if bool(torch.isfinite(gq).all() and torch.isfinite(gj).all())
            else {"stationary": False}
        )
        return {
            "objective": value,
            "data": value,
            "fit_rms_mm": 1.0,
            "weighted_smoothness": 0.0,
            "neighbor_rms": 0.0,
            "activation_tensor_rms_kpa": 0.0,
            "primal_objective_correction_estimate": 0.0,
            "residual_corrected_objective_estimate": value,
            "stationarity": stationarity,
        }

    state = {
        "expression": "A",
        "expression_index": 0,
        "activation": q.detach().clone(),
        "jaw_normalized": jaw.detach().clone(),
        "gradient_q": gq,
        "gradient_jaw": gj,
        "displacement_m": fitter.neutral.clone(),
        "accepted_steps": 1,
        "history": [1.0],
        "qualifying_consecutive": 0,
        "inverse_converged": False,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "metrics": metrics(1.0, q.detach(), jaw.detach()),
    }
    expression_dir = directory / "expressions/A"
    expression_dir.mkdir(parents=True)
    module.atomic_torch(expression_dir / "latest.pt", module.cpu_tree(state))
    module.append(
        expression_dir / "trace.jsonl", {"accepted_steps": 1, **state["metrics"]}
    )
    fitter.status["expressions"]["A"] = {"status": "fitting", "accepted_steps": 1}

    def publish(phase: str, expression: str | None = None) -> None:
        fitter.status.update(phase=phase, current_expression=expression)

    def evaluate(
        index: int,
        qvalue: torch.Tensor,
        jvalue: torch.Tensor,
        _seed: torch.Tensor,
        _seed_jaw: torch.Tensor,
    ) -> dict:
        name = fitter.names[index]
        calls.append(name)
        fitter.runtime.warm_adjoints[name] = torch.tensor([42.0, 43.0])
        value = (
            2.0
            if reject
            else 1.0
            + float((gq * qvalue.detach()).sum() + (gj * jvalue.detach()).sum())
        )
        return {
            "displacement_m": fitter.neutral.clone(),
            "gradient_q": gq.clone(),
            "gradient_jaw": gj.clone(),
            "metrics": metrics(value, qvalue.detach(), jvalue.detach()),
        }

    fitter.publish = publish
    fitter.evaluate = evaluate
    fitter.refine_neutral = lambda: None
    fitter.calibrate = lambda: None
    return fitter, state, calls


def assert_feasible(module: Any, q: torch.Tensor, jaw: torch.Tensor) -> None:
    eigenvalues = torch.linalg.eigvalsh(symmetric_matrices(q))
    assert float(eigenvalues.min()) >= -1e-12
    assert float(eigenvalues.max()) <= module.CAP + 1e-12
    assert bool((jaw >= module.HINGE_MIN).all())
    assert bool((jaw <= module.HINGE_MAX).all())


def direction_case(module: Any) -> dict:
    gq = counterexample_gradient()
    q, jaw = torch.zeros_like(gq), torch.zeros(1)
    dq, dj, slope, scale = module.projected_gradient_direction(
        q, jaw, gq, torch.zeros(1), torch.full((2,), 0.5), 0.003
    )
    assert_feasible(module, q + dq, jaw + dj)
    assert float(slope) < 0
    assert float(scale) > 0
    assert float(dq.abs().max()) <= 0.003 + 1e-14
    assert abs(float(slope) - float((gq * dq).sum())) <= 1e-14
    return {
        "slope": float(slope),
        "scale": float(scale),
        "maximum_coordinate_step": float(dq.abs().max()),
        "feasible": True,
    }


def accepted_case(module: Any, directory: Path) -> dict:
    fitter, _, calls = fixture(module, directory)
    path = directory / "expressions/A/latest.pt"
    fitter.fit_step(0)
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["accepted_steps"] == 2
    assert state["direction_method"] == "volume_metric_projected_gradient"
    assert state["momentum_restarted"]
    assert state["optimizer"]["state"] == {}
    assert_feasible(module, state["activation"], state["jaw_normalized"])
    rows = [
        json.loads(line)
        for line in (path.parent / "trace.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["direction_method"] == state["direction_method"]
    assert rows[-1]["objective"] < 1.0
    # Continue the real sequential scheduler: one more A update reaches its cap.
    fitter.run()
    assert calls == ["A", "A"], calls
    assert fitter.status["phase"] == "expression_iteration_budget_reached"
    assert fitter.status["expressions"]["A"]["accepted_steps"] == 3
    assert fitter.status["expressions"]["B"]["status"] == "queued"
    assert not fitter.status["inverse_converged"]
    return {
        "method": state["direction_method"],
        "accepted_objective": rows[-1]["objective"],
        "adam_moments_reset": True,
        "sequential_calls": calls,
        "terminal_phase": fitter.status["phase"],
    }


def rejected_case(module: Any, directory: Path) -> dict:
    fitter, state, calls = fixture(module, directory, reject=True)
    path = directory / "expressions/A/latest.pt"
    before = sha256(path)
    old_adjoint = fitter.runtime.warm_adjoints["A"].clone()
    fitter.fit_step(0)
    assert sha256(path) == before
    assert fitter.status["expressions"]["A"]["status"] == "line_search_failed"
    assert fitter.status["expressions"]["A"]["accepted_steps"] == 1
    assert calls == ["A"] * fitter.cfg.max_backtracks
    torch.testing.assert_close(fitter.runtime.warm_adjoints["A"], old_adjoint)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in state["optimizer"]["state"]:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                saved["optimizer"]["state"][key][field],
                state["optimizer"]["state"][key][field],
            )
    trace = (path.parent / "trace.jsonl").read_text().splitlines()
    assert len(trace) == 1
    return {
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": sha256(path),
        "adam_state_and_accepted_state_unchanged": True,
        "warm_adjoint_restored": True,
        "rejected_trials": len(calls),
    }


def stationary_case(module: Any, directory: Path, *, unresolved: bool) -> dict:
    gradient = symmetric_coordinates(torch.eye(3)).repeat(2, 1)
    fitter, state, calls = fixture(
        module, directory, gradient=gradient, jaw_gradient=1.0
    )
    path = directory / "expressions/A/latest.pt"
    if unresolved:
        state["metrics"]["primal_objective_correction_estimate"] = 1e-3
        state["metrics"]["residual_corrected_objective_estimate"] = 1.001
        module.atomic_torch(path, state)
    assert state["metrics"]["stationarity"]["stationary"]
    fitter.fit_step(0)
    expected = "stationary_primal_unresolved" if unresolved else "converged"
    assert fitter.status["expressions"]["A"]["status"] == expected
    assert fitter.status["expressions"]["A"]["accepted_steps"] == 1
    assert calls == []
    saved = torch.load(path, map_location="cpu", weights_only=False)
    torch.testing.assert_close(saved["activation"], state["activation"])
    torch.testing.assert_close(saved["jaw_normalized"], state["jaw_normalized"])
    assert saved["accepted_steps"] == 1
    return {
        "status": expected,
        "new_accepted_steps": 0,
        "no_trial_evaluation": True,
        "jaw_lower_bound": 0.0,
    }


def nonfinite_case(module: Any, directory: Path) -> dict:
    failures = []
    for name, value in (("nan", float("nan")), ("positive_infinity", float("inf"))):
        gradient = counterexample_gradient()
        gradient[0, 0] = value
        try:
            module.projected_gradient_direction(
                torch.zeros_like(gradient),
                torch.zeros(1),
                gradient,
                torch.zeros(1),
                torch.full((2,), 0.5),
                0.003,
            )
        except AssertionError:
            pass
        else:
            message = "Nonfinite direction input was accepted"
            raise AssertionError(message)
        fitter, _, calls = fixture(module, directory / name, gradient=gradient)
        path = directory / name / "expressions/A/latest.pt"
        before = sha256(path)
        try:
            fitter.fit_step(0)
        except AssertionError:
            pass
        else:
            message = "Nonfinite cached gradient was accepted"
            raise AssertionError(message)
        assert calls == []
        assert sha256(path) == before
        failures.append(name)
    return {
        "visibly_rejected": failures,
        "checkpoint_unchanged": True,
        "no_trial_evaluation": True,
    }


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    module = load_runner()
    report = {
        "schema": "projected-descent-cpu-regression-v1",
        "success": False,
        "cpu_only": True,
        "implementation_sha256": {
            str(Path(module.__file__).resolve()): sha256(Path(module.__file__)),
            str(Path(__file__).resolve()): sha256(Path(__file__)),
        },
    }
    write_json(cfg.output_dir / "summary.json", report)
    report["old_adam_counterexample"] = old_projected_adam(module)
    report["corrected_direction"] = direction_case(module)
    report["accepted_and_sequential"] = accepted_case(
        module, cfg.output_dir / "accepted"
    )
    report["transactional_rejection"] = rejected_case(
        module, cfg.output_dir / "rejected"
    )
    report["constrained_stationary"] = stationary_case(
        module, cfg.output_dir / "stationary", unresolved=False
    )
    report["stationary_primal_unresolved"] = stationary_case(
        module, cfg.output_dir / "unresolved", unresolved=True
    )
    report["nonfinite_gradients"] = nonfinite_case(module, cfg.output_dir / "nonfinite")
    for path, digest in report["implementation_sha256"].items():
        assert sha256(Path(path)) == digest
    report["success"] = True
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
