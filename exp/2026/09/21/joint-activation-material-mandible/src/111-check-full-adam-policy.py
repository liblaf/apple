"""CPU regression checks for the diagnostic one-step full Adam policy."""

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
from joint_fields import symmetric_coordinates

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/full-adam-policy-check-005"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "full_adam_fitter", Path(__file__).with_name("93-fit-expressions.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counterexample_gradient() -> torch.Tensor:
    matrix = torch.tensor([[0.1, -0.5, -0.1], [-0.5, 1.0, -0.1], [-0.1, -0.1, 1.0]])
    return symmetric_coordinates(matrix).repeat(2, 1)


def fixture(
    module: Any, directory: Path, *, policy: str, fresh: bool
) -> tuple[Any, list[str]]:
    """Build a checkpointable fitter whose sole trial is deliberately uphill."""
    fitter = module.Fitter.__new__(module.Fitter)
    fitter.cfg = SimpleNamespace(
        output_dir=directory,
        pose_collision=True,
        pose_first=False,
        coupled_predictor=False,
        trial_prescreen=False,
        learning_rate=0.003,
        max_backtracks=3,
        armijo=1e-4,
        neighbor_rms_budget=0.05,
        outer_step_policy=policy,
        magnitude_weight=0.0,
        jaw_weight=0.0,
    )
    fitter.names = ("A",)
    fitter.neutral = torch.zeros((2, 3))
    fitter.neutral_seed = fitter.neutral
    fitter.mass = torch.full((2,), 0.5)
    fitter.graph = SimpleNamespace(
        i=torch.tensor([0]), j=torch.tensor([1]), conductance_m=torch.ones(1)
    )
    fitter.runtime = SimpleNamespace(warm_adjoints={})
    fitter.started = time.perf_counter()
    fitter.elapsed_offset = 0.0
    fitter.smooth_weight = 1.0
    fitter.status = {
        "running": True,
        "inverse_converged": False,
        "expressions": {"A": {"status": "queued", "accepted_steps": 0}},
    }
    gradient = counterexample_gradient()
    calls: list[str] = []

    def metrics(value: float) -> dict:
        return {
            "objective": value,
            "data": value,
            "fit_rms_mm": 1.0,
            "weighted_smoothness": 0.0,
            "weighted_magnitude": 0.0,
            "weighted_jaw_prior": 0.0,
            "neighbor_rms": 10.0,
            "activation_tensor_rms_kpa": 0.0,
            "primal_objective_correction_estimate": 1e-3,
            "residual_corrected_objective_estimate": value + 1e-3,
            "stationarity": {"stationary": False},
        }

    def evaluate(
        index: int, _q: torch.Tensor, _jaw: torch.Tensor, *_args: Any, **_kwargs: Any
    ) -> dict:
        calls.append(fitter.names[index])
        return {
            "displacement_m": fitter.neutral.clone(),
            "gradient_q": gradient.clone(),
            "gradient_jaw": torch.zeros(1),
            "metrics": metrics(1.0 if fresh and len(calls) == 1 else 2.0),
        }

    fitter.publish = lambda phase, expression=None: fitter.status.update(
        phase=phase, current_expression=expression
    )
    fitter.evaluate = evaluate
    directory.joinpath("expressions/A").mkdir(parents=True)
    if not fresh:
        q = torch.nn.Parameter(torch.zeros_like(gradient))
        jaw = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam((q, jaw), lr=fitter.cfg.learning_rate)
        q.grad = gradient.clone()
        jaw.grad = torch.zeros(1)
        optimizer.step()  # establish ordinary nonzero Adam moments
        with torch.no_grad():
            q.zero_()
        state = {
            "expression": "A",
            "expression_index": 0,
            "fit_stage": "joint",
            "activation": q.detach(),
            "jaw_normalized": jaw.detach(),
            "gradient_q": gradient,
            "gradient_jaw": torch.zeros(1),
            "displacement_m": fitter.neutral.clone(),
            "accepted_steps": 1,
            "history": [1.0],
            "qualifying_consecutive": 0,
            "inverse_converged": False,
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "outer_step_policy": policy,
            "metrics": metrics(1.0),
        }
        module.atomic_torch(
            directory / "expressions/A/latest.pt", module.cpu_tree(state)
        )
        module.append(
            directory / "expressions/A/trace.jsonl",
            {"accepted_steps": 1, **state["metrics"]},
        )
        fitter.status["expressions"]["A"] = {"status": "fitting", "accepted_steps": 1}
    return fitter, calls


def full_adam_case(module: Any, directory: Path, *, fresh: bool) -> dict:
    fitter, calls = fixture(module, directory, policy="full_adam", fresh=fresh)
    path = directory / "expressions/A/latest.pt"
    expected_q = None
    expected_optimizer = None
    if not fresh:
        before = torch.load(path, map_location="cpu", weights_only=False)
        q = torch.nn.Parameter(before["activation"].clone())
        jaw = torch.nn.Parameter(before["jaw_normalized"].clone())
        expected_optimizer = torch.optim.Adam((q, jaw), lr=fitter.cfg.learning_rate)
        expected_optimizer.load_state_dict(copy.deepcopy(before["optimizer"]))
        q.grad = before["gradient_q"].clone()
        jaw.grad = before["gradient_jaw"].clone()
        expected_optimizer.step()
        module.project_activation_(q, module.CAP)
        with torch.no_grad():
            jaw.clamp_(module.HINGE_MIN, module.HINGE_MAX)
        expected_q = q.detach().clone()
        slope = float(
            (before["gradient_q"] * (expected_q - before["activation"])).sum()
        )
        assert slope >= 0, slope  # would have triggered the Armijo fallback.
    old_neighbor_rms = module.neighbor_rms
    module.neighbor_rms = lambda *_args: 10.0
    try:
        fitter.fit_step(0)
    finally:
        module.neighbor_rms = old_neighbor_rms
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected_calls = 2 if fresh else 1
    assert calls == ["A"] * expected_calls, calls
    assert state["accepted_steps"] == (1 if fresh else 2)
    assert state["outer_step_policy"] == "full_adam"
    assert state["direction_method"] == "full_projected_adam"
    assert not state["momentum_restarted"]
    assert state["accepted_fraction"] == 1.0
    assert state["metrics"]["objective"] == 2.0  # uphill is diagnostic evidence.
    assert state["metrics"]["neighbor_rms"] > fitter.cfg.neighbor_rms_budget
    assert state["metrics"]["primal_objective_correction_estimate"] > 1e-6
    assert state["metrics"]["weighted_magnitude"] == 0.0
    assert state["metrics"]["weighted_jaw_prior"] == 0.0
    assert state["optimizer"]["state"]  # ordinary Adam state was retained.
    if expected_q is not None:
        torch.testing.assert_close(state["activation"], expected_q)
        assert expected_optimizer is not None
        expected_state = expected_optimizer.state_dict()["state"]
        for key, values in expected_state.items():
            for field in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    state["optimizer"]["state"][key][field], values[field]
                )
    rows = [
        json.loads(line)
        for line in (path.parent / "trials.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["accepted"]
    assert rows[0]["fraction"] == 1.0
    return {
        "fresh_checkpoint": fresh,
        "evaluate_calls": len(calls),
        "accepted_uphill": True,
        "nonnegative_projected_adam_slope": not fresh,
        "over_budget_accepted": True,
        "unresolved_primal_accepted": True,
    }


def armijo_case(module: Any, directory: Path) -> dict:
    fitter, calls = fixture(module, directory, policy="armijo", fresh=False)
    path = directory / "expressions/A/latest.pt"
    before = sha256(path)
    fitter.fit_step(0)
    assert sha256(path) == before
    assert calls == ["A"] * fitter.cfg.max_backtracks
    assert fitter.status["expressions"]["A"]["status"] == "line_search_failed"
    return {"transactional_rejection_preserved": True, "evaluations": len(calls)}


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    module = load_runner()
    report = {"schema": "full-adam-policy-cpu-regression-v1", "success": False}
    report["full_adam_checkpoint"] = full_adam_case(
        module, cfg.output_dir / "checkpoint", fresh=False
    )
    report["full_adam_fresh"] = full_adam_case(
        module, cfg.output_dir / "fresh", fresh=True
    )
    report["armijo"] = armijo_case(module, cfg.output_dir / "armijo")
    report["implementation_sha256"] = {
        str(Path(module.__file__).resolve()): sha256(Path(module.__file__)),
        str(Path(__file__).resolve()): sha256(Path(__file__)),
    }
    report["success"] = True
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
