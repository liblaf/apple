"""CPU checks of pose-only initialization followed by joint expression fitting."""

from __future__ import annotations

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
    output_dir: Path = GROUP / "data/pose-first-check-001"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "pose_first_fitter", Path(__file__).with_name("93-fit-expressions.py")
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(
    module: Any,
    directory: Path,
    *,
    target: float = 0.12,
    initial_jaw: float = 0.0,
    reject: bool = False,
    primal_correction: float = 0.0,
) -> tuple[Any, list[dict]]:
    directory.mkdir(parents=True)
    fitter = module.Fitter.__new__(module.Fitter)
    fitter.cfg = SimpleNamespace(
        output_dir=directory,
        pose_collision=True,
        coupled_predictor=False,
        trial_prescreen=False,
        pose_first=True,
        pose_max_step_deg=1.0,
        pose_stationarity_tolerance=1e-3,
        learning_rate=0.003,
        max_backtracks=3,
        armijo=1e-4,
        neighbor_rms_budget=0.05,
        maximum_iterations_per_expression=20,
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
    matrix = torch.tensor([[0.1, -0.5, -0.1], [-0.5, 1.0, -0.1], [-0.1, -0.1, 1.0]])
    base_gq = symmetric_coordinates(matrix).repeat(2, 1)
    calls = []

    def result(q: torch.Tensor, jaw: torch.Tensor, *, rejected: bool = False) -> dict:
        q, jaw = q.detach(), jaw.detach()
        error = jaw[0] - target
        value = (
            1.0
            + 2.0 * float(error.square())
            + float((base_gq * q).sum() + 0.5 * q.square().sum())
        )
        if rejected:
            value = 2.0 + value
        gq, gj = base_gq + q, 4.0 * (jaw - target)
        return {
            "displacement_m": jaw[0].expand(2, 3).clone() + q[:, 0, None],
            "gradient_q": gq.clone(),
            "gradient_jaw": gj.clone(),
            "metrics": {
                "fit_stage": "pose_only",
                "objective": value,
                "data": value,
                "fit_rms_mm": 1.0,
                "weighted_smoothness": 0.0,
                "neighbor_rms": 0.0,
                "activation_tensor_rms_kpa": float(q.norm()),
                "mandible_angle_deg": float(jaw[0]) * 10.0,
                "primal_objective_correction_estimate": primal_correction,
                "residual_corrected_objective_estimate": value + primal_correction,
                "stationarity": module.stationarity(q, jaw, gq, gj, fitter.mass),
            },
        }

    q, jaw = torch.zeros_like(base_gq), torch.tensor([initial_jaw])
    state = {
        "expression": "A",
        "expression_index": 0,
        "fit_stage": "pose_only",
        "activation": q,
        "jaw_normalized": jaw,
        "accepted_steps": 0,
        "pose_accepted_steps": 0,
        "joint_accepted_steps": 0,
        "history": [],
        "qualifying_consecutive": 0,
        "inverse_converged": False,
        **result(q, jaw),
    }
    destination = directory / "expressions/A"
    destination.mkdir(parents=True)
    module.atomic_torch(destination / "latest.pt", state)
    module.append(
        destination / "trace.jsonl",
        {"accepted_steps": 0, "fit_stage": "pose_only", **state["metrics"]},
    )
    fitter.status["expressions"]["A"] = {
        "status": "fitting",
        "fit_stage": "pose_only",
        "accepted_steps": 0,
    }

    def publish(phase: str, expression: str | None = None) -> None:
        fitter.status.update(phase=phase, current_expression=expression)

    def evaluate(
        index: int,
        active: torch.Tensor,
        jvalue: torch.Tensor,
        _seed: torch.Tensor,
        _seed_jaw: torch.Tensor,
    ) -> dict:
        name = fitter.names[index]
        calls.append(
            {
                "expression": name,
                "activation": active.detach().clone(),
                "jaw": jvalue.detach().clone(),
            }
        )
        fitter.runtime.warm_adjoints[name] = torch.tensor([42.0, 43.0])
        return result(active, jvalue, rejected=reject)

    fitter.publish, fitter.evaluate = publish, evaluate
    fitter.refine_neutral = lambda: None
    fitter.calibrate = lambda: None
    return fitter, calls


def saved(directory: Path) -> dict:
    return torch.load(
        directory / "expressions/A/latest.pt", map_location="cpu", weights_only=False
    )


def physical_equal(left: dict, right: dict) -> None:
    for name in (
        "activation",
        "jaw_normalized",
        "displacement_m",
        "gradient_q",
        "gradient_jaw",
    ):
        torch.testing.assert_close(left[name], right[name], atol=0.0, rtol=0.0)


def stages_case(module: Any, directory: Path) -> dict:
    fitter, calls = fixture(module, directory)
    start = saved(directory)
    assert not start["metrics"]["stationarity"]["stationary"]
    fitter.fit_step(0)
    first = saved(directory)
    assert torch.count_nonzero(first["activation"]) == 0
    assert abs(float(first["jaw_normalized"][0]) - 0.1) < 1e-14
    assert first["accepted_steps"] == first["pose_accepted_steps"] == 1
    assert first["joint_accepted_steps"] == 0
    assert first["metrics"]["objective"] < start["metrics"]["objective"]
    direction, curvature = module.pose_only_direction(first, 1.0)
    assert abs(curvature - 0.25) < 1e-14
    assert abs(direction - 0.02) < 1e-14
    fitter.fit_step(0)
    second = saved(directory)
    assert torch.count_nonzero(second["activation"]) == 0
    assert abs(float(second["jaw_normalized"][0]) - 0.12) < 1e-14
    assert second["metrics"]["objective"] < first["metrics"]["objective"]
    assert second["accepted_steps"] == second["pose_accepted_steps"] == 2
    # Deliberately stale metadata must not leak across the stage boundary.
    second.update(
        optimizer={"sentinel": True}, history=[999.0], qualifying_consecutive=4
    )
    module.atomic_torch(directory / "expressions/A/latest.pt", second)
    count = len(calls)
    fitter.fit_step(0)
    joint = saved(directory)
    final_path = directory / "expressions/A/pose-only-final.pt"
    pose_final = torch.load(final_path, weights_only=False)
    joint_initial = torch.load(
        directory / "expressions/A/joint-initial.pt", weights_only=False
    )
    for state in (pose_final, joint_initial, joint):
        physical_equal(second, state)
    assert len(calls) == count
    assert joint["fit_stage"] == "joint"
    assert joint["pose_converged"]
    assert joint["accepted_steps"] == 2
    assert joint["joint_accepted_steps"] == 0
    assert not joint["inverse_converged"]
    assert joint["history"] == []
    assert joint["qualifying_consecutive"] == 0
    assert "optimizer" not in joint
    assert "pose_previous" not in joint
    certificate = json.loads((directory / "expressions/A/pose-stage.json").read_text())
    assert certificate["pose_converged"]
    assert certificate["checkpoint_sha256"] == sha256(final_path)
    fitter.fit_step(0)
    active = saved(directory)
    assert torch.count_nonzero(active["activation"]) > 0
    assert active["accepted_steps"] == 3
    assert active["joint_accepted_steps"] == 1
    assert active["pose_accepted_steps"] == 2
    assert active["metrics"]["objective"] < joint["metrics"]["objective"]
    assert all(call["expression"] == "A" for call in calls)
    assert all(torch.count_nonzero(call["activation"]) == 0 for call in calls[:count])
    return {
        "pose_angles_deg": [
            float(state["jaw_normalized"][0]) * 10 for state in (start, first, second)
        ],
        "secant_inverse_curvature": curvature,
        "pose_objectives": [
            state["metrics"]["objective"] for state in (start, first, second)
        ],
        "activation_exactly_zero_through_pose": True,
        "transition_preserves_physical_state_and_full_gradient": True,
        "transition_resets_optimizer_without_accepted_step": True,
        "joint_step_unlocks_activation": True,
        "accepted_counts": {
            key: active[key]
            for key in ("accepted_steps", "pose_accepted_steps", "joint_accepted_steps")
        },
    }


def rejection_case(module: Any, directory: Path) -> dict:
    fitter, calls = fixture(module, directory, reject=True)
    path = directory / "expressions/A/latest.pt"
    digest = sha256(path)
    adjoint = fitter.runtime.warm_adjoints["A"].clone()
    fitter.fit_step(0)
    assert len(calls) == fitter.cfg.max_backtracks
    assert sha256(path) == digest
    torch.testing.assert_close(
        adjoint, fitter.runtime.warm_adjoints["A"], atol=0, rtol=0
    )
    assert fitter.status["expressions"]["A"]["status"] == "line_search_failed"
    assert not (path.parent / "pose-stage.json").exists()
    assert not (path.parent / "joint-initial.pt").exists()
    assert all(torch.count_nonzero(call["activation"]) == 0 for call in calls)
    return {
        "checkpoint_bytes_unchanged": True,
        "adjoint_restored": True,
        "no_joint_transition": True,
        "rejected_trials": len(calls),
    }


def boundary_case(
    module: Any, directory: Path, *, upper: bool = False, unresolved: bool = False
) -> dict:
    fitter, calls = fixture(
        module,
        directory,
        initial_jaw=4.0 if upper else 0.0,
        target=5.0 if upper else -0.2,
        primal_correction=1e-3 if unresolved else 0.0,
    )
    path = directory / "expressions/A/latest.pt"
    digest = sha256(path)
    before = saved(directory)
    assert not before["metrics"]["stationarity"]["stationary"]
    fitter.fit_step(0)
    after = saved(directory)
    assert not calls
    physical_equal(before, after)
    assert after["accepted_steps"] == 0
    if unresolved:
        assert sha256(path) == digest
        assert (
            fitter.status["expressions"]["A"]["status"]
            == "stationary_primal_unresolved"
        )
        assert after["fit_stage"] == "pose_only"
        assert not (path.parent / "pose-stage.json").exists()
    else:
        assert after["fit_stage"] == "joint"
        assert after["pose_converged"]
        assert not after["inverse_converged"]
    return {
        "bound": "upper" if upper else "lower",
        "primal_unresolved": unresolved,
        "no_trial_evaluation_or_accepted_step": True,
        "fit_stage": after["fit_stage"],
    }


def schedule_case(module: Any, directory: Path) -> dict:
    fitter, calls = fixture(module, directory, target=0.35)
    fitter.cfg.maximum_iterations_per_expression = 2
    fitter.run()
    state = saved(directory)
    assert [call["expression"] for call in calls] == ["A", "A"]
    assert state["accepted_steps"] == state["pose_accepted_steps"] == 2
    assert state["fit_stage"] == "pose_only"
    assert fitter.status["phase"] == "expression_iteration_budget_reached"
    assert not fitter.status["running"]
    assert not fitter.status["inverse_converged"]
    assert fitter.status["expressions"]["B"]["status"] == "queued"
    assert not (directory / "expressions/B").exists()
    assert all(torch.count_nonzero(call["activation"]) == 0 for call in calls)
    return {
        "evaluated_expressions": [call["expression"] for call in calls],
        "phase": fitter.status["phase"],
        "accepted_pose_steps": state["pose_accepted_steps"],
        "budget_stops_without_advancing_or_unlocking_activation": True,
    }


def initialization_case(module: Any, directory: Path) -> dict:
    fitter, calls = fixture(module, directory)
    (directory / "expressions/A/latest.pt").unlink()
    fitter.fit_step(0)
    initial = torch.load(directory / "expressions/A/initial.pt", weights_only=False)
    assert initial["fit_stage"] == initial["metrics"]["fit_stage"] == "pose_only"
    assert (
        initial["accepted_steps"]
        == initial["pose_accepted_steps"]
        == initial["joint_accepted_steps"]
        == 0
    )
    assert torch.count_nonzero(initial["activation"]) == 0
    assert len(calls) == 2
    assert saved(directory)["pose_accepted_steps"] == 1
    return {
        "fresh_expression_starts_pose_only": True,
        "first_update_retains_zero_activation": True,
    }


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    module = load_runner()
    report = {
        "schema": "pose-first-cpu-regression-v1",
        "success": False,
        "cpu_only": True,
        "implementation_sha256": {
            str(Path(module.__file__).resolve()): sha256(Path(module.__file__)),
            str(Path(__file__).resolve()): sha256(Path(__file__)),
        },
    }
    write_json(cfg.output_dir / "summary.json", report)
    report["stages"] = stages_case(module, cfg.output_dir / "stages")
    report["rejection"] = rejection_case(module, cfg.output_dir / "rejection")
    report["lower_boundary"] = boundary_case(module, cfg.output_dir / "lower-boundary")
    report["upper_boundary"] = boundary_case(
        module, cfg.output_dir / "upper-boundary", upper=True
    )
    report["unresolved_primal"] = boundary_case(
        module, cfg.output_dir / "unresolved", unresolved=True
    )
    report["sequential_budget"] = schedule_case(module, cfg.output_dir / "schedule")
    report["initialization"] = initialization_case(
        module, cfg.output_dir / "initialization"
    )
    for path, digest in report["implementation_sha256"].items():
        assert sha256(Path(path)) == digest
    report["success"] = True
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
