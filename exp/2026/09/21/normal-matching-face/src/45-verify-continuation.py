"""Audit an Adam continuation without treating a partial continuation as complete."""

# ruff: noqa: PLR0915

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile

from liblaf import cherries

BRANCHES = (
    "smooth-off-l2",
    "smooth-off-normal",
    "smooth-on-l2",
    "smooth-on-normal",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("40-continuation")
    output: Path = Path("45-verification")


def _load_old_verify() -> Any:
    path = Path(__file__).with_name("20-verify.py")
    spec = importlib.util.spec_from_file_location("old20_verify", path)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = _load_old_verify()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _require_digest(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    assert path.exists(), path
    assert _sha256(path) == record["sha256"], path
    return path


def _csv_rows(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="") as stream:
        rows = [
            {
                key: int(value) if key == "step" else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]
    assert rows
    steps = [int(row["step"]) for row in rows]
    assert steps == list(range(len(rows))), steps
    return rows


def _receipt_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    return rows


def _tensor_equal(left: Any, right: Any) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and torch.equal(left, right)
    )


def _array_equal(left: Any, right: Any) -> bool:
    return (
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and np.array_equal(left, right)
    )


def _optimizer_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:  # noqa: PLR0911
    if left.keys() != right.keys():
        return False
    if left["param_groups"] != right["param_groups"]:
        return False
    left_state, right_state = left["state"], right["state"]
    if left_state.keys() != right_state.keys():
        return False
    for key in left_state:
        if left_state[key].keys() != right_state[key].keys():
            return False
        for name in left_state[key]:
            a, b = left_state[key][name], right_state[key][name]
            if isinstance(a, torch.Tensor):
                if not _tensor_equal(a, b):
                    return False
            elif a != b:
                return False
    return True


def _adam_step(parent: dict[str, Any], gradient: np.ndarray) -> np.ndarray:
    optimizer = parent["optimizer"]
    groups = optimizer["param_groups"]
    assert len(groups) == 1
    group = groups[0]
    state = optimizer["state"]
    assert len(state) == 1
    item = next(iter(state.values()))
    q = parent["q"].detach().cpu().numpy()
    grad = np.asarray(gradient, dtype=q.dtype)
    assert grad.shape == q.shape
    step = int(item["step"].item()) + 1
    beta1, beta2 = group["betas"]
    m = item["exp_avg"].detach().cpu().numpy()
    v = item["exp_avg_sq"].detach().cpu().numpy()
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**step)
    v_hat = v / (1 - beta2**step)
    return q - float(group["lr"]) * m_hat / (np.sqrt(v_hat) + float(group["eps"]))


def _branch_audit(
    source: Path,
    parent: Path,
    name: str,
    target_step: int,
    metrics_context: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    details: dict[str, Any] = {}
    parent_folder, folder = parent / name, source / name
    parent_trace, trace = (
        _csv_rows(parent_folder / "trace.csv"),
        _csv_rows(folder / "trace.csv"),
    )
    parent_receipts = _receipt_rows(parent_folder / "solver-receipts.jsonl")
    receipts = _receipt_rows(folder / "solver-receipts.jsonl")
    from_step = len(parent_trace) - 1
    assert from_step == 100
    assert len(parent_receipts) == from_step + 1
    assert trace[: len(parent_trace)] == parent_trace
    assert receipts[: len(parent_receipts)] == parent_receipts
    assert len(trace) == len(receipts)
    last_step = len(trace) - 1
    assert last_step <= target_step
    parent_checkpoints = OLD._checkpoint_steps(parent_folder)  # noqa: SLF001
    checkpoints = OLD._checkpoint_steps(folder)  # noqa: SLF001
    for step, path in parent_checkpoints.items():
        assert _sha256(checkpoints[step]) == _sha256(path)
    required_new_checkpoints = {101} if last_step >= 101 else set()
    required_new_checkpoints.update(range(110, last_step + 1, 10))
    if last_step == target_step:
        required_new_checkpoints.add(last_step)
    assert required_new_checkpoints <= set(checkpoints)
    for receipt in receipts[from_step + 1 :]:
        assert receipt["forward"]["success"] is True
        assert receipt["adjoint"]["success"] is True
    details.update(
        {
            "last_step": last_step,
            "trace_parent_prefix_exact": True,
            "receipt_parent_prefix_exact": True,
            "checkpoint_parent_prefix_exact": True,
            "accepted_continuation_receipts": last_step - from_step,
        }
    )

    parent_state = torch.load(
        parent_folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    start = torch.load(
        folder / "continuation-start.pt", map_location="cpu", weights_only=False
    )
    assert int(parent_state["step"]) == from_step == int(start["step"])
    assert parent_state["branch"] == name == start["branch"]
    assert _tensor_equal(parent_state["q"], start["q"])
    assert _array_equal(parent_state["u"], start["u"])
    assert _optimizer_equal(parent_state["optimizer"], start["optimizer"])
    details["continuation_start_exact_parent"] = True

    replay_paths = (folder / "resume-replay.json", folder / "resume-gradient.npz")
    if all(path.exists() for path in replay_paths):
        replay = _json(replay_paths[0])
        assert replay["passed"] is True
        assert int(replay["step"]) == from_step
        assert replay["optimizer_state"] == {
            "step": from_step,
            "q_equal": True,
            "m_equal": True,
            "v_equal": True,
        }
        assert replay["errors"].keys() == replay["error_limits"].keys()
        for metric, error in replay["errors"].items():
            assert float(error) <= float(replay["error_limits"][metric]), metric
        for receipt_name in ("forward", "adjoint"):
            assert replay[receipt_name]["success"] is True
        details["resume_replay_passed"] = True
        with np.load(replay_paths[1], allow_pickle=False) as state:
            assert int(state["step"]) == from_step
            assert np.array_equal(state["q"], parent_state["q"].detach().cpu().numpy())
            replay_u_error = float(np.max(np.abs(state["u"] - parent_state["u"])))
            assert replay_u_error == float(
                replay["forward_displacement_max_abs_difference_m"]
            )
            assert replay_u_error <= float(
                replay["error_limits"]["forward_displacement_max_abs_difference_m"]
            )
            gradient = np.asarray(state["gradient"])
        details["resume_gradient_q_exact_parent_state"] = True
        details["resume_gradient_u_max_abs_difference_m"] = replay_u_error
        if last_step >= from_step + 1:
            q101, _u101, step101 = OLD._load_state(folder / "step-0101.npz")  # noqa: SLF001
            assert step101 == from_step + 1
            expected_q101 = _adam_step(parent_state, gradient)
            error = float(np.max(np.abs(q101 - expected_q101)))
            assert np.allclose(q101, expected_q101, rtol=2e-12, atol=2e-13), error
            details["first_adam_update_q101_max_abs_error"] = error
    else:
        assert not any(path.exists() for path in replay_paths)
        assert last_step == from_step
        details["replay_missing"] = True

    summary = _json(folder / "summary.json")
    assert {
        "status",
        "last_step",
        "initial_metrics",
        "last_metrics",
        "best_metrics",
        "best_noninverted_metrics",
        "failure",
        "elapsed_seconds",
    } <= summary.keys()
    assert int(summary["last_step"]) == last_step
    assert int(summary["continuation_from_step"]) == from_step
    assert int(summary["continuation_updates"]) == last_step - from_step
    assert "continuation_parent" in summary
    assert "continuation_elapsed_seconds" in summary
    completed = last_step == target_step
    if completed:
        assert summary["status"] == "completed_budget_not_convergence_certified"
        assert summary["failure"] is None
    else:
        assert summary["status"] == "failed_before_budget_completed"
        assert isinstance(summary["failure"], dict)
    details.update(
        {
            "status": summary["status"],
            "failure": summary["failure"],
            "completed_budget": completed,
        }
    )

    last_path = folder / "last.npz"
    q, u, saved_step = OLD._load_state(last_path)  # noqa: SLF001
    assert saved_step == last_step
    latest = torch.load(
        folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    assert int(latest["step"]) == last_step
    assert latest["branch"] == name
    assert np.array_equal(q, latest["q"].detach().cpu().numpy())
    assert np.array_equal(u, latest["u"])
    latest_optimizer_state = next(iter(latest["optimizer"]["state"].values()))
    assert int(latest_optimizer_state["step"]) == last_step
    details["last_state_and_adam_counter_exact"] = True
    metric_arguments = {
        key: value for key, value in metrics_context.items() if key != "metrics_engine"
    }
    computed = OLD._metrics(q, u, **metric_arguments)  # noqa: SLF001
    computed.update(
        metrics_context["metrics_engine"].evaluate_surface(
            u[metrics_context["skin_ids"]]
        )
    )
    trace_metrics = {
        key: float(value) for key, value in trace[last_step].items() if key != "step"
    }
    errors = OLD._check_reported(computed, trace_metrics)  # noqa: SLF001
    errors.update(
        {
            f"summary_last/{key}": value
            for key, value in OLD._check_reported(  # noqa: SLF001
                computed, summary["last_metrics"]
            ).items()
        }
    )
    details["endpoint_metric_absolute_errors"] = errors
    return details, completed


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Any] = {"passed": False, "all_completed": False, "branches": {}}
    try:
        source = cherries.input(cfg.comparison_dir)
        protocol = _json(source / "protocol.json")
        config = protocol["config"]
        continuation = protocol["continuation"]
        assert int(config["steps"]) == int(continuation["to_step"])
        assert int(continuation["from_step"]) == 100
        assert set(config["branches"].split(",")) == set(BRANCHES)
        parent = Path(continuation["parent"])
        assert parent.exists()
        assert (
            _sha256(parent / "protocol.json") == continuation["parent_protocol_sha256"]
        )
        checks["continuation"] = {
            "parent": str(parent),
            "from_step": 100,
            "to_step": int(config["steps"]),
            "parent_protocol_sha256": continuation["parent_protocol_sha256"],
        }
        for key in ("parent_audit", "cpu_gate"):
            _require_digest(continuation[key])
        assert _json(Path(continuation["parent_audit"]["path"]))["passed"] is True
        assert _json(Path(continuation["cpu_gate"]["path"]))["passed"] is True
        if continuation["smoke_gate"] is not None:
            _require_digest(continuation["smoke_gate"])
            smoke = _json(Path(continuation["smoke_gate"]["path"]))
            assert smoke["passed"] is True
            assert smoke["all_completed"] is True
        checks["gates"] = {
            key: continuation[key] for key in ("parent_audit", "cpu_gate", "smoke_gate")
        }
        checks["provenance"] = OLD._check_sources(protocol)  # noqa: SLF001
        assert Path(protocol["sources"]["__main__"]["path"]).name == "40-continue.py"

        with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
            rest = np.asarray(mesh["rest_points"], dtype=np.float64)
            skin_ids = np.asarray(mesh["skin_ids"], dtype=np.int64)
            metrics_context = {
                "rest": rest,
                "skin_ids": skin_ids,
                "target": np.asarray(
                    mesh["target_displacement_skin"], dtype=np.float64
                ),
                "weights": np.asarray(mesh["skin_vertex_weights"], dtype=np.float64),
                "triangles": np.asarray(mesh["triangles"], dtype=np.int64),
                "tets": np.asarray(mesh["tets"], dtype=np.int64),
                "active_ids": np.asarray(mesh["active_ids"], dtype=np.int64),
                "fixed": np.asarray(mesh["fixed_mask"], dtype=bool),
                "fixed_values": np.asarray(mesh["fixed_values"], dtype=np.float64),
                "edge_i": np.asarray(mesh["edge_i"], dtype=np.int64),
                "edge_j": np.asarray(mesh["edge_j"], dtype=np.int64),
                "edge_weight": np.asarray(mesh["edge_weight"], dtype=np.float64),
                "regularizer_factor": float(mesh["regularizer_factor"]),
            }
        metrics_context["metrics_engine"] = OLD.StudyMetrics()
        completed: list[bool] = []
        for name in BRANCHES:
            record = continuation["branches"][name]
            expected_parent_files = {
                "checkpoint": "optimizer-latest.pt",
                "summary": "summary.json",
                "trace": "trace.csv",
                "solver_receipts": "solver-receipts.jsonl",
            }
            for item, filename in expected_parent_files.items():
                _require_digest(record[item])
                assert (
                    Path(record[item]["path"]).resolve()
                    == (parent / name / filename).resolve()
                )
            branch, branch_completed = _branch_audit(
                source, parent, name, int(config["steps"]), metrics_context
            )
            checks["branches"][name] = branch
            completed.append(branch_completed)
        checks["all_completed"] = all(completed)
        checks["passed"] = True
    except Exception as error:  # noqa: BLE001 - preserve an auditable negative result
        checks["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    _write(output / "checks.json", checks)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
