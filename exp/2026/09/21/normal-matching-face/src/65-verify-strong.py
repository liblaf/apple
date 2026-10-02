"""Independent CPU audit of the fresh beta=0.25 normal-loss fits."""

from __future__ import annotations

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

BRANCHES = ("smooth-off-normal", "smooth-on-normal")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("60-strong-normal")
    output: Path = Path("65-verification")
    baseline_dir: Path = Path("40-continuation")


def _old() -> Any:
    spec = importlib.util.spec_from_file_location(
        "old20", Path(__file__).with_name("20-verify.py")
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OLD = _old()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def initial(folder: Path, branch: str, beta: float, coefficient: float) -> None:
    with np.load(folder / "initial-state.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 0
        assert bool(state["activation_identity"])
        assert bool(state["adjoint_initial_guess_zero"])
        for key in ("q", "u", "m", "v"):
            assert not np.count_nonzero(state[key]), key
    update = read(folder / "initial-update.json")
    assert float(update["beta"]) == beta
    assert float(update["smooth_coefficient"]) == coefficient
    with np.load(folder / "initial-gradient.npz", allow_pickle=False) as state:
        gradient, delta = state["gradient"], state["adam_delta"]
        expected = -0.3 * gradient / (np.abs(gradient) + 0.01)
        assert np.array_equal(delta, expected)
    saved = torch.load(
        folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    assert saved["branch"] == branch
    assert int(saved["step"]) > 0
    item = next(iter(saved["optimizer"]["state"].values()))
    assert int(item["step"]) == int(saved["step"])


def context(source: Path) -> dict[str, Any]:
    with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
        return {
            "rest": np.asarray(mesh["rest_points"], dtype=np.float64),
            "skin_ids": np.asarray(mesh["skin_ids"], dtype=np.int64),
            "target": np.asarray(mesh["target_displacement_skin"], dtype=np.float64),
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


def hash_record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def main(cfg: Config) -> None:  # noqa: PLR0915
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Any] = {"passed": False, "all_completed": False, "branches": {}}
    try:
        source, baseline = (
            cherries.input(cfg.comparison_dir),
            cherries.input(cfg.baseline_dir),
        )
        validation = cherries.input(Path("56-strong-validation"))
        protocol, old_protocol = (
            read(source / "protocol.json"),
            read(baseline / "protocol.json"),
        )
        config = protocol["config"]
        assert float(config["beta"]) == 0.25
        assert set(config["branches"].split(",")) == set(BRANCHES)
        assert int(config["steps"]) >= 1
        for key in (
            "learning_rate",
            "adam_eps",
            "smooth_coefficient",
            "smooth_length_m",
            "checkpoint_interval",
        ):
            assert config[key] == old_protocol["config"][key]
        assert protocol["normalization"] == old_protocol["normalization"]
        assert protocol["materials"] == old_protocol["materials"]
        assert protocol["forward_tolerance"] == old_protocol["forward_tolerance"]
        assert protocol["fixture"] == old_protocol["fixture"]
        baseline_sources = {
            str(Path(details["path"]).resolve()): details["sha256"]
            for details in old_protocol["sources"].values()
        }
        for details in protocol["sources"].values():
            assert (
                details["sha256"]
                == baseline_sources[str(Path(details["path"]).resolve())]
            )
        baseline_audit = baseline.parents[0] / "45-verification" / "checks.json"
        assert read(baseline_audit)["passed"] is True
        assert read(baseline_audit)["all_completed"] is True
        baseline_records = {
            "protocol": hash_record(baseline / "protocol.json"),
            "audit": hash_record(baseline_audit),
            "branches": {},
        }
        for branch in (
            "smooth-off-l2",
            "smooth-off-normal",
            "smooth-on-l2",
            "smooth-on-normal",
        ):
            baseline_records["branches"][branch] = {
                "trace": hash_record(baseline / branch / "trace.csv"),
                "endpoint": hash_record(baseline / branch / "last.npz"),
            }
        checks["baseline"] = baseline_records
        checks["provenance"] = OLD._check_sources(protocol)  # noqa: SLF001
        validation_protocol = read(validation / "protocol.json")
        assert float(validation_protocol["config"]["beta"]) == 0.25
        assert validation_protocol["fixture"] == protocol["fixture"]
        validation_source_hashes = {
            str(Path(details["path"]).resolve()): details["sha256"]
            for details in validation_protocol["sources"].values()
        }
        assert validation_source_hashes == {
            str(Path(details["path"]).resolve()): details["sha256"]
            for details in protocol["sources"].values()
        }
        assert digest(source / "gradient-validation.json") == digest(
            validation / "gradient-validation.json"
        )
        assert read(source / "gradient-validation.json")["status"] == "passed"
        assert read(source / "normal-validation.json")["passed"] is True
        checks["gates"] = {
            "gradient": hash_record(validation / "gradient-validation.json"),
            "normal": True,
        }
        if (source / "strong-preflight.json").exists():
            preflight = read(source / "strong-preflight.json")
            for label, record in preflight.items():
                assert Path(record["path"]).exists(), label
                assert digest(Path(record["path"])) == record["sha256"], label
            smoke_protocol = read(
                Path(preflight["data/59-strong-smoke/protocol.json"]["path"])
            )
            smoke_checks = read(
                Path(preflight["data/59-strong-verification/checks.json"]["path"])
            )
            assert smoke_checks["passed"] is True
            assert smoke_checks["all_completed"] is True
            assert float(smoke_protocol["config"]["beta"]) == 0.25
            assert {
                str(Path(item["path"]).resolve()): item["sha256"]
                for item in smoke_protocol["sources"].values()
            } == {
                str(Path(item["path"]).resolve()): item["sha256"]
                for item in protocol["sources"].values()
            }
            assert (
                read(Path(preflight["data/45-verification/checks.json"]["path"]))[
                    "passed"
                ]
                is True
            )
            checks["strong_preflight"] = {
                "records_checked": len(preflight),
                "smoke_passed": True,
            }
        checks["smooth_off_coverage"] = {
            "raw6_regularizer_fd_relative_error": read(
                source.parent / "05-normal-verification" / "checks.json"
            )["raw6_regularizer_derivative_relative_error"],
            "reason": "R(q) is direct q-only; smooth-off sets its coefficient to zero, so the combined beta=0.25 normal-plus-L2 derivative is the independently gated derivative with lambda*dR/dq removed.",
        }
        values = context(source)
        engine = OLD.StudyMetrics()
        done = []
        for branch in BRANCHES:
            folder = source / branch
            coefficient = (
                float(config["smooth_coefficient"])
                if branch.startswith("smooth-on")
                else 0.0
            )
            initial(folder, branch, 0.25, coefficient)
            trace = OLD._read_trace(folder / "trace.csv")  # noqa: SLF001
            last_step = max(trace)
            assert list(trace) == list(range(last_step + 1))
            receipts = OLD._check_solver_receipts(  # noqa: SLF001
                folder / "solver-receipts.jsonl", last_step
            )
            summary = read(folder / "summary.json")
            assert int(summary["last_step"]) == last_step
            completed = last_step == int(config["steps"])
            if completed:
                assert summary["status"] == "completed_budget_not_convergence_certified"
                assert summary["failure"] is None
            else:
                assert summary["status"] == "failed_before_budget_completed"
                assert isinstance(summary["failure"], dict)
            q, u, step = OLD._load_state(folder / "last.npz")  # noqa: SLF001
            assert step == last_step
            q1_error: float | None = None
            if last_step == 1:
                with np.load(
                    folder / "initial-gradient.npz", allow_pickle=False
                ) as initial_gradient:
                    q1_expected = initial_gradient["adam_delta"]
                assert step == 1
                q1_error = float(np.max(np.abs(q - q1_expected)))
                assert np.allclose(q, q1_expected, rtol=2e-12, atol=2e-13), q1_error
            latest = torch.load(
                folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
            )
            assert np.array_equal(q, latest["q"].numpy())
            assert np.array_equal(u, latest["u"])
            group = latest["optimizer"]["param_groups"]
            assert len(group) == 1
            assert group[0]["lr"] == 0.3
            assert group[0]["eps"] == 0.01
            assert group[0]["betas"] == (0.9, 0.999)
            computed = OLD._metrics(q, u, **values)  # noqa: SLF001
            computed.update(engine.evaluate_surface(u[values["skin_ids"]]))
            errors = OLD._check_reported(computed, trace[last_step])  # noqa: SLF001
            errors.update(
                {
                    f"summary_last/{key}": value
                    for key, value in OLD._check_reported(  # noqa: SLF001
                        computed, summary["last_metrics"]
                    ).items()
                }
            )
            objective = (
                float(computed["position_loss_component_mm2"])
                + 0.25
                * float(protocol["normalization"]["L20"])
                / float(protocol["normalization"]["N0"])
                * float(computed["normal_loss"])
                + coefficient * float(computed["activation_smoothness"])
            )
            errors["objective"] = OLD._close(objective, trace[last_step]["objective"])  # noqa: SLF001
            checks["branches"][branch] = {
                "last_step": last_step,
                "status": summary["status"],
                "failure": summary["failure"],
                "completed_budget": completed,
                "receipts": receipts,
                "endpoint_metric_absolute_errors": errors,
                "initial_adam_q1_max_abs_error": q1_error,
            }
            done.append(completed)
        checks["all_completed"] = all(done)
        checks["passed"] = True
    except Exception as error:  # noqa: BLE001
        checks["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    write(out / "checks.json", checks)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
