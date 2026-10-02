"""Independent CPU audit for fixed-reference-length Raw6 face fits."""

from __future__ import annotations

import copy
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
L_REF_MM = 13.236093032531715
NORMAL_WEIGHT = 1.0
BASE_SMOOTH_COEFFICIENT = 0.003214147722027223
BASE_ADAM_EPS = 0.01


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("130-reference-fit")
    output: Path = Path("135-verification")
    validation_dir: Path = Path("121-reference-validation")
    selected_config: Path = Path("110-shape-loss-config/loss-config.json")


def _old() -> Any:
    spec = importlib.util.spec_from_file_location(
        "old20_reference", Path(__file__).with_name("20-verify.py")
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
    assert isinstance(value, dict), path
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": digest(path)}


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


def initial(
    folder: Path, branch: str, smooth_coefficient: float, adam_eps: float
) -> None:
    with np.load(folder / "initial-state.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 0
        assert bool(state["activation_identity"])
        assert bool(state["adjoint_initial_guess_zero"])
        for name in ("q", "u", "m", "v"):
            assert not np.count_nonzero(state[name]), name
    update = read(folder / "initial-update.json")
    # ``run_branch`` is deliberately reused; its beta field is the direct
    # normal weight for ReferenceStudy, as recorded in the protocol.
    assert float(update["beta"]) == NORMAL_WEIGHT
    assert float(update["smooth_coefficient"]) == smooth_coefficient
    with np.load(folder / "initial-gradient.npz", allow_pickle=False) as state:
        gradient = np.asarray(state["gradient"])
        actual = np.asarray(state["adam_delta"])
    expected = -0.3 * gradient / (np.abs(gradient) + adam_eps)
    assert np.array_equal(actual, expected)
    saved = torch.load(
        folder / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    assert saved["branch"] == branch
    assert int(saved["step"]) > 0
    item = next(iter(saved["optimizer"]["state"].values()))
    assert int(item["step"]) == int(saved["step"])


def config_value(config: dict[str, Any], name: str) -> float:
    value = float(config[name])
    assert np.isfinite(value)
    return value


def check_loss_terms(
    metrics: dict[str, float | int], reported: dict[str, float], coefficient: float
) -> dict[str, float]:
    expected = {
        "position_contribution": float(metrics["position_loss_component_mm2"])
        / L_REF_MM**2,
        "normal_contribution": NORMAL_WEIGHT * float(metrics["normal_loss"]),
        "regularizer_contribution": coefficient
        * float(metrics["activation_smoothness"]),
    }
    expected["objective"] = sum(expected.values())
    return {name: OLD._close(value, reported[name]) for name, value in expected.items()}  # noqa: SLF001


def source_paths(sources: dict[str, dict[str, str]]) -> dict[str, str]:
    """Collapse module aliases only after proving their frozen hashes agree."""
    grouped: dict[str, set[str]] = {}
    for item in sources.values():
        path = str(Path(item["path"]).resolve())
        grouped.setdefault(path, set()).add(item["sha256"])
    assert all(len(hashes) == 1 for hashes in grouped.values())
    return {path: next(iter(hashes)) for path, hashes in grouped.items()}


def check_resume_prefix(  # noqa: PLR0915
    source: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Verify the accepted interrupted prefix and the first resumed Adam step."""
    resume = protocol["resume"]
    assert resume["from_step"] == 102
    assert resume["to_step"] == 200
    assert resume["resumed_branches"] == ["smooth-off-normal"]
    assert resume["fresh_branches"] == ["smooth-on-normal"]
    parent = Path(resume["parent"])
    assert record(parent / "protocol.json") == resume["parent_protocol_record"]
    gate_path = Path(resume["cpu_gate"]["path"])
    assert record(gate_path) == resume["cpu_gate"]
    gate = read(gate_path)
    assert gate["passed"] is True
    assert gate["parent_protocol_record"] == resume["parent_protocol_record"]
    assert gate["parent_artifacts"] == resume["parent_artifacts"]
    assert record(Path(resume["protocol_record"]["path"])) == resume["protocol_record"]
    assert (
        record(Path(resume["interruption_record"]["path"]))
        == resume["interruption_record"]
    )
    for item in resume["parent_artifacts"]:
        assert record(Path(item["path"])) == item
    parent_protocol = read(parent / "protocol.json")
    parent_sources = source_paths(parent_protocol["sources"])
    resumed_sources = source_paths(protocol["sources"])
    assert parent_sources.items() <= resumed_sources.items()
    parent_branch, branch = parent / "smooth-off-normal", source / "smooth-off-normal"
    parent_trace = OLD._read_trace(parent_branch / "trace.csv")  # noqa: SLF001
    trace = OLD._read_trace(branch / "trace.csv")  # noqa: SLF001
    assert list(parent_trace) == list(range(103))
    assert all(parent_trace[step] == trace[step] for step in parent_trace)
    parent_receipts = (parent_branch / "solver-receipts.jsonl").read_text().splitlines()
    receipts = (branch / "solver-receipts.jsonl").read_text().splitlines()
    assert receipts[:103] == parent_receipts
    parent_start = torch.load(
        parent_branch / "optimizer-latest.pt", map_location="cpu", weights_only=False
    )
    copied_start = torch.load(
        branch / "continuation-start.pt", map_location="cpu", weights_only=False
    )
    assert parent_start["step"] == copied_start["step"] == 102
    assert parent_start["branch"] == copied_start["branch"] == "smooth-off-normal"
    assert torch.equal(parent_start["q"], copied_start["q"])
    assert np.array_equal(parent_start["u"], copied_start["u"])
    assert (
        parent_start["optimizer"]["param_groups"]
        == copied_start["optimizer"]["param_groups"]
    )
    parent_state = next(iter(parent_start["optimizer"]["state"].values()))
    copied_state = next(iter(copied_start["optimizer"]["state"].values()))
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(parent_state[name], copied_state[name])
    replay = read(branch / "resume-replay.json")
    assert replay["passed"] is True
    assert replay["step"] == 102
    assert replay["optimizer_state"] == {
        "step": 102,
        "q_equal": True,
        "m_equal": True,
        "v_equal": True,
    }
    old = parent_trace[102]
    expected_limits = {
        "objective": 1e-4 * abs(old["objective"]),
        "position_contribution": 1e-4 * abs(old["position_contribution"]),
        "fit_rms_mm": 1e-3,
        "normal_angle_rms_deg": 0.01,
        "normal_loss": 1e-4 * abs(old["normal_loss"]),
        "activation_smoothness": 1e-12 * max(1.0, abs(old["activation_smoothness"])),
        "detF_min": 0.001,
        "inverted_all_cells": 0.0,
        "forward_displacement_max_abs_difference_m": 1e-6,
        "physical_gradient_rms_relative_difference": 0.02,
    }
    assert replay["error_limits"] == expected_limits
    assert all(
        replay["errors"][name] <= limit for name, limit in expected_limits.items()
    )
    start = torch.load(
        branch / "continuation-start.pt", map_location="cpu", weights_only=False
    )
    with np.load(branch / "resume-gradient.npz", allow_pickle=False) as state:
        gradient = torch.as_tensor(state["gradient"])
        assert int(state["step"]) == 102
        assert np.array_equal(state["q"], start["q"].numpy())
    q = torch.nn.Parameter(start["q"].detach().clone())
    optimizer = torch.optim.Adam(
        [q], lr=0.3, eps=BASE_ADAM_EPS / L_REF_MM**2, betas=(0.9, 0.999)
    )
    optimizer.load_state_dict(copy.deepcopy(start["optimizer"]))
    q.grad = gradient.clone()
    optimizer.step()
    with np.load(branch / "step-0103.npz", allow_pickle=False) as state:
        assert int(state["step"]) == 103
        q103_error = float(np.max(np.abs(state["q"] - q.detach().numpy())))
        assert q103_error <= 3e-16, q103_error
    assert int(optimizer.state[q]["step"]) == 103
    return {
        "from_step": 102,
        "prefix_trace_rows": 103,
        "prefix_receipts": 103,
        "replay_passed": True,
        "first_resumed_adam_step": 103,
        "first_resumed_adam_q_max_abs_error": q103_error,
    }


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Any] = {"passed": False, "all_completed": False, "branches": {}}
    try:
        source = cherries.input(cfg.comparison_dir)
        validation = cherries.input(cfg.validation_dir)
        chosen = cherries.input(cfg.selected_config)
        baseline = cherries.input(Path("90-beta1"))
        chosen_config = read(chosen)
        assert config_value(chosen_config, "l_ref_mm") == L_REF_MM
        assert config_value(chosen_config, "normal_weight") == NORMAL_WEIGHT
        assert chosen_config["objective"] == (
            "position_component_mse_mm2 / l_ref_mm**2 + normal_weight * normal_chord_squared"
        )
        protocol = read(source / "protocol.json")
        resumed = "resume" in protocol
        config = protocol["config"]
        assert int(config["steps"]) >= 1
        assert set(config["branches"].split(",")) == set(BRANCHES)
        # The frozen runner's effective Config is kept for its branch and Adam
        # bookkeeping.  Reference-specific quantities live in the protocol.
        assert config_value(config, "beta") == NORMAL_WEIGHT
        expected_smooth = BASE_SMOOTH_COEFFICIENT / L_REF_MM**2
        expected_eps = BASE_ADAM_EPS / L_REF_MM**2
        assert config_value(config, "smooth_coefficient") == expected_smooth
        assert config_value(config, "adam_eps") == expected_eps
        assert config_value(config, "learning_rate") == 0.3
        assert config_value(config, "smooth_length_m") == 0.005
        assert int(config["checkpoint_interval"]) > 0
        if resumed:
            assert protocol["start"].startswith("Both branches originate from neutral;")
        else:
            assert protocol["start"].startswith("q=0, B=I, u=0")
        assert protocol["inverse_stationarity_claimed"] is False
        assert protocol["mechanical_stability_claimed"] is False
        reference = protocol["reference_normalization"]
        assert config_value(reference, "l_ref_mm") == L_REF_MM
        assert config_value(reference, "normal_weight") == NORMAL_WEIGHT
        assert config_value(reference, "position_coefficient") == 1 / L_REF_MM**2
        assert config_value(reference, "legacy_objective_scale") == L_REF_MM**2
        assert config_value(reference, "equivalent_previous_beta") == (
            L_REF_MM**2
            * config_value(protocol["normalization"], "N0")
            / config_value(protocol["normalization"], "L20")
        )
        assert config_value(reference, "legacy_adam_eps") == BASE_ADAM_EPS
        assert (
            config_value(reference, "legacy_smooth_coefficient")
            == BASE_SMOOTH_COEFFICIENT
        )
        assert reference["beta_key_semantics"] == (
            "direct normal weight in reused runner interface; no initial-error multiplier"
        )
        assert (
            protocol["objective"]
            == "L2/l_ref_mm**2 + normal_weight*normal_chord_squared + smooth_coefficient*R"
        )
        checks["selected_loss"] = record(chosen)
        assert protocol["selected_loss_record"] == checks["selected_loss"]
        checks["source_protocol_record"] = record(source / "protocol.json")
        checks["source_dir"] = str(source.resolve())
        checks["provenance"] = OLD._check_sources(protocol)  # noqa: SLF001
        baseline_protocol = read(baseline / "protocol.json")
        for name in (
            "fixture",
            "materials",
            "forward_tolerance",
            "normalization",
            "surface_points",
            "surface_triangles",
            "volume_points",
            "tetrahedra",
            "active_cells",
        ):
            assert baseline_protocol[name] == protocol[name], name
        baseline_sources = source_paths(baseline_protocol["sources"])
        reference_sources = source_paths(protocol["sources"])
        assert len(baseline_protocol["sources"]) == 97
        assert len(protocol["sources"]) >= 99
        for value in baseline_protocol["sources"].values():
            path = str(Path(value["path"]).resolve())
            assert reference_sources[path] == value["sha256"], path
        checks["beta1_lineage"] = {
            "protocol": record(baseline / "protocol.json"),
            "shared_source_records_checked": len(baseline_protocol["sources"]),
            "reference_source_records_checked": len(protocol["sources"]),
            "shared_unique_paths_checked": len(baseline_sources),
            "reference_unique_paths_checked": len(reference_sources),
        }

        validation_protocol = read(validation / "protocol.json")
        validation_config = validation_protocol["config"]
        for name, expected in (
            ("beta", NORMAL_WEIGHT),
            ("smooth_coefficient", expected_smooth),
            ("adam_eps", expected_eps),
        ):
            assert config_value(validation_config, name) == expected
        assert validation_protocol["fixture"] == protocol["fixture"]
        validation_sources = source_paths(validation_protocol["sources"])
        protocol_sources = source_paths(protocol["sources"])
        assert validation_sources.items() <= protocol_sources.items()
        gradient = read(source / "gradient-validation.json")
        assert gradient["status"] == "passed"
        assert digest(source / "gradient-validation.json") == digest(
            validation / "gradient-validation.json"
        )
        assert read(source / "normal-validation.json")["passed"] is True
        checks["gates"] = {
            "gradient": record(validation / "gradient-validation.json"),
            "normal": record(source / "normal-validation.json"),
        }

        assert validation_protocol["reference_normalization"] == reference
        assert (
            validation_protocol["selected_loss_record"]
            == protocol["selected_loss_record"]
        )
        assert validation_protocol["protocol_record"] == protocol["protocol_record"]
        preflight = read(source / "reference-preflight.json")
        required = {
            "selected_loss",
            "protocol",
            "gradient_protocol",
            "gradient_validation",
            "normal_validation",
        }
        if int(config["steps"]) > 1:
            required.update({"smoke_protocol", "smoke_audit"})
        assert required <= preflight.keys(), sorted(required - preflight.keys())
        for label, value in preflight.items():
            path = Path(value["path"])
            assert path.exists(), label
            assert digest(path) == value["sha256"], label
        normal_preflight = preflight["normal_validation"]
        normal_path = Path(normal_preflight["path"])
        assert normal_path.name == "checks.json"
        assert normal_path.parent.name == "05-normal-verification"
        assert digest(source / "normal-validation.json") == normal_preflight["sha256"]
        checks["preflight"] = {"records_checked": len(preflight)}
        if int(config["steps"]) > 1:
            smoke_audit_path = Path(preflight["smoke_audit"]["path"])
            smoke_checks = read(smoke_audit_path)
            assert smoke_checks["passed"] is True
            assert smoke_checks["all_completed"] is True
            smoke_protocol_path = Path(preflight["smoke_protocol"]["path"])
            smoke_protocol = read(smoke_protocol_path)
            assert record(smoke_protocol_path) == preflight["smoke_protocol"]
            assert config_value(smoke_protocol["config"], "beta") == NORMAL_WEIGHT
            assert smoke_protocol["reference_normalization"] == reference
            assert (
                smoke_protocol["selected_loss_record"]
                == protocol["selected_loss_record"]
            )
            smoke_sources = source_paths(smoke_protocol["sources"])
            protocol_sources = source_paths(protocol["sources"])
            if resumed:
                assert smoke_sources.items() <= protocol_sources.items()
            else:
                assert smoke_sources == protocol_sources
            checks["preflight"]["smoke_passed"] = True

        if resumed:
            checks["resume"] = check_resume_prefix(source, protocol)

        values = context(source)
        engine = OLD.StudyMetrics()
        completed: list[bool] = []
        for branch in BRANCHES:
            folder = source / branch
            coefficient = expected_smooth if branch.startswith("smooth-on") else 0.0
            initial(folder, branch, coefficient, expected_eps)
            trace = OLD._read_trace(folder / "trace.csv")  # noqa: SLF001
            last_step = max(trace)
            assert list(trace) == list(range(last_step + 1))
            receipts = OLD._check_solver_receipts(  # noqa: SLF001
                folder / "solver-receipts.jsonl", last_step
            )
            summary = read(folder / "summary.json")
            assert int(summary["last_step"]) == last_step
            done = last_step == int(config["steps"])
            if done:
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
                ) as state:
                    q1_expected = state["adam_delta"]
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
            assert group[0]["eps"] == expected_eps
            assert group[0]["betas"] == (0.9, 0.999)
            computed = OLD._metrics(q, u, **values)  # noqa: SLF001
            computed.update(engine.evaluate_surface(u[values["skin_ids"]]))
            errors = OLD._check_reported(computed, trace[last_step])  # noqa: SLF001
            errors.update(check_loss_terms(computed, trace[last_step], coefficient))
            errors.update(
                {
                    f"summary_last/{name}": value
                    for name, value in OLD._check_reported(  # noqa: SLF001
                        computed, summary["last_metrics"]
                    ).items()
                }
            )
            initial_q, initial_u = OLD._load_initial(folder / "initial-state.npz")  # noqa: SLF001
            initial_computed = OLD._metrics(initial_q, initial_u, **values)  # noqa: SLF001
            initial_computed.update(
                engine.evaluate_surface(initial_u[values["skin_ids"]])
            )
            initial_errors = OLD._check_reported(  # noqa: SLF001
                initial_computed, trace[0]
            )
            initial_errors.update(
                check_loss_terms(initial_computed, trace[0], coefficient)
            )
            checks["branches"][branch] = {
                "last_step": last_step,
                "status": summary["status"],
                "failure": summary["failure"],
                "completed_budget": done,
                "receipts": receipts,
                "endpoint_metric_absolute_errors": errors,
                "initial_metric_absolute_errors": initial_errors,
                "initial_adam_q1_max_abs_error": q1_error,
            }
            completed.append(done)
        checks["all_completed"] = all(completed)
        checks["passed"] = True
    except Exception as error:  # noqa: BLE001
        checks["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    write(output / "checks.json", checks)
    assert checks["passed"], checks.get("error")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
