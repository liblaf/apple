"""Explicit checkpoint contracts and diagnostics for tensor continuations."""

# ruff: noqa: PLR0915

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensor_controls import matrices, project

SPEC = importlib.util.spec_from_file_location(
    "tensor_face_base", Path(__file__).with_name("20-face-inverse.py")
)
assert SPEC is not None
assert SPEC.loader is not None
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)

PHYSICS_FILES = (
    "experiment/face_physics.py",
    "experiment/tensor_active.py",
    "experiment/tensor_controls.py",
    "experiment/20-face-inverse.py",
)
PHYSICAL_CONFIG_KEYS = (
    "fixture",
    "model",
    "stress_reference_mpa",
    "stress_cap_mpa",
    "smooth_length_m",
)


def rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def tensor_rms_mpa(delta: torch.Tensor, qref: float) -> float:
    """Uniform-cell RMS of the physical tensor Frobenius norm."""
    return float(qref * delta.square().sum(-1).mean().sqrt())


def initial_replay_metrics(current: dict, parent: dict) -> dict:
    """Express restored-state agreement consistently in RMS and squared loss."""
    tolerance_mm = 1e-6
    keys = ("fit_rms_mm", "area_fit_rms_mm", "area_motion_rms_mm")
    differences = {key: abs(current[key] - parent[key]) for key in keys}
    # The fit objective averages three coordinates: D = F_unweighted**2 / 3.
    objective_tolerance = (
        2 * parent["fit_rms_mm"] * tolerance_mm + tolerance_mm**2
    ) / 3
    objective_difference = abs(
        current["data_objective_mm2"] - parent["data_objective_mm2"]
    )
    identity_errors = {
        label: abs(row["data_objective_mm2"] - row["fit_rms_mm"] ** 2 / 3)
        for label, row in (("parent", parent), ("recomputed", current))
    }
    return {
        "passed": all(value <= tolerance_mm for value in differences.values())
        and objective_difference <= objective_tolerance
        and all(value <= 1e-12 for value in identity_errors.values()),
        "rms_tolerance_mm": tolerance_mm,
        "absolute_rms_differences_mm": differences,
        "absolute_data_objective_difference_mm2": objective_difference,
        "derived_data_objective_tolerance_mm2": objective_tolerance,
        "data_objective_identity": "unweighted fit RMS squared / 3",
        "data_objective_identity_errors_mm2": identity_errors,
        "forward_and_adjoint_tolerances_changed": False,
    }


def _typed_trace_row(row: dict[str, str], schema: dict[str, Any]) -> dict[str, Any]:
    assert row.keys() == schema.keys()
    typed: dict[str, Any] = {}
    for key, value in row.items():
        expected = schema[key]
        if expected is None:
            assert value == ""
            typed[key] = None
        elif isinstance(expected, bool):
            assert value in {"True", "False"}
            typed[key] = value == "True"
        elif isinstance(expected, int):
            typed[key] = int(value)
        elif isinstance(expected, float):
            typed[key] = float(value)
        else:
            assert isinstance(expected, str)
            typed[key] = value
    return typed


def load_checkpoint(
    path: Path, cfg: Any, *, allow_intermediate: bool = False
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert {"step", "q", "u", "optimizer", "config"} <= state.keys()
    assert state["q"].shape == (288235, 6)
    assert state["q"].dtype == torch.float64
    assert not state["q"].requires_grad
    assert state["q"].grad is None
    assert np.asarray(state["u"]).shape == (228660, 3)
    assert torch.isfinite(state["q"]).all()
    assert np.isfinite(state["u"]).all()
    for key in PHYSICAL_CONFIG_KEYS:
        actual, expected = getattr(cfg, key), state["config"][key]
        if key == "fixture":
            assert Path(actual).resolve() == Path(expected).resolve()
        else:
            assert actual == expected, (key, actual, expected)
    groups = state["optimizer"]["param_groups"]
    assert len(groups) == 1
    assert len(groups[0]["params"]) == 1
    moment = state["optimizer"]["state"][groups[0]["params"][0]]
    assert int(moment["step"]) == int(state["step"])
    for key in ("exp_avg", "exp_avg_sq"):
        assert moment[key].shape == state["q"].shape
        assert torch.isfinite(moment[key]).all()
    assert (moment["exp_avg_sq"] >= 0).all()
    step = int(state["step"])
    step_name = f"step-{step:04d}.npz"
    summary_path = path.parent / "summary.json"
    provenance_path = path.parent / "provenance.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] in {
        "completed_fixed_budget",
        "completed_fixed_budget_continuation",
    }
    assert provenance_path.is_file()
    snapshots: dict[str, dict[str, np.ndarray]] = {}
    names = (step_name,) if allow_intermediate else ("final.npz", step_name)
    for name in names:
        with np.load(path.parent / name) as saved:
            assert int(saved["step"]) == step
            assert bool(saved["solver_valid"])
            assert np.array_equal(saved["q"], state["q"].numpy())
            assert np.array_equal(saved["u"], state["u"])
            assert saved["active_ids"].shape == (288235,)
            expected_q = (
                float(state["config"]["stress_reference_mpa"]) * matrices(state["q"])
            ).numpy()
            assert np.allclose(saved["Q"], expected_q, rtol=0.0, atol=1e-15)
            snapshots[name] = {
                key: saved[key].copy() for key in ("q", "u", "active_ids", "Q")
            }
    if allow_intermediate:
        assert summary["status"] == "completed_fixed_budget_continuation"
        assert path.name == f"optimizer-step-{step:04d}.pt"
        assert summary["config"] == state["config"]
        assert state["config"]["phase"] == "regularization"
        assert state["config"]["smoothness_weight"] > 0
        assert state["config"]["rank_weight"] == 0
        assert state["config"]["magnitude_weight"] == 0
        assert int(summary["initial_endpoint"]["step"]) < step
        assert step < int(summary["primary_endpoint"]["step"])
        trace_path = path.parent / "trace.csv"
        with trace_path.open(newline="") as stream:
            raw_rows = list(csv.DictReader(stream))
        rows = [_typed_trace_row(row, summary["primary_endpoint"]) for row in raw_rows]
        assert [row["step"] for row in rows] == list(
            range(
                int(summary["initial_endpoint"]["step"]),
                int(summary["primary_endpoint"]["step"]) + 1,
            )
        )
        matches = [row for row in rows if row["step"] == step]
        assert len(matches) == 1
        endpoint = matches[0]
        assert endpoint["local_step"] > 0
        assert endpoint["solver_valid"] is True
        solver_path = path.parent / "solver-receipts.jsonl"
        receipts = [json.loads(line) for line in solver_path.read_text().splitlines()]
        solver_matches = [receipt for receipt in receipts if receipt["step"] == step]
        assert len(solver_matches) == 1
        solver = solver_matches[0]
        assert solver["local_step"] == endpoint["local_step"]
        assert solver["forward"]["success"] is True
        assert solver["adjoint"]["success"] is True
        assert solver["forward"]["steps"] == endpoint["forward_steps"]
        assert solver["forward"]["grad_norm"] == endpoint["forward_grad_norm"]
        vtu_path = path.parent / f"step-{step:04d}.vtu"
        assert vtu_path.is_file()
        import pyvista as pv

        mesh = pv.read(vtu_path)
        active_ids = snapshots[step_name]["active_ids"]
        assert np.array_equal(
            np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)),
            active_ids,
        )
        assert np.array_equal(mesh.point_data["Displacement"], state["u"])
        mesh_q = np.asarray(mesh.cell_data["ActiveStressMatrixMPa"]).reshape(-1, 3, 3)
        assert np.array_equal(mesh_q[active_ids], snapshots[step_name]["Q"])
        state["checkpoint_evidence"] = {
            "kind": "intermediate",
            "optimizer": {"path": str(path.resolve()), "sha256": BASE.sha256(path)},
            "snapshot": {
                "path": str((path.parent / step_name).resolve()),
                "sha256": BASE.sha256(path.parent / step_name),
            },
            "mesh": {"path": str(vtu_path.resolve()), "sha256": BASE.sha256(vtu_path)},
            "trace": {
                "path": str(trace_path.resolve()),
                "sha256": BASE.sha256(trace_path),
            },
            "solver_receipts": {
                "path": str(solver_path.resolve()),
                "sha256": BASE.sha256(solver_path),
            },
            "summary": {
                "path": str(summary_path.resolve()),
                "sha256": BASE.sha256(summary_path),
            },
            "provenance": {
                "path": str(provenance_path.resolve()),
                "sha256": BASE.sha256(provenance_path),
            },
            "solver_receipt": solver,
        }
    else:
        for key in snapshots["final.npz"]:
            assert np.array_equal(
                snapshots["final.npz"][key], snapshots[step_name][key]
            )
        endpoint = summary["primary_endpoint"]
        assert int(endpoint["step"]) == step
        assert endpoint["solver_valid"] is True
        state["checkpoint_evidence"] = {"kind": "final"}
    state["active_ids"] = snapshots[step_name]["active_ids"]
    state["parent_endpoint"] = endpoint
    return state


def verify_parent_sources(parent: dict[str, Any], current: dict[str, Any]) -> None:
    for name in PHYSICS_FILES:
        assert parent["sources"][name] == current["sources"][name], name
    for name, digest in parent["sources"].items():
        if name.startswith("liblaf/apple/"):
            assert current["sources"][name] == digest, name
    assert parent["inputs"] == current["inputs"]


@torch.no_grad()
def projected_gradient_metrics(
    q: torch.Tensor, gradient: torch.Tensor, maximum: float, eta: float
) -> dict[str, float]:
    """Euclidean gradient mapping, independent of Adam's moment scaling."""
    assert eta > 0
    trial = q.detach().clone() - eta * gradient
    project(trial, maximum)
    mapping = (q.detach() - trial) / eta
    raw_rms = rms(gradient)
    result = {
        "projected_gradient_mapping_rms": rms(mapping),
        "projected_gradient_mapping_max_abs": float(mapping.abs().max()),
        "projected_gradient_eta": eta,
    }
    result["projected_to_raw_gradient_rms_ratio"] = (
        result["projected_gradient_mapping_rms"] / raw_rms if raw_rms else 0.0
    )
    return result


@torch.no_grad()
def adam_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    group = optimizer.param_groups[0]
    state = optimizer.state[group["params"][0]]
    step = int(state["step"])
    denominator_part = (state["exp_avg_sq"] / (1 - group["betas"][1] ** step)).sqrt()
    flattened = denominator_part.flatten()
    values = torch.quantile(flattened, flattened.new_tensor((0.5, 0.9, 0.99, 1.0)))
    eps = group["eps"]
    return {
        "adam_step": step,
        "adam_learning_rate": float(group["lr"]),
        "adam_eps": float(eps),
        "adam_sqrt_vhat_median": float(values[0]),
        "adam_sqrt_vhat_p90": float(values[1]),
        "adam_sqrt_vhat_p99": float(values[2]),
        "adam_sqrt_vhat_max": float(values[3]),
        "adam_eps_dominant_coordinate_fraction": float(
            (flattened < eps).double().mean()
        ),
    }
