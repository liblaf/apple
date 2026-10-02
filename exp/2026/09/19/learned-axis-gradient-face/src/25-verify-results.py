# ruff: noqa: PLR0915, PT018
"""CPU-only audit of completed learned-axis smoothness branches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]


class ProfileVerification(profiles.Profile):
    """Keep local verification logs after disabled Comet startup."""

    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = GROUP / "data/20-comparison"
    preparation_dir: Path = GROUP / "data/06-preparation"
    activation_gate: Path = GROUP / "data/05-activation/checks.json"
    output: Path = GROUP / "data/25-verification/checks.json"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert rows
    assert [row["step"] for row in rows] == list(range(len(rows)))
    return rows


def _state(path: Path) -> dict[str, np.ndarray | int | float]:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "s",
            "n",
            "q",
            "u",
            "step",
            "solver_valid",
            "physical_volume_energy",
            "smoothness_coefficient",
        }
        absent = required - set(saved.files)
        assert not absent, (path, absent)
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        return {
            "s": np.asarray(saved["s"], dtype=np.float64),
            "n": np.asarray(saved["n"], dtype=np.float64),
            "q": np.asarray(saved["q"], dtype=np.float64),
            "u": np.asarray(saved["u"], dtype=np.float64),
            "step": int(saved["step"]),
            "coefficient": float(saved["smoothness_coefficient"]),
        }


def _pack(strength: np.ndarray, axis: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    assert np.all(norm > 0)
    n = axis / norm
    delta = strength[:, None, None] * n[:, :, None] * n[:, None, :]
    return np.column_stack(
        (
            delta[:, 0, 0],
            delta[:, 1, 1],
            delta[:, 2, 2],
            delta[:, 0, 1],
            delta[:, 1, 2],
            delta[:, 0, 2],
        )
    )


def _frobenius_raw6(values: np.ndarray) -> np.ndarray:
    return np.sum(values[:, :3] ** 2, axis=1) + 2 * np.sum(values[:, 3:] ** 2, axis=1)


def _regularizer(
    q: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    weight: np.ndarray,
    active_volumes: np.ndarray,
    length: float,
) -> float:
    difference = q[edge_i] - q[edge_j]
    return float(
        length**2 / active_volumes.sum() * np.sum(weight * _frobenius_raw6(difference))
    )


def _assert_close(actual: float, reported: float, *, tolerance: float = 1e-9) -> float:
    error = abs(actual - reported)
    assert error <= tolerance * max(1.0, abs(actual), abs(reported)), (
        actual,
        reported,
        error,
    )
    return error


def _validate_sources(protocol: dict) -> int:
    records = protocol["sources"]
    assert records
    for record in records.values():
        snapshot = Path(record["snapshot"])
        assert snapshot.is_file()
        assert _digest(snapshot) == record["sha256"]
    return len(records)


def _receipts(path: Path, expected_steps: int) -> dict[str, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == expected_steps + 1
    assert [row["step"] for row in rows] == list(range(expected_steps + 1))
    assert all(row["forward"]["success"] for row in rows)
    assert all(row["adjoint"]["success"] for row in rows)
    return {
        "count": len(rows),
        "forward_successes": len(rows),
        "adjoint_successes": len(rows),
    }


def main(cfg: Config) -> None:
    comparison = cherries.input(cfg.comparison_dir.resolve())
    preparation_dir = cherries.input(cfg.preparation_dir.resolve())
    activation_gate_path = cherries.input(cfg.activation_gate.resolve())
    output = cherries.output(cfg.output.resolve(), mkdir=True)
    protocol = _json(comparison / "protocol.json")
    preparation = _json(preparation_dir / "protocol.json")
    calibration = _json(comparison / "calibration.json")
    coefficients = _json(comparison / "coefficients.json")
    activation_gate = _json(activation_gate_path)
    derivative_gate = _json(preparation_dir / "gradient-validation.json")
    assert activation_gate["passed"] is True
    assert derivative_gate["status"] == "passed"
    assert protocol["fixture"] == preparation["fixture"]
    assert protocol["data_loss"] == "gradient"
    assert protocol["data_scale"] == calibration["data_scale"]
    for name in ("learning_rate", "adam_eps", "smooth_length_m"):
        assert protocol["config"][name] == calibration[name]
    assert (
        protocol["sources"]["activation_model"]["sha256"]
        == preparation["sources"]["activation_model"]["sha256"]
    )
    expected_steps = int(protocol["config"]["steps"])
    source_count = _validate_sources(protocol)

    with np.load(comparison / "mesh.npz", allow_pickle=False) as mesh:
        saved_rest = np.asarray(mesh["rest_points"], dtype=np.float64)
        edge_i = np.asarray(mesh["edge_i"], dtype=np.int64)
        edge_j = np.asarray(mesh["edge_j"], dtype=np.int64)
        edge_weight = np.asarray(mesh["edge_weight"], dtype=np.float64)
        active_volumes = np.asarray(mesh["active_volumes"], dtype=np.float64)
        active_ids = np.asarray(mesh["active_ids"], dtype=np.int64)
        initial_u = np.asarray(mesh["initial_u"], dtype=np.float64)
    assert len(edge_i) == len(edge_j) == len(edge_weight)
    assert np.all(edge_weight > 0) and np.all(active_volumes > 0)
    volume = pv.read(protocol["fixture"]["volume.vtu"]["path"])
    rest = np.asarray(volume.points, dtype=np.float64)
    assert np.array_equal(rest, saved_rest)
    cells = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    rest_volume = (
        np.linalg.det(np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1)))
        / 6
    )
    assert np.all(rest_volume > 0)

    branches: dict[str, dict] = {}
    initial: dict[str, dict[str, np.ndarray | int | float]] = {}
    final: dict[str, dict[str, np.ndarray | int | float]] = {}
    for name in ("off", "on"):
        folder = comparison / name
        summary = _json(folder / "summary.json")
        trace = _trace(folder / "trace.csv")
        start = _state(folder / "step-0000.npz")
        endpoint = _state(folder / "last.npz")
        final_step = int(endpoint["step"])
        assert final_step <= expected_steps
        assert final_step == int(summary["last_step"])
        assert int(trace[-1]["step"]) == final_step
        assert summary["status"] in {
            "completed_budget_not_convergence_certified",
            "stationarity_and_plateau_thresholds_met",
        }
        assert int(start["step"]) == 0
        assert np.max(np.abs(np.asarray(start["q"]))) == 0
        assert np.max(np.abs(np.asarray(start["u"]) - initial_u)) < 1e-14
        s, n, q = (np.asarray(endpoint[key]) for key in ("s", "n", "q"))
        reconstructed = _pack(s, n)
        q_error = float(np.max(np.abs(q - reconstructed)))
        assert q_error < 1e-12
        norm_error = float(np.max(np.abs(np.linalg.norm(n, axis=1) - 1)))
        assert norm_error < 1e-12
        assert float(s.min()) >= 0
        C = np.zeros((len(q), 3, 3))
        C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = q[:, :3].T
        C[:, 0, 1] = C[:, 1, 0] = q[:, 3]
        C[:, 1, 2] = C[:, 2, 1] = q[:, 4]
        C[:, 0, 2] = C[:, 2, 0] = q[:, 5]
        eigenvalues = np.linalg.eigvalsh(C)
        assert float(eigenvalues.min()) >= -1e-12
        rank_error = float(np.max(np.abs(eigenvalues[:, :2])))
        assert rank_error < 1e-10
        u = np.asarray(endpoint["u"])
        deformed = rest + u
        detf = (
            np.linalg.det(
                np.transpose(deformed[tets[:, 1:]] - deformed[tets[:, :1]], (0, 2, 1))
            )
            / 6
            / rest_volume
        )
        R = _regularizer(
            q,
            edge_i,
            edge_j,
            edge_weight,
            active_volumes,
            protocol["config"]["smooth_length_m"],
        )
        reported = summary["last_metrics"]
        coefficient = float(coefficients[name])
        assert _assert_close(float(endpoint["coefficient"]), coefficient) == 0
        objective = (
            protocol["data_scale"] * float(reported["surface_gradient_loss"])
            + coefficient * R
        )
        errors = {
            "regularizer": _assert_close(R, float(reported["activation_smoothness"])),
            "regularizer_contribution": _assert_close(
                coefficient * R, float(reported["regularizer_contribution"])
            ),
            "data_objective": _assert_close(
                protocol["data_scale"] * float(reported["surface_gradient_loss"]),
                float(reported["data_objective"]),
            ),
            "objective": _assert_close(objective, float(reported["objective"])),
            "trace_objective": _assert_close(objective, trace[-1]["objective"]),
            "detF_min": _assert_close(float(detf.min()), float(reported["detF_min"])),
            "detF_max": _assert_close(float(detf.max()), float(reported["detF_max"])),
        }
        inverted_all = int(np.count_nonzero(detf <= 0))
        inverted_active = int(np.count_nonzero(detf[active_ids] <= 0))
        assert inverted_all == int(reported["inverted_all_cells"])
        assert inverted_active == int(reported["inverted_active_cells"])
        branches[name] = {
            "status": summary["status"],
            "final_step": final_step,
            "q_reconstruction_inf": q_error,
            "axis_unit_inf": norm_error,
            "minimum_strength": float(s.min()),
            "minimum_delta_activation_eigenvalue": float(eigenvalues.min()),
            "minimum_activation_eigenvalue": float(1 + eigenvalues.min()),
            "rank_one_error": rank_error,
            "regularizer": R,
            "objective_errors": errors,
            "physical_detF": {
                "minimum": float(detf.min()),
                "maximum": float(detf.max()),
                "inverted_all_cells": inverted_all,
                "inverted_active_cells": inverted_active,
            },
            "solver_receipts": _receipts(folder / "solver-receipts.jsonl", final_step),
        }
        initial[name], final[name] = start, endpoint

    q0_difference = float(
        np.max(np.abs(np.asarray(initial["off"]["q"]) - np.asarray(initial["on"]["q"])))
    )
    u0_difference = float(
        np.max(np.abs(np.asarray(initial["off"]["u"]) - np.asarray(initial["on"]["u"])))
    )
    final_q_difference = float(
        np.max(np.abs(np.asarray(final["off"]["q"]) - np.asarray(final["on"]["q"])))
    )
    assert q0_difference == 0 and u0_difference < 1e-14
    assert final_q_difference > 1e-10
    off_trace, on_trace = (
        _trace(comparison / "off/trace.csv"),
        _trace(comparison / "on/trace.csv"),
    )
    matched_steps = min(len(off_trace), len(on_trace))
    lr_difference = max(
        abs(left["learning_rate"] - right["learning_rate"])
        for left, right in zip(
            off_trace[:matched_steps], on_trace[:matched_steps], strict=True
        )
    )
    assert lr_difference == 0
    checks = {
        "source_count": source_count,
        "activation_gate": str(activation_gate_path),
        "derivative_gate": str(preparation_dir / "gradient-validation.json"),
        "data_loss": protocol["data_loss"],
        "data_scale": protocol["data_scale"],
        "smooth_length_m": protocol["config"]["smooth_length_m"],
        "coefficient_off": coefficients["off"],
        "coefficient_on": coefficients["on"],
        "neutral_q_difference": q0_difference,
        "neutral_u_difference": u0_difference,
        "final_q_difference": final_q_difference,
        "matched_learning_rate_max_difference": lr_difference,
        "matched_learning_rate_steps": matched_steps,
        "branches": branches,
        "passed": True,
    }
    output.write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileVerification)
