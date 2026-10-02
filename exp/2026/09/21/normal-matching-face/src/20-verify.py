"""Independent CPU checks of the saved Raw6 normal-matching face states."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment import Profile

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
LEGACY = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
sys.path.insert(0, str(LEGACY))
from study_metrics import StudyMetrics  # noqa: E402

BRANCHES = (
    "smooth-off-l2",
    "smooth-off-normal",
    "smooth-on-l2",
    "smooth-on-normal",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = Path("10-comparison")
    output: Path = Path("20-verification")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict), path
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _close(actual: float, reported: float, *, tolerance: float = 2e-8) -> float:
    error = abs(actual - reported)
    assert error <= tolerance * max(1.0, abs(actual), abs(reported)), (
        actual,
        reported,
        error,
    )
    return error


def _activation(q: np.ndarray) -> np.ndarray:
    assert q.ndim == 2
    assert q.shape[1] == 6
    B = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    B[:, (0, 1, 2), (0, 1, 2)] += q[:, :3]
    B[:, 0, 1] = B[:, 1, 0] = q[:, 3]
    B[:, 1, 2] = B[:, 2, 1] = q[:, 4]
    B[:, 0, 2] = B[:, 2, 0] = q[:, 5]
    return B


def _gradient_energy(
    residual: np.ndarray, points: np.ndarray, triangles: np.ndarray
) -> float:
    vertices = points[triangles]
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    norm2 = np.einsum("ij,ij->i", normal, normal)
    assert np.all(norm2 > 0)
    gradients = (
        np.stack(
            (
                np.cross(normal, vertices[:, 2] - vertices[:, 1]),
                np.cross(normal, vertices[:, 0] - vertices[:, 2]),
                np.cross(normal, vertices[:, 1] - vertices[:, 0]),
            ),
            axis=1,
        )
        / norm2[:, None, None]
    )
    field_gradient = np.einsum("tvi,tvj->tij", residual[triangles], gradients)
    area = 0.5 * np.sqrt(norm2)
    return float(np.sum(area * np.sum(field_gradient**2, axis=(1, 2))) / area.sum())


def _normal_metrics(
    skin_u: np.ndarray,
    rest_skin: np.ndarray,
    target_skin: np.ndarray,
    triangles: np.ndarray,
) -> dict[str, float]:
    def normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vertices = points[triangles]
        cross = np.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        )
        double_area = np.linalg.norm(cross, axis=1)
        assert np.all(np.isfinite(double_area))
        assert np.all(double_area > 1e-14), "collapsed skin triangle"
        return cross / double_area[:, None], double_area

    reference, reference_double_area = normals(rest_skin)
    del reference
    target, _ = normals(rest_skin + target_skin)
    current, current_double_area = normals(rest_skin + skin_u)
    weights = reference_double_area / reference_double_area.sum()
    chord2 = np.sum((current - target) ** 2, axis=1)
    angles = np.arccos(np.clip(np.sum(current * target, axis=1), -1.0, 1.0))
    return {
        "normal_loss": float(weights @ chord2),
        "normal_angle_rms_deg": float(np.degrees(np.sqrt(weights @ angles**2))),
        "normal_chord_rms": float(np.sqrt(weights @ chord2)),
        "minimum_triangle_area_ratio": float(
            np.min(current_double_area / reference_double_area)
        ),
    }


def _regularizer(
    q: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
    factor: float,
) -> float:
    difference = q[edge_i] - q[edge_j]
    frobenius2 = np.sum(difference[:, :3] ** 2, axis=1) + 2 * np.sum(
        difference[:, 3:] ** 2, axis=1
    )
    return float(factor * (edge_weight @ frobenius2))


def _detf(rest: np.ndarray, u: np.ndarray, tets: np.ndarray) -> np.ndarray:
    result = np.empty(len(tets))
    points = rest + u
    for start in range(0, len(tets), 100_000):
        stop = min(start + 100_000, len(tets))
        cells = tets[start:stop]
        dm = np.transpose(rest[cells[:, 1:]] - rest[cells[:, :1]], (0, 2, 1))
        ds = np.transpose(points[cells[:, 1:]] - points[cells[:, :1]], (0, 2, 1))
        denominator = np.linalg.det(dm)
        assert np.all(denominator > 0)
        result[start:stop] = np.linalg.det(ds) / denominator
    return result


def _read_trace(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="") as stream:
        values = list(csv.DictReader(stream))
    assert values
    result: dict[int, dict[str, float]] = {}
    for row in values:
        step = int(row.pop("step"))
        assert step not in result
        result[step] = {key: float(value) for key, value in row.items()}
    assert list(result) == list(range(max(result) + 1))
    return result


def _load_initial(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as state:
        assert {
            "q",
            "u",
            "m",
            "v",
            "step",
            "activation_identity",
            "adjoint_initial_guess_zero",
        } <= set(state.files), state.files
        assert int(state["step"]) == 0
        assert bool(state["activation_identity"])
        assert bool(state["adjoint_initial_guess_zero"])
        q, u = (
            np.asarray(state["q"], dtype=np.float64),
            np.asarray(state["u"], dtype=np.float64),
        )
        assert not np.count_nonzero(q)
        assert not np.count_nonzero(u)
        assert not np.count_nonzero(state["m"])
        assert not np.count_nonzero(state["v"])
        return q, u


def _load_state(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as state:
        assert {"q", "u", "step", "solver_valid", "physical_volume_energy"} <= set(
            state.files
        ), state.files
        assert bool(state["solver_valid"])
        assert bool(state["physical_volume_energy"])
        return (
            np.asarray(state["q"], dtype=np.float64),
            np.asarray(state["u"], dtype=np.float64),
            int(state["step"]),
        )


def _checkpoint_steps(folder: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in folder.glob("step-*.npz"):
        match = re.fullmatch(r"step-(\d+)\.npz", path.name)
        assert match is not None
        result[int(match.group(1))] = path
    assert result
    assert 0 in result
    return result


def _check_solver_receipts(path: Path, last_step: int) -> dict[str, float | int]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == last_step + 1
    assert [int(row["step"]) for row in rows] == list(range(last_step + 1))
    for row in rows:
        assert isinstance(row["forward"], dict)
        assert isinstance(row["adjoint"], dict)
        assert row["forward"].get("success") is True
        assert row["adjoint"].get("success") is True
    return {"receipt_count": len(rows)}


def _check_sources(protocol: dict[str, Any]) -> dict[str, Any]:
    source_records = protocol["sources"]
    assert isinstance(source_records, dict)
    assert source_records
    checked = 0
    current_checked = 0
    for details in source_records.values():
        if not isinstance(details, dict) or "snapshot" not in details:
            continue
        snapshot = Path(details["snapshot"])
        assert snapshot.exists(), snapshot
        assert _sha256(snapshot) == details["sha256"], snapshot
        current = Path(details["path"])
        assert current.exists(), current
        assert _sha256(current) == details["sha256"], current
        checked += 1
        current_checked += 1
    assert checked > 0
    fixture = protocol["fixture"]
    for details in fixture.values():
        path = Path(details["path"])
        assert path.exists()
        assert _sha256(path) == details["sha256"]
    return {
        "source_snapshots_checked": checked,
        "current_sources_checked": current_checked,
        "fixture_receipts_checked": len(fixture),
    }


def _metrics(
    q: np.ndarray,
    u: np.ndarray,
    *,
    rest: np.ndarray,
    skin_ids: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    triangles: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    fixed: np.ndarray,
    fixed_values: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_weight: np.ndarray,
    regularizer_factor: float,
) -> dict[str, float | int]:
    prediction = u[skin_ids]
    residual = prediction - target
    mean = np.sum(weights[:, None] * residual, axis=0)
    detf = _detf(rest, u, tets)
    eigen = np.linalg.eigvalsh(_activation(q))
    l2 = float(1e6 * np.sum(weights[:, None] * residual**2) / 3)
    gradient = _gradient_energy(residual, rest[skin_ids], triangles)
    normal = _normal_metrics(prediction, rest[skin_ids], target, triangles)
    return {
        "position_loss_component_mm2": l2,
        "surface_gradient_loss": gradient,
        "surface_gradient_rms": float(math.sqrt(gradient)),
        "fit_rms_mm": float(1000 * math.sqrt(np.sum(weights[:, None] * residual**2))),
        "motion_rms_mm": float(
            1000 * math.sqrt(np.sum(weights[:, None] * prediction**2))
        ),
        "centered_fit_rms_mm": float(
            1000 * math.sqrt(np.sum(weights[:, None] * (residual - mean) ** 2))
        ),
        "mean_error_norm_mm": float(1000 * np.linalg.norm(mean)),
        "target_projection": float(
            np.sum(weights[:, None] * prediction * target)
            / np.sum(weights[:, None] * target**2)
        ),
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
        "inverted_all_cells": int(np.count_nonzero(detf <= 0)),
        "inverted_active_cells": int(np.count_nonzero(detf[active_ids] <= 0)),
        "non_spd_active_cells": int(np.count_nonzero(eigen[:, 0] <= 0)),
        "activation_eigen_min": float(eigen.min()),
        "activation_eigen_max": float(eigen.max()),
        "fixed_displacement_error": float(
            np.max(np.abs(u[fixed] - fixed_values[fixed]))
        ),
        "activation_smoothness": _regularizer(
            q, edge_i, edge_j, edge_weight, regularizer_factor
        ),
        **normal,
    }


def _check_reported(
    computed: dict[str, float | int], reported: dict[str, float]
) -> dict[str, float]:
    errors = {}
    for name, value in computed.items():
        assert name in reported, name
        errors[name] = _close(float(value), float(reported[name]))
    return errors


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    source = cherries.input(cfg.comparison_dir)
    protocol = _json(source / "protocol.json")
    config = protocol["config"]
    normalization = protocol["normalization"]
    beta = float(config["beta"])
    smooth_coefficient = float(config["smooth_coefficient"])
    assert beta == 0.05
    assert smooth_coefficient == 0.003214147722027223
    expected_steps = int(config["steps"])
    assert expected_steps >= 0
    assert set(config["branches"].split(",")) == set(BRANCHES)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    checks: dict[str, Any] = {"provenance": _check_sources(protocol), "branches": {}}
    gate_files = {
        "gradient": source / "gradient-validation.json",
        "normal": source / "normal-validation.json",
    }
    assert _json(gate_files["gradient"])["status"] == "passed"
    assert _json(gate_files["normal"])["passed"] is True
    checks["gates"] = {name: str(path) for name, path in gate_files.items()}

    with np.load(source / "mesh.npz", allow_pickle=False) as mesh:
        required = {
            "rest_points",
            "skin_ids",
            "triangles",
            "target_displacement_skin",
            "skin_vertex_weights",
            "tets",
            "edge_i",
            "edge_j",
            "edge_weight",
            "active_ids",
            "active_volume_weights",
            "regularizer_factor",
            "fixed_mask",
            "fixed_values",
        }
        assert required <= set(mesh.files), sorted(required - set(mesh.files))
        rest = np.asarray(mesh["rest_points"], dtype=np.float64)
        skin_ids = np.asarray(mesh["skin_ids"], dtype=np.int64)
        triangles = np.asarray(mesh["triangles"], dtype=np.int64)
        target = np.asarray(mesh["target_displacement_skin"], dtype=np.float64)
        weights = np.asarray(mesh["skin_vertex_weights"], dtype=np.float64)
        tets = np.asarray(mesh["tets"], dtype=np.int64)
        edge_i, edge_j = (
            np.asarray(mesh["edge_i"], dtype=np.int64),
            np.asarray(mesh["edge_j"], dtype=np.int64),
        )
        edge_weight = np.asarray(mesh["edge_weight"], dtype=np.float64)
        active_ids = np.asarray(mesh["active_ids"], dtype=np.int64)
        active_weights = np.asarray(mesh["active_volume_weights"], dtype=np.float64)
        regularizer_factor = float(mesh["regularizer_factor"])
        fixed = np.asarray(mesh["fixed_mask"], dtype=bool)
        fixed_values = np.asarray(mesh["fixed_values"], dtype=np.float64)
    assert weights.shape == (len(skin_ids),)
    assert np.isclose(weights.sum(), 1.0)
    assert target.shape == (len(skin_ids), 3)
    assert active_ids.shape == active_weights.shape
    assert np.all((active_ids >= 0) & (active_ids < len(tets)))
    assert len(active_weights) == max(edge_i.max(), edge_j.max()) + 1
    assert np.all(edge_weight > 0)
    assert regularizer_factor > 0
    assert np.isclose(sum(active_weights), 1.0)

    rest_skin = rest[skin_ids]
    triangle_vertices = rest_skin[triangles]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    assert np.all(triangle_area > 0)
    reconstructed_weights = np.zeros(len(skin_ids))
    np.add.at(
        reconstructed_weights, triangles.reshape(-1), np.repeat(triangle_area / 3, 3)
    )
    reconstructed_weights /= reconstructed_weights.sum()
    assert np.allclose(weights, reconstructed_weights, rtol=1e-14, atol=1e-16)
    neutral_l2 = float(1e6 * np.sum(weights[:, None] * target**2) / 3)
    neutral_normal = _normal_metrics(
        np.zeros_like(target), rest_skin, target, triangles
    )["normal_loss"]
    _close(neutral_l2, float(normalization["L20"]))
    _close(neutral_normal, float(normalization["N0"]))
    checks["normalization"] = {
        "L20": neutral_l2,
        "N0": neutral_normal,
        "skin_weights": "reference-triangle area lumped to vertices then normalized",
    }

    traces = {name: _read_trace(source / name / "trace.csv") for name in BRANCHES}
    for trace in traces.values():
        assert max(trace) == expected_steps
        assert len(trace) == expected_steps + 1
        _close(trace[0]["position_loss_component_mm2"], neutral_l2)
        _close(trace[0]["normal_loss"], neutral_normal)
    checkpoints = {name: _checkpoint_steps(source / name) for name in BRANCHES}
    first_q: dict[str, np.ndarray] = {}
    first_u: dict[str, np.ndarray] = {}
    metrics_engine = StudyMetrics()

    def evaluate_state(
        path: Path, reported: dict[str, float]
    ) -> tuple[dict[str, float | int], dict[str, float]]:
        q, u, _step = _load_state(path)
        computed = _metrics(
            q,
            u,
            rest=rest,
            skin_ids=skin_ids,
            target=target,
            weights=weights,
            triangles=triangles,
            tets=tets,
            active_ids=active_ids,
            fixed=fixed,
            fixed_values=fixed_values,
            edge_i=edge_i,
            edge_j=edge_j,
            edge_weight=edge_weight,
            regularizer_factor=regularizer_factor,
        )
        surface = metrics_engine.evaluate_surface(u[skin_ids])
        computed.update(surface)
        errors = _check_reported(computed, reported)
        return computed, errors

    for name in BRANCHES:
        folder = source / name
        q0, u0 = _load_initial(folder / "initial-state.npz")
        assert q0.shape == (len(active_weights), 6)
        first_q[name], first_u[name] = q0, u0
        _q, _u, step = _load_state(folder / "last.npz")
        assert step == max(traces[name])
        assert traces[name][0]["inverted_all_cells"] == 0
        assert not np.count_nonzero(q0)
        assert not np.count_nonzero(u0)
        summary = _json(folder / "summary.json")
        assert int(summary["last_step"]) == step
        assert step == expected_steps
        assert summary["status"] == "completed_budget_not_convergence_certified"
        assert summary["failure"] is None
        computed, errors = evaluate_state(folder / "last.npz", traces[name][step])
        errors.update(
            {
                f"summary_last/{metric}": error
                for metric, error in _check_reported(
                    computed, summary["last_metrics"]
                ).items()
            }
        )
        for metric, value in summary["initial_metrics"].items():
            if metric in traces[name][0]:
                _close(float(value), traces[name][0][metric])
        expected_normal = name.endswith("normal")
        expected_smooth = name.startswith("smooth-on")
        objective = float(computed["position_loss_component_mm2"])
        if expected_normal:
            objective += (
                beta
                * float(normalization["L20"])
                / float(normalization["N0"])
                * float(computed["normal_loss"])
            )
        if expected_smooth:
            objective += smooth_coefficient * float(computed["activation_smoothness"])
        errors["objective"] = _close(objective, traces[name][step]["objective"])
        checks["branches"][name] = {
            "last_step": step,
            "status": summary["status"],
            "failure": summary["failure"],
            "endpoint_metric_absolute_errors": errors,
            "endpoint_inverted_all": computed["inverted_all_cells"],
            "endpoint_detF_min": computed["detF_min"],
            "endpoint_normal_angle_rms_deg": computed["normal_angle_rms_deg"],
            "solver_receipts": _check_solver_receipts(
                folder / "solver-receipts.jsonl", step
            ),
        }

    reference_q = first_q[BRANCHES[0]]
    reference_u = first_u[BRANCHES[0]]
    for name in BRANCHES[1:]:
        assert np.array_equal(first_q[name], reference_q)
        assert np.array_equal(first_u[name], reference_u)
    checks["neutral_initialization"] = {
        "all_four_q_equal": True,
        "all_four_u_equal": True,
        "q_nonzero_count": 0,
        "u_nonzero_count": 0,
    }
    checks["completion_contract"] = {
        "expected_updates": expected_steps,
        "expected_forward_and_adjoint_receipts": expected_steps + 1,
        "all_branches_completed_budget": True,
        "all_branches_failure_null": True,
    }

    common_steps = set.intersection(*(set(items) for items in checkpoints.values()))
    shared_step = max(common_steps)
    common_noninverted = [
        step
        for step in sorted(common_steps, reverse=True)
        if all(int(traces[name][step]["inverted_all_cells"]) == 0 for name in BRANCHES)
    ]
    assert common_noninverted
    shared_noninverted_step = common_noninverted[0]
    comparisons: dict[str, Any] = {}
    for label, step in (
        ("latest_common_checkpoint", shared_step),
        ("latest_common_inversion_free_checkpoint", shared_noninverted_step),
    ):
        branch_results = {}
        for name in BRANCHES:
            computed, errors = evaluate_state(
                checkpoints[name][step], traces[name][step]
            )
            branch_results[name] = {
                "metric_absolute_errors": errors,
                "inverted_all": computed["inverted_all_cells"],
                "detF_min": computed["detF_min"],
                "fit_rms_mm": computed["fit_rms_mm"],
                "normal_angle_rms_deg": computed["normal_angle_rms_deg"],
                "activation_smoothness": computed["activation_smoothness"],
                "target_relative_highpass_residual_mm": computed[
                    "primary_union_normal_residual_highpass_5mm_rms_mm"
                ],
                "highpass_displacement_mm": computed[
                    "primary_union_normal_displacement_highpass_5mm_rms_mm"
                ],
            }
        if label.endswith("inversion_free_checkpoint"):
            assert all(row["inverted_all"] == 0 for row in branch_results.values())
        comparisons[label] = {"step": step, "branches": branch_results}
    checks["comparisons"] = comparisons
    checks["passed"] = True
    _write(output / "checks.json", checks)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
