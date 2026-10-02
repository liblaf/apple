"""Independently verify the completed released-axis continuation on CPU."""

# ruff: noqa: C901, EM102, PLR0912, PLR0915, PT018, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import comet_ml  # noqa: F401
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
RELEASED = GROUP / "data/90-released-axes"
FIXED = GROUP / "data/42-fixed-directions-400"
FIXED_VERIFICATION = GROUP / "data/62-verification-400/receipt.json"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
COMPLETED_STATUS = "completed_budget_not_stationarity_certified"
PARAMETERIZATION = "B=I+vvT; learned-axis"
CHECK_STEPS = (0, 128, 200)
NEAR_ZERO_ANGLE_TOLERANCE_DEG = 2e-6


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    released_dir: Path = RELEASED
    fixed_dir: Path = FIXED
    expected_steps: int = 200
    output_dir: Path = cherries.output("94-released-axes-verification", mkdir=True)


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def artifact_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else GROUP / path


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def close(actual: float, expected: float, label: str) -> float:
    error = abs(actual - expected)
    tolerance = 5e-11 * max(1.0, abs(actual), abs(expected))
    if error > tolerance:
        raise AssertionError(
            f"{label}: {actual!r} != {expected!r}; error {error} > {tolerance}"
        )
    return error


def max_error(actual: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert error <= tolerance, (error, tolerance)
    return error


def scaled_error(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    absolute = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 0.0)
    normalized = absolute / scale
    tolerance = atol + rtol * scale
    assert absolute <= tolerance, (absolute, normalized, scale, tolerance)
    return {
        "max_abs_error": absolute,
        "reference_scale": scale,
        "max_normalized_error": normalized,
        "absolute_plus_relative_tolerance": tolerance,
    }


def tensors(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c = np.einsum("ni,nj->nij", v, v)
    b = np.eye(3) + c
    z = np.einsum("nij,nkj->nik", b, b) - np.eye(3)
    s = np.sum(v * v, axis=1)
    return s, c, b, z


def load_state(path: Path, active_ids: np.ndarray, rest: np.ndarray) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "v",
            "u",
            "step",
            "source_fixed_step",
            "active_ids",
            "rest_points",
            "C",
            "B",
            "Z",
            "solver_valid",
            "physical_volume_energy",
            "physical_noninverted",
            "parameterization",
        }
        assert required.issubset(saved.files), (path, required - set(saved.files))
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert int(saved["source_fixed_step"]) == 400
        assert str(saved["parameterization"]) == (
            "B=I+vvT; learned reference axis and strength"
        )
        assert np.array_equal(saved["active_ids"], active_ids)
        assert np.array_equal(saved["rest_points"], rest)
        result = {
            "v": np.asarray(saved["v"], dtype=np.float64).copy(),
            "u": np.asarray(saved["u"], dtype=np.float64).copy(),
            "step": int(saved["step"]),
            "physical_noninverted": bool(saved["physical_noninverted"]),
            "stored_C": np.asarray(saved["C"], dtype=np.float64).copy(),
            "stored_B": np.asarray(saved["B"], dtype=np.float64).copy(),
            "stored_Z": np.asarray(saved["Z"], dtype=np.float64).copy(),
        }
    assert result["v"].shape == (len(active_ids), 3)
    assert result["u"].shape == rest.shape
    assert np.isfinite(result["v"]).all() and np.isfinite(result["u"]).all()
    s, c, b, z = tensors(result["v"])
    field_errors = {
        "C_max_abs_error": max_error(result["stored_C"], c, 2e-13),
        "B_max_abs_error": max_error(result["stored_B"], b, 2e-13),
        "Z_scale_aware_error": scaled_error(
            result["stored_Z"], z, atol=3e-13, rtol=2e-14
        ),
    }
    return {**result, "s": s, "C": c, "B": b, "Z": z, "field_errors": field_errors}


def detf(rest: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    ds = np.transpose((rest + u)[tets[:, 1:]] - (rest + u)[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ np.linalg.inv(dm))


def inversion_details(
    state: dict[str, Any],
    *,
    rest: np.ndarray,
    tets: np.ndarray,
    active_mask: np.ndarray,
    muscle: np.ndarray,
    fat: np.ndarray,
    aponeurosis: np.ndarray,
    muscle_id: np.ndarray,
    control_id: np.ndarray,
    original_cell_id: np.ndarray,
) -> dict[str, Any]:
    determinant = detf(rest, tets, state["u"])
    inverted = np.flatnonzero(determinant <= 0)
    return {
        "step": state["step"],
        "count": len(inverted),
        "tetrahedra": [
            {
                "global_tetrahedron_id": int(cell),
                "vtk_original_cell_id": int(original_cell_id[cell]),
                "detF": float(determinant[cell]),
                "is_active": bool(active_mask[cell]),
                "MuscleFraction": float(muscle[cell]),
                "FatFraction": float(fat[cell]),
                "AponeurosisFraction": float(aponeurosis[cell]),
                "MuscleId": int(muscle_id[cell]),
                "ActivationControlId": int(control_id[cell]),
            }
            for cell in inverted
        ],
    }


def active_graph(
    rest: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    region: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = tets[active_ids]
    face_pattern = np.asarray(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
    )
    faces = np.sort(active[:, face_pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(active_ids), dtype=np.int64), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(pair) == 1)
    i, j = owner[pair], owner[pair + 1]
    same = region[i] == region[j]
    i, j, face = i[same], j[same], faces[pair[same]]
    xyz = rest[face]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    centers = rest[active].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    frac = fraction[active_ids]
    weight = area / distance * (2 * frac[i] * frac[j] / (frac[i] + frac[j]))
    assert np.all(distance > 0) and np.all(weight > 0) and np.isfinite(weight).all()
    return i, j, weight


def transverse_directions(axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct deterministic transverse directions without using eigendecomposition."""
    least_aligned = np.argmin(np.abs(axes), axis=1)
    basis = np.eye(3)[least_aligned]
    first = np.cross(axes, basis)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(axes, first)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    return first, second


def recompute_metrics(
    state: dict[str, Any],
    *,
    initial_s: np.ndarray,
    initial_axes: np.ndarray,
    rest: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    active_volume: np.ndarray,
    pure_muscle: np.ndarray,
    top: np.ndarray,
    target: np.ndarray,
    area_weights: np.ndarray,
    fixed: np.ndarray,
    fixed_value: np.ndarray,
    graph: tuple[np.ndarray, np.ndarray, np.ndarray],
    diagnostics: StudyMetrics,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    v, u, s, c, b, z = (
        state["v"],
        state["u"],
        state["s"],
        state["C"],
        state["B"],
        state["Z"],
    )
    norms = np.sqrt(s)
    nonzero = s > 0
    axes = np.divide(v, norms[:, None], out=np.zeros_like(v), where=nonzero[:, None])
    compare = (initial_s > 0) & nonzero
    dot = np.clip(np.abs(np.sum(axes[compare] * initial_axes[compare], axis=1)), 0, 1)
    angles = np.degrees(np.arccos(dot))
    pred, target_top = u[top], target[top]
    error = pred - target_top
    determinant = detf(rest, tets, u)
    fixed_error = float(np.max(np.abs(u[fixed] - fixed_value[fixed])))
    assert fixed_error < 1e-14
    i, j, edge_weight = graph
    pair = compare[i] & compare[j]
    pair_dot = np.clip(np.einsum("ni,ni->n", axes[i], axes[j]), -1, 1)
    values: dict[str, float | int] = {
        "objective_mm2": float(np.mean(error**2) * 1e6),
        "uniform_fit_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(area_weights * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000 * np.sqrt(np.sum(area_weights * np.sum(pred**2, axis=1)))
        ),
        "target_projection": float(np.sum(pred * target_top) / np.sum(target_top**2)),
        "direction_drift_mean_deg": float(np.mean(angles)),
        "direction_drift_volume_weighted_mean_deg": float(
            np.average(angles, weights=active_volume[compare])
        ),
        "direction_drift_max_deg": float(np.max(angles)),
        "axis_neighbor_projector_jump_rms": float(
            np.sqrt(
                np.average(2 * (1 - pair_dot[pair] ** 2), weights=edge_weight[pair])
            )
        ),
        "C_neighbor_jump_rms": float(
            np.sqrt(
                np.average(np.sum((c[i] - c[j]) ** 2, axis=(1, 2)), weights=edge_weight)
            )
        ),
        "Z_neighbor_jump_rms": float(
            np.sqrt(
                np.average(np.sum((z[i] - z[j]) ** 2, axis=(1, 2)), weights=edge_weight)
            )
        ),
        "s_max": float(np.max(s)),
        "s_change_from_initial_rms": float(np.sqrt(np.mean((s - initial_s) ** 2))),
        "zero_strength_cells": int(np.sum(s == 0)),
        "initially_zero_now_positive_cells": int(np.sum((initial_s == 0) & (s > 0))),
        "detF_min": float(np.min(determinant)),
        "detF_max": float(np.max(determinant)),
        "inverted_all_cells": int(np.sum(determinant <= 0)),
        "inverted_active_cells": int(np.sum(determinant[active_ids] <= 0)),
        "inverted_pure_muscle_cells": int(np.sum((determinant <= 0) & pure_muscle)),
        "active_volume_weighted_rms_detF_minus_one": float(
            np.sqrt(
                np.average((determinant[active_ids] - 1) ** 2, weights=active_volume)
            )
        ),
        "fixed_max_error_m": fixed_error,
    }
    for percentile in (50, 90, 99):
        values[f"direction_drift_p{percentile}_deg"] = float(
            np.percentile(angles, percentile)
        )
    for angle in (15, 30, 60):
        values[f"direction_drift_volume_fraction_above_{angle}deg"] = float(
            np.average(angles > angle, weights=active_volume[compare])
        )
    for percentile in (0, 50, 90, 99, 100):
        values[f"active_axial_stretch_p{percentile}"] = float(
            np.percentile(1 / (1 + s), percentile)
        )
    values.update(diagnostics.evaluate_surface(u[diagnostics.skin_ids]))
    assert math.isclose(
        float(values["uniform_fit_rms_mm"]) ** 2,
        3 * float(values["objective_mm2"]),
        rel_tol=2e-12,
    )

    positive = np.flatnonzero(nonzero)
    first, second = transverse_directions(axes[positive])
    b_axis_actual = np.einsum("nij,nj->ni", b[positive], axes[positive])
    b_axis_expected = (1 + s[positive])[:, None] * axes[positive]
    b_first_actual = np.einsum("nij,nj->ni", b[positive], first)
    b_second_actual = np.einsum("nij,nj->ni", b[positive], second)
    z_axis_actual = np.einsum("nij,nj->ni", z[positive], axes[positive])
    z_axis_expected = (2 * s[positive] + s[positive] ** 2)[:, None] * axes[positive]
    z_first_actual = np.einsum("nij,nj->ni", z[positive], first)
    z_second_actual = np.einsum("nij,nj->ni", z[positive], second)
    b_scale = max(1.0, float(np.max(1 + s[positive])))
    z_scale = max(1.0, float(np.max(2 * s[positive] + s[positive] ** 2)))
    b_transverse_abs = float(
        max(
            np.max(np.abs(b_first_actual - first)),
            np.max(np.abs(b_second_actual - second)),
        )
    )
    z_transverse_abs = float(
        max(np.max(np.abs(z_first_actual)), np.max(np.abs(z_second_actual)))
    )
    direction_checks = {
        "axis_unit_max_abs_error": float(
            np.max(np.abs(np.linalg.norm(axes[positive], axis=1) - 1))
        ),
        "transverse_orthogonality_max_abs_error": float(
            max(
                np.max(np.abs(np.einsum("ni,ni->n", axes[positive], first))),
                np.max(np.abs(np.einsum("ni,ni->n", axes[positive], second))),
                np.max(np.abs(np.einsum("ni,ni->n", first, second))),
            )
        ),
        "B_axis_relation": scaled_error(
            b_axis_actual, b_axis_expected, atol=3e-13, rtol=2e-14
        ),
        "B_transverse_relation": {
            "max_abs_error": b_transverse_abs,
            "reference_scale": b_scale,
            "max_normalized_error": b_transverse_abs / b_scale,
            "absolute_plus_relative_tolerance": 3e-13 + 2e-14 * b_scale,
        },
        "Z_axis_relation": scaled_error(
            z_axis_actual, z_axis_expected, atol=3e-13, rtol=2e-14
        ),
        "Z_transverse_relation": {
            "max_abs_error": z_transverse_abs,
            "reference_scale": z_scale,
            "max_normalized_error": z_transverse_abs / z_scale,
            "absolute_plus_relative_tolerance": 3e-13 + 2e-14 * z_scale,
        },
    }
    assert direction_checks["axis_unit_max_abs_error"] < 5e-15, direction_checks
    assert direction_checks["transverse_orthogonality_max_abs_error"] < 5e-15, (
        direction_checks
    )
    assert b_transverse_abs <= 3e-13 + 2e-14 * b_scale, direction_checks
    assert z_transverse_abs <= 3e-13 + 2e-14 * z_scale, direction_checks
    assert state["physical_noninverted"] == (values["inverted_all_cells"] == 0)
    return values, direction_checks


def compare_metrics(
    actual: dict[str, float | int], reported: dict[str, Any], label: str
) -> dict[str, Any]:
    errors = []
    near_zero_angles = {}
    for key, value in actual.items():
        assert key in reported, (label, key)
        actual_value, reported_value = float(value), float(reported[key])
        error = abs(actual_value - reported_value)
        if (
            key.endswith("_deg")
            and max(abs(actual_value), abs(reported_value))
            <= NEAR_ZERO_ANGLE_TOLERANCE_DEG
        ):
            assert error <= NEAR_ZERO_ANGLE_TOLERANCE_DEG, (label, key, error)
            near_zero_angles[key] = {
                "recomputed_deg": actual_value,
                "reported_deg": reported_value,
                "absolute_error_deg": error,
                "absolute_tolerance_deg": NEAR_ZERO_ANGLE_TOLERANCE_DEG,
            }
        else:
            close(actual_value, reported_value, f"{label}/{key}")
        errors.append(error)
    return {
        "max_abs_error": max(errors, default=0.0),
        "near_zero_angle_checks": near_zero_angles,
    }


def main(cfg: Config) -> None:
    released, fixed = cfg.released_dir.resolve(), cfg.fixed_dir.resolve()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    assert cfg.expected_steps == 200

    paths = {
        "initial_converted": released / "initial-converted.npz",
        "initial": released / "initial.npz",
        "step_128": released / "step-0128.npz",
        "step_200": released / "step-0200.npz",
        "best": released / "best.npz",
        "best_noninverted": released / "best-noninverted.npz",
        "last": released / "last.npz",
        "trace_json": released / "trace.json",
        "trace_csv": released / "trace.csv",
        "summary": released / "summary.json",
        "protocol": released / "protocol.json",
        "gradient_validation": released / "gradient-validation.json",
        "optimizer_latest": released / "optimizer-latest.pt",
        "fixed_best": fixed / "best.npz",
        "fixed_initialization": fixed / "initialization.npz",
        "fixed_summary": fixed / "summary.json",
        "fixed_protocol": fixed / "protocol.json",
        "fixed_verification": FIXED_VERIFICATION,
    }
    assert all(path.is_file() for path in paths.values())
    summary = json.loads(paths["summary"].read_text())
    protocol = json.loads(paths["protocol"].read_text())
    trace = json.loads(paths["trace_json"].read_text())
    audit = json.loads(paths["gradient_validation"].read_text())
    fixed_receipt = json.loads(paths["fixed_verification"].read_text())
    fixed_protocol = json.loads(paths["fixed_protocol"].read_text())
    assert summary["status"] == COMPLETED_STATUS
    assert (
        summary["failure"] is None and summary["inverse_stationarity_claimed"] is False
    )
    assert (
        protocol["parameterization"]
        == "B=I+vvT; 3 real v components per active tetrahedron"
    )
    assert protocol["objective"].endswith("no regularizers or added barriers")
    assert protocol["optimizer"] == {
        "name": "Adam",
        "learning_rate": 0.3,
        "eps": 0.01,
        "betas": [0.9, 0.999],
        "weight_decay": 0,
        "steps": 200,
        "moments": "fresh; scalar moments are not transported to vector coordinates",
        "projection": "none",
    }
    assert protocol["materials"] == fixed_protocol["materials"]
    assert protocol["materials"]["muscle_model"] == "stable-active-physical-volume"

    assert audit["status"] == "passed" and protocol["gradient_validation"] == audit
    assert audit["normal_vs_tight_gradient_relative_difference"] < 0.02
    assert len(audit["checks"]) == 4
    assert {row["direction"] for row in audit["checks"]} == {"radial", "rotational"}
    assert {float(row["epsilon"]) for row in audit["checks"]} == {0.005, 0.0025}
    assert all(float(row["relative_error"]) < 0.02 for row in audit["checks"])
    assert all(
        row["forward"][side]["success"] for row in audit["checks"] for side in (0, 1)
    )

    mesh = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    diagnostics = StudyMetrics(FIXTURE)
    rest = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    active_ids = np.flatnonzero(
        np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    )
    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    muscle = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    fat = np.asarray(mesh.cell_data["FatFraction"], dtype=np.float64)
    aponeurosis = np.asarray(mesh.cell_data["AponeurosisFraction"], dtype=np.float64)
    control_id = np.asarray(mesh.cell_data["ActivationControlId"], dtype=np.int64)
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
    original_cell_id = np.asarray(mesh.cell_data["vtkOriginalCellIds"], dtype=np.int64)
    region = control_id[active_ids]
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    active_volume = volume[active_ids] * muscle[active_ids]
    pure_muscle = (muscle >= 1 - 1e-12) & (fat == 0) & (aponeurosis == 0)
    target = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
    top = np.flatnonzero(
        np.asarray(mesh.point_data["IsFace"], dtype=bool)
        & np.isfinite(target).all(axis=1)
    )
    fixed_mask = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(mesh.point_data["FixedValue"], dtype=np.float64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    skin_faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    assert np.all(skin_faces[:, 0] == 3)
    triangles = skin_ids[skin_faces[:, 1:]]
    xyz = rest[triangles]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    area = np.zeros(len(rest), dtype=np.float64)
    np.add.at(area, triangles.ravel(), np.repeat(triangle_area / 3, 3))
    area_weights = area[top] / np.sum(area[top])
    graph = active_graph(rest, tets, active_ids, region, muscle)
    assert len(rest) == 228_660 and len(tets) == 1_146_517
    assert len(active_ids) == 288_235 and len(top) == 15_302

    with np.load(paths["fixed_best"], allow_pickle=False) as saved:
        assert int(saved["step"]) == 400 and bool(saved["solver_valid"])
        fixed_s = np.asarray(saved["s"], dtype=np.float64).copy()
        fixed_u = np.asarray(saved["u"], dtype=np.float64).copy()
        assert np.array_equal(saved["active_ids"], active_ids)
    with np.load(paths["fixed_initialization"], allow_pickle=False) as saved:
        initial_axes = np.asarray(saved["axes"], dtype=np.float64).copy()
        assert np.array_equal(saved["active_ids"], active_ids)
        assert np.array_equal(saved["rest_points"], rest)
    expected_v0 = np.sqrt(fixed_s)[:, None] * initial_axes
    zero = fixed_s == 0
    assert (
        int(np.sum(zero)) == protocol["conversion"]["exact_zero_inactive_cells"] == 2305
    )

    converted = load_state(paths["initial_converted"], active_ids, rest)
    assert converted["step"] == 0
    assert np.array_equal(converted["v"], expected_v0)
    assert np.array_equal(converted["u"], fixed_u)
    converted_s_error = max_error(converted["s"], fixed_s, 2e-14)

    assert isinstance(trace, list) and len(trace) == cfg.expected_steps + 1
    steps = np.asarray([int(row["step"]) for row in trace], dtype=np.int64)
    assert np.array_equal(steps, np.arange(cfg.expected_steps + 1))
    assert summary["last_evaluated_step"] == cfg.expected_steps
    with paths["trace_csv"].open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(trace)
    assert [int(row["step"]) for row in csv_rows] == steps.tolist()
    for index, (csv_row, json_row) in enumerate(zip(csv_rows, trace, strict=True)):
        assert set(csv_row) == set(json_row)
        for key, value in json_row.items():
            close(float(csv_row[key]), float(value), f"trace-csv/{index}/{key}")

    running_best_step = min(
        range(len(trace)), key=lambda k: float(trace[k]["objective_mm2"])
    )
    noninverted_steps = [
        int(row["step"]) for row in trace if int(row["inverted_all_cells"]) == 0
    ]
    assert noninverted_steps
    running_best_noninverted_step = min(
        noninverted_steps, key=lambda k: float(trace[k]["objective_mm2"])
    )
    assert summary["best_step"] == running_best_step
    assert summary["best_noninverted_step"] == running_best_noninverted_step
    for row in trace:
        prefix = trace[: int(row["step"]) + 1]
        expected = min(prefix, key=lambda item: float(item["objective_mm2"]))
        assert int(row["best_step"]) == int(expected["step"])

    selected_paths = {
        "step0": paths["initial"],
        "step128": paths["step_128"],
        "step200": paths["step_200"],
        "best": paths["best"],
        "best_noninverted": paths["best_noninverted"],
        "last": paths["last"],
    }
    expected_selected_steps = {
        "step0": 0,
        "step128": 128,
        "step200": 200,
        "best": running_best_step,
        "best_noninverted": running_best_noninverted_step,
        "last": 200,
    }
    selected_receipts = {}
    state_cache = {}
    for name, path in selected_paths.items():
        state = load_state(path, active_ids, rest)
        assert state["step"] == expected_selected_steps[name]
        assert np.all(state["v"][zero] == 0)
        actual, directions = recompute_metrics(
            state,
            initial_s=fixed_s,
            initial_axes=initial_axes,
            rest=rest,
            tets=tets,
            active_ids=active_ids,
            active_volume=active_volume,
            pure_muscle=pure_muscle,
            top=top,
            target=target,
            area_weights=area_weights,
            fixed=fixed_mask,
            fixed_value=fixed_value,
            graph=graph,
            diagnostics=diagnostics,
        )
        metric_checks = compare_metrics(actual, trace[state["step"]], name)
        selected_receipts[name] = {
            "file": record(path),
            "step": state["step"],
            "metric_checks": metric_checks,
            "field_errors": state["field_errors"],
            "independent_direction_checks": directions,
            "zero_cells_unchanged": int(np.sum(np.all(state["v"][zero] == 0, axis=1))),
        }
        state_cache[name] = state
    assert np.array_equal(state_cache["step200"]["v"], state_cache["last"]["v"])
    assert np.array_equal(state_cache["step200"]["u"], state_cache["last"]["u"])
    assert np.array_equal(
        state_cache["best"]["v"], load_state(paths["best"], active_ids, rest)["v"]
    )

    for label, reported, row in (
        ("summary-best", summary["best_metrics"], trace[running_best_step]),
        ("summary-last", summary["last_metrics"], trace[-1]),
        (
            "summary-best-noninverted",
            summary["best_noninverted_metrics"],
            trace[running_best_noninverted_step],
        ),
    ):
        for key, value in reported.items():
            if isinstance(value, (int, float)):
                close(float(value), float(row[key]), f"{label}/{key}")
            else:
                assert value == row[key]

    inverted_steps = [
        int(row["step"]) for row in trace if row["inverted_all_cells"] > 0
    ]
    first_inverted_path = released / "first-inverted.npz"
    first_inverted = None
    inversion_receipts = {
        "best": inversion_details(
            state_cache["best"],
            rest=rest,
            tets=tets,
            active_mask=active_mask,
            muscle=muscle,
            fat=fat,
            aponeurosis=aponeurosis,
            muscle_id=muscle_id,
            control_id=control_id,
            original_cell_id=original_cell_id,
        ),
        "best_noninverted": inversion_details(
            state_cache["best_noninverted"],
            rest=rest,
            tets=tets,
            active_mask=active_mask,
            muscle=muscle,
            fat=fat,
            aponeurosis=aponeurosis,
            muscle_id=muscle_id,
            control_id=control_id,
            original_cell_id=original_cell_id,
        ),
    }
    if inverted_steps:
        assert first_inverted_path.is_file()
        first_state = load_state(first_inverted_path, active_ids, rest)
        assert first_state["step"] == inverted_steps[0]
        first_inverted = {
            "step": first_state["step"],
            "file": record(first_inverted_path),
        }
        inversion_receipts["first_inverted"] = inversion_details(
            first_state,
            rest=rest,
            tets=tets,
            active_mask=active_mask,
            muscle=muscle,
            fat=fat,
            aponeurosis=aponeurosis,
            muscle_id=muscle_id,
            control_id=control_id,
            original_cell_id=original_cell_id,
        )
        assert inversion_receipts["first_inverted"]["count"] == int(
            trace[inverted_steps[0]]["inverted_all_cells"]
        )
    else:
        assert not first_inverted_path.exists()
        inversion_receipts["first_inverted"] = None
    assert inversion_receipts["best"]["count"] == int(
        trace[running_best_step]["inverted_all_cells"]
    )
    assert inversion_receipts["best_noninverted"]["count"] == 0

    checkpoint = torch.load(
        paths["optimizer_latest"], map_location="cpu", weights_only=False
    )
    assert checkpoint["parameterization"] == PARAMETERIZATION
    assert int(checkpoint["source_fixed_step"]) == 400
    assert int(checkpoint["step"]) == cfg.expected_steps
    assert np.array_equal(checkpoint["active_ids"], active_ids)
    checkpoint_v = checkpoint["v"].detach().cpu().numpy()
    checkpoint_gradient = checkpoint["gradient"].detach().cpu().numpy()
    checkpoint_u = np.asarray(checkpoint["u"], dtype=np.float64)
    assert np.array_equal(checkpoint_v, state_cache["last"]["v"])
    assert np.array_equal(checkpoint_u, state_cache["last"]["u"])
    assert np.array_equal(checkpoint["initial_v"], expected_v0)
    assert np.array_equal(checkpoint["initial_axes"], initial_axes)
    assert np.array_equal(checkpoint["initial_s"], fixed_s)
    assert np.all(checkpoint_v[zero] == 0) and np.all(checkpoint_gradient[zero] == 0)
    assert np.isfinite(checkpoint_gradient).all()
    for key, state_name in (("best", "best"), ("best_noninverted", "best_noninverted")):
        stored = checkpoint[key]
        assert int(stored["step"]) == state_cache[state_name]["step"]
        assert np.array_equal(stored["v"], state_cache[state_name]["v"])
        assert np.array_equal(stored["u"], state_cache[state_name]["u"])
    optimizer = checkpoint["optimizer"]
    assert len(optimizer["param_groups"]) == 1 and len(optimizer["state"]) == 1
    group = optimizer["param_groups"][0]
    assert group["lr"] == 0.3 and group["eps"] == 0.01
    assert tuple(group["betas"]) == (0.9, 0.999)
    assert group["weight_decay"] == 0 and group["amsgrad"] is False
    adam = optimizer["state"][group["params"][0]]
    assert int(adam["step"].item()) == cfg.expected_steps
    for key in ("exp_avg", "exp_avg_sq"):
        assert tuple(adam[key].shape) == expected_v0.shape
        assert adam[key].dtype == torch.float64 and torch.isfinite(adam[key]).all()
        assert torch.all(adam[key][torch.as_tensor(zero)] == 0)
    assert torch.all(adam["exp_avg_sq"] >= 0)

    shared_sources = (
        "experiment_profile",
        "study_metrics",
        "study_physics",
        "volume_preserving_active",
    )
    source_checks = {}
    for name in ("__main__", "__mp_main__", "tensor_active", *shared_sources):
        reported = protocol["sources"][name]
        current = record(artifact_path(reported["path"]))
        snapshot = record(artifact_path(reported["snapshot"]))
        assert current["sha256"] == snapshot["sha256"] == reported["sha256"]
        assert current["bytes"] == snapshot["bytes"] == reported["bytes"]
        source_checks[name] = {"current": current, "executed_snapshot": snapshot}
    assert source_checks["__main__"] == source_checks["__mp_main__"]
    for name in shared_sources:
        fixed_source = fixed_receipt["executed_inverse_sources"][name][
            "executed_snapshot"
        ]
        assert (
            source_checks[name]["executed_snapshot"]["sha256"] == fixed_source["sha256"]
        )
        assert (
            source_checks[name]["executed_snapshot"]["bytes"] == fixed_source["bytes"]
        )
    fixed_tensor = fixed_protocol["sources"]["tensor_active"]
    fixed_tensor_snapshot = record(artifact_path(fixed_tensor["snapshot"]))
    assert (
        source_checks["tensor_active"]["executed_snapshot"]["sha256"]
        == fixed_tensor_snapshot["sha256"]
    )
    assert (
        source_checks["tensor_active"]["executed_snapshot"]["bytes"]
        == fixed_tensor_snapshot["bytes"]
    )
    source_checks["fixed_tensor_active_snapshot"] = fixed_tensor_snapshot

    source_dir = out / "sources"
    source_dir.mkdir()
    verifier_snapshot = source_dir / Path(__file__).name
    shutil.copy2(__file__, verifier_snapshot)
    receipt = {
        "status": "passed",
        "scope": "CPU-only independent verification of saved released-axis artifacts; no solve, derivative rerun, or optimization",
        "inputs": {name: record(path) for name, path in paths.items()},
        "fixture": {
            "volume": record(FIXTURE / "volume.vtu"),
            "skin": record(FIXTURE / "skin.vtp"),
            "summary": record(FIXTURE / "summary.json"),
            "points": len(rest),
            "tetrahedra": len(tets),
            "active_cells": len(active_ids),
            "target_vertices": len(top),
        },
        "initial_conversion": {
            "v_exact": True,
            "u_exact": True,
            "s_max_abs_error": converted_s_error,
            "zero_cells": int(np.sum(zero)),
            "field_errors": converted["field_errors"],
        },
        "trace": {
            "rows": len(trace),
            "steps_exact_0_through_200": True,
            "csv_matches_json": True,
            "best_step": running_best_step,
            "best_noninverted_step": running_best_noninverted_step,
            "inverted_steps": inverted_steps,
            "first_inverted": first_inverted,
        },
        "selected_states": selected_receipts,
        "inversion_details": inversion_receipts,
        "optimizer_checkpoint": {
            "file": record(paths["optimizer_latest"]),
            "step": int(adam["step"].item()),
            "v_matches_last": True,
            "u_matches_last": True,
            "zero_cells_and_moments_unchanged": True,
        },
        "gradient_validation": {
            "file": record(paths["gradient_validation"]),
            "matches_protocol": True,
            "directions": sorted({row["direction"] for row in audit["checks"]}),
            "epsilons": sorted({row["epsilon"] for row in audit["checks"]}),
            "maximum_relative_error": max(
                row["relative_error"] for row in audit["checks"]
            ),
            "normal_vs_tight_gradient_relative_difference": audit[
                "normal_vs_tight_gradient_relative_difference"
            ],
        },
        "source_identity": source_checks,
        "verifier_source_snapshot": record(verifier_snapshot),
        "limitations": [
            "Verification recomputes saved-state algebra, geometry, and metrics but does not rerun mechanics or derivatives.",
            "The 2305 exact-zero vector controls cannot reactivate under B=I+vvT.",
            "Completion of 200 Adam updates is not a stationarity certificate.",
        ],
    }
    write_json(out / "receipt.json", receipt)
    cherries.log_metrics(
        {
            "verified_states": len(selected_receipts),
            "trace_rows": len(trace),
            "zero_cells": int(np.sum(zero)),
            "max_gradient_relative_error": receipt["gradient_validation"][
                "maximum_relative_error"
            ],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
