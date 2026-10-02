"""Independently verify the completed fixed-direction inverse artifacts on CPU."""

# ruff: noqa: EM102, PLR0915, PT018, TRY003

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
FORWARD = GROUP / "data/10-forward"
INVERSE = GROUP / "data/40-fixed-directions"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
COMPLETED_STATUS = "completed_budget_not_stationarity_certified"
PARAMETERIZATION = "B=I+s*n*nT;s>=0;fixed-reference-axes"
CORE_METRICS = (
    "objective_mm2",
    "uniform_fit_rms_mm",
    "uniform_motion_rms_mm",
    "area_weighted_fit_rms_mm",
    "area_weighted_motion_rms_mm",
    "target_projection",
    "s_min",
    "s_max",
    "s_change_from_initial_rms",
    "zero_strength_cells",
    "initially_zero_now_positive_cells",
    "detF_min",
    "detF_max",
    "inverted_all_cells",
    "inverted_active_cells",
    "inverted_pure_muscle_cells",
    "active_volume_weighted_rms_detF_minus_one",
    "fixed_max_error_m",
    "active_axial_stretch_p0",
    "active_axial_stretch_p50",
    "active_axial_stretch_p90",
    "active_axial_stretch_p99",
    "active_axial_stretch_p100",
)
LATEST_GRADIENT_METRICS = (
    "raw_gradient_rms",
    "projected_gradient_rms",
    "projected_gradient_max",
    "kkt_residual_rms",
    "kkt_residual_max",
)
TAIL_PERCENTILES = (0, 1, 5, 10, 50, 90, 99, 100)
TAIL_THRESHOLDS = (0.5, 0.65)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("60-verification", mkdir=True)
    inverse_dir: Path = INVERSE
    expected_steps: int = 200


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def detf(rest: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    ds = np.transpose((rest + u)[tets[:, 1:]] - (rest + u)[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ np.linalg.inv(dm))


def fields(s: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projector = axes[:, :, None] * axes[:, None, :]
    b = np.eye(3) + s[:, None, None] * projector
    z = (2.0 * s + s * s)[:, None, None] * projector
    return b, z


def axial_stretch_tail(s: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    stretch = 1.0 / (1.0 + s)
    assert stretch.shape == weights.shape
    assert np.isfinite(stretch).all() and np.all(stretch > 0.0)
    assert np.isfinite(weights).all() and np.all(weights > 0.0)
    order = np.argsort(stretch, kind="stable")
    sorted_stretch = stretch[order]
    cumulative_weight = np.cumsum(weights[order])
    total_weight = float(cumulative_weight[-1])
    weighted_percentiles = {}
    for percentile in TAIL_PERCENTILES:
        index = int(
            np.searchsorted(
                cumulative_weight,
                percentile / 100.0 * total_weight,
                side="left",
            )
        )
        weighted_percentiles[f"p{percentile}"] = float(
            sorted_stretch[min(index, len(sorted_stretch) - 1)]
        )
    return {
        "unweighted_percentiles": {
            f"p{percentile}": float(np.percentile(stretch, percentile))
            for percentile in TAIL_PERCENTILES
        },
        "active_muscle_volume_weighted_percentiles": weighted_percentiles,
        "strictly_below_threshold": {
            str(threshold): {
                "cell_fraction": float(np.mean(stretch < threshold)),
                "active_muscle_volume_fraction": float(
                    np.sum(weights[stretch < threshold]) / total_weight
                ),
            }
            for threshold in TAIL_THRESHOLDS
        },
    }


def load_state(path: Path, active_ids: np.ndarray, axes_hash: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "s",
            "u",
            "step",
            "active_ids",
            "axes_sha256",
            "solver_valid",
            "physical_volume_energy",
        }
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert np.array_equal(saved["active_ids"], active_ids)
        assert str(saved["axes_sha256"]) == axes_hash
        result = {
            "s": np.asarray(saved["s"], dtype=np.float64).copy(),
            "u": np.asarray(saved["u"], dtype=np.float64).copy(),
            "step": int(saved["step"]),
        }
    assert result["s"].shape == (len(active_ids),)
    assert np.isfinite(result["s"]).all() and np.min(result["s"]) >= 0.0
    assert np.isfinite(result["u"]).all()
    return result


def state_metrics(
    *,
    state: dict[str, Any],
    initial_s: np.ndarray,
    axes: np.ndarray,
    rest: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    active_volume: np.ndarray,
    pure_muscle: np.ndarray,
    top: np.ndarray,
    target: np.ndarray,
    area: np.ndarray,
    fixed: np.ndarray,
    fixed_value: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    s = state["s"]
    u = state["u"]
    assert u.shape == rest.shape
    pred = u[top]
    target_top = target[top]
    error = pred - target_top
    determinant = detf(rest, tets, u)
    b, z = fields(s, axes)
    projector = axes[:, :, None] * axes[:, None, :]
    b_values = np.linalg.eigvalsh(b)
    z_values = np.linalg.eigvalsh(z)
    expected_b_values = np.stack((np.ones_like(s), np.ones_like(s), 1.0 + s), axis=-1)
    expected_z_values = np.stack(
        (np.zeros_like(s), np.zeros_like(s), 2.0 * s + s * s), axis=-1
    )
    b_spectrum_error = float(np.max(np.abs(b_values - expected_b_values)))
    z_spectrum_error = float(np.max(np.abs(z_values - expected_z_values)))
    b_axis_error = float(
        np.max(np.abs(np.einsum("nij,nj->ni", b, axes) - (1.0 + s)[:, None] * axes))
    )
    projector_error = float(np.max(np.abs(projector @ projector - projector)))
    assert b_spectrum_error < 2e-12
    assert z_spectrum_error < 2e-11
    assert b_axis_error < 2e-12
    assert projector_error < 2e-15
    fixed_error = float(np.max(np.abs(u[fixed] - fixed_value[fixed])))
    assert fixed_error < 1e-14
    result: dict[str, float | int] = {
        "objective_mm2": float(np.mean(error**2) * 1e6),
        "uniform_fit_rms_mm": float(
            1000.0 * np.sqrt(np.mean(np.sum(error**2, axis=1)))
        ),
        "uniform_motion_rms_mm": float(
            1000.0 * np.sqrt(np.mean(np.sum(pred**2, axis=1)))
        ),
        "area_weighted_fit_rms_mm": float(
            1000.0 * np.sqrt(np.sum(area * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000.0 * np.sqrt(np.sum(area * np.sum(pred**2, axis=1)))
        ),
        "target_projection": float(np.sum(pred * target_top) / np.sum(target_top**2)),
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "s_change_from_initial_rms": float(np.sqrt(np.mean((s - initial_s) ** 2))),
        "zero_strength_cells": int(np.sum(s == 0.0)),
        "initially_zero_now_positive_cells": int(
            np.sum((initial_s == 0.0) & (s > 0.0))
        ),
        "detF_min": float(determinant.min()),
        "detF_max": float(determinant.max()),
        "inverted_all_cells": int(np.sum(determinant <= 0.0)),
        "inverted_active_cells": int(np.sum(determinant[active_ids] <= 0.0)),
        "inverted_pure_muscle_cells": int(np.sum((determinant <= 0.0) & pure_muscle)),
        "active_volume_weighted_rms_detF_minus_one": float(
            np.sqrt(
                np.average(
                    (determinant[active_ids] - 1.0) ** 2,
                    weights=active_volume,
                )
            )
        ),
        "fixed_max_error_m": fixed_error,
    }
    for quantile in (0, 50, 90, 99, 100):
        result[f"active_axial_stretch_p{quantile}"] = float(
            np.percentile(1.0 / (1.0 + s), quantile)
        )
    mechanics = {
        "positive_strength_cells_with_exactly_one_contractile_mode": int(
            np.sum(s > 0.0)
        ),
        "zero_strength_cells_with_no_contractile_mode": int(np.sum(s == 0.0)),
        "B_spectrum_max_abs_error": b_spectrum_error,
        "Z_spectrum_max_abs_error": z_spectrum_error,
        "B_axis_relation_max_abs_error": b_axis_error,
        "projector_idempotence_max_abs_error": projector_error,
        "B_min_eigenvalue": float(b_values.min()),
        "natural_axial_stretch_min": float(np.min(1.0 / (1.0 + s))),
        "natural_axial_stretch_max": float(np.max(1.0 / (1.0 + s))),
        "axial_stretch_tail": axial_stretch_tail(s, active_volume),
    }
    assert math.isclose(
        float(result["uniform_fit_rms_mm"]) ** 2,
        3.0 * float(result["objective_mm2"]),
        rel_tol=2e-12,
    )
    return result, mechanics


def compare_metrics(
    actual: dict[str, float | int], reported: dict[str, Any], label: str
) -> float:
    errors = [
        close(float(actual[key]), float(reported[key]), f"{label}/{key}")
        for key in CORE_METRICS
    ]
    return max(errors, default=0.0)


def main(cfg: Config) -> None:
    assert cfg.expected_steps > 0
    inverse = cfg.inverse_dir.resolve()
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output must be empty: {out}"

    input_paths = {
        "initialization": inverse / "initialization.npz",
        "best": inverse / "best.npz",
        "last": inverse / "last.npz",
        "trace": inverse / "trace.json",
        "summary": inverse / "summary.json",
        "optimizer_latest": inverse / "optimizer-latest.pt",
        "protocol": inverse / "protocol.json",
        "gradient_validation": inverse / "gradient-validation.json",
        "full": FORWARD / "baseline-replay.npz",
        "dominant": FORWARD / "dominant-only.npz",
    }
    assert all(path.is_file() for path in input_paths.values())
    summary = json.loads(input_paths["summary"].read_text())
    protocol = json.loads(input_paths["protocol"].read_text())
    trace = json.loads(input_paths["trace"].read_text())
    gradient_validation = json.loads(input_paths["gradient_validation"].read_text())
    assert summary["status"] == COMPLETED_STATUS
    assert summary["failure"] is None
    assert summary["inverse_stationarity_claimed"] is False
    assert gradient_validation["status"] == "passed"
    assert gradient_validation["chain_check"]["status"] == "passed"
    assert len(gradient_validation["checks"]) == 4
    assert {row["direction"] for row in gradient_validation["checks"]} == {
        "signed_by_muscle",
        "gradient_sign",
    }
    assert {float(row["epsilon"]) for row in gradient_validation["checks"]} == {
        0.005,
        0.0025,
    }
    assert all(
        float(row["relative_error"]) <= 0.02 for row in gradient_validation["checks"]
    )

    mesh = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    rest = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    assert np.all(cells[:, 0] == 4)
    tets = cells[:, 1:]
    active_ids = np.flatnonzero(
        np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    )
    muscle = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    fat = np.asarray(mesh.cell_data["FatFraction"], dtype=np.float64)
    aponeurosis = np.asarray(mesh.cell_data["AponeurosisFraction"], dtype=np.float64)
    pure_muscle = (muscle >= 1.0 - 1e-12) & (fat == 0.0) & (aponeurosis == 0.0)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    active_volume = volume[active_ids] * muscle[active_ids]
    target = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
    top = np.flatnonzero(
        np.asarray(mesh.point_data["IsFace"], dtype=bool)
        & np.isfinite(target).all(axis=1)
    )
    fixed = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(mesh.point_data["FixedValue"], dtype=np.float64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    triangles = skin_ids[np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]]
    triangle_points = rest[triangles]
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        ),
        axis=1,
    )
    area = np.zeros(len(rest), dtype=np.float64)
    np.add.at(area, triangles.ravel(), np.repeat(triangle_area / 3.0, 3))
    area = area[top]
    area /= area.sum()
    assert len(rest) == 228_660 and len(tets) == 1_146_517
    assert len(active_ids) == 288_235 and len(top) == 15_302
    assert np.all(active_volume > 0.0) and np.isfinite(active_volume).all()

    with np.load(input_paths["full"], allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert np.array_equal(saved["active_ids"], active_ids)
        assert np.array_equal(saved["rest_points"], rest)
        full_b = np.asarray(saved["B"], dtype=np.float64)
        full_z = np.asarray(saved["Z"], dtype=np.float64)
    full_z_error = float(
        np.max(np.abs(full_b @ full_b.swapaxes(-1, -2) - np.eye(3) - full_z))
    )
    assert full_z_error < 2e-13
    eigenvalues, eigenvectors = np.linalg.eigh(full_z)

    with np.load(input_paths["initialization"], allow_pickle=False) as saved:
        assert set(saved.files) == {
            "axes",
            "initial_s",
            "active_ids",
            "rest_points",
            "eigenvalues_Z0",
            "axes_sha256",
        }
        axes = np.asarray(saved["axes"], dtype=np.float64).copy()
        initial_s = np.asarray(saved["initial_s"], dtype=np.float64).copy()
        stored_eigenvalues = np.asarray(saved["eigenvalues_Z0"], dtype=np.float64)
        assert np.array_equal(saved["active_ids"], active_ids)
        assert np.array_equal(saved["rest_points"], rest)
        stored_axes_hash = str(saved["axes_sha256"])
    assert axes.shape == (len(active_ids), 3)
    assert initial_s.shape == (len(active_ids),)
    assert np.max(np.abs(np.linalg.norm(axes, axis=1) - 1.0)) < 1e-14
    axes_hash = array_digest(axes)
    assert stored_axes_hash == axes_hash == protocol["frozen_axes_sha256"]
    assert np.max(np.abs(stored_eigenvalues - eigenvalues)) < 2e-13
    top_eigen_residual = float(
        np.max(
            np.abs(
                np.einsum("nij,nj->ni", full_z, axes) - eigenvalues[:, -1, None] * axes
            )
        )
    )
    projector = axes[:, :, None] * axes[:, None, :]
    recomputed_projector = eigenvectors[:, :, -1, None] * eigenvectors[:, None, :, -1]
    projector_error = float(np.max(np.abs(projector - recomputed_projector)))
    assert top_eigen_residual < 3e-13
    assert projector_error < 3e-13
    expected_initial_s = np.sqrt(1.0 + np.maximum(eigenvalues[:, -1], 0.0)) - 1.0
    assert np.max(np.abs(initial_s - expected_initial_s)) < 2e-15

    with np.load(input_paths["dominant"], allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert np.array_equal(saved["active_ids"], active_ids)
        assert np.array_equal(saved["rest_points"], rest)
        dominant_b = np.asarray(saved["B"], dtype=np.float64)
        dominant_z = np.asarray(saved["Z"], dtype=np.float64)
    initial_b, initial_z = fields(initial_s, axes)
    initial_b_error = float(np.max(np.abs(initial_b - dominant_b)))
    initial_z_error = float(np.max(np.abs(initial_z - dominant_z)))
    assert initial_b_error < 1e-12 and initial_z_error < 1e-12

    assert protocol["trainable_parameter_count"] == len(active_ids)
    assert protocol["parameterization"].startswith("B_i=I+s_i*n_i*n_iT")
    assert protocol["objective"] == (
        "uniform finite-IsFace Cartesian component MSE times1e6; no regularizers"
    )
    assert protocol["materials"]["muscle_model"] == "stable-active-physical-volume"
    assert protocol["materials"]["skin_E_MPa"] == 0.0
    assert protocol["materials"]["contact_enabled"] is False
    assert protocol["optimizer"]["name"] == "Adam"
    assert protocol["optimizer"]["learning_rate"] == 0.3
    assert protocol["optimizer"]["eps"] == 0.01
    assert protocol["optimizer"]["betas"] == [0.9, 0.999]
    assert protocol["optimizer"]["weight_decay"] == 0
    assert protocol["optimizer"]["steps"] == cfg.expected_steps
    assert protocol["inputs"]["full"]["sha256"] == record(input_paths["full"])["sha256"]
    assert (
        protocol["inputs"]["dominant"]["sha256"]
        == record(input_paths["dominant"])["sha256"]
    )
    near_repeated = eigenvalues[:, -1] - eigenvalues[:, -2] <= 1e-6 * np.maximum(
        1.0, np.abs(eigenvalues[:, -1])
    )
    assert protocol["unsupported_positive_axis_cells"] == int(np.sum(initial_s == 0.0))
    assert protocol["near_repeated_top_axis_cells"] == int(np.sum(near_repeated))

    executed_sources = {}
    for name in (
        "__main__",
        "experiment_profile",
        "study_metrics",
        "study_physics",
        "volume_preserving_active",
    ):
        source = protocol["sources"][name]
        snapshot = Path(source["snapshot"])
        snapshot_record = record(snapshot)
        assert snapshot_record["sha256"] == source["sha256"]
        assert snapshot_record["bytes"] == source["bytes"]
        executed_sources[name] = {
            "reported_source_path": source["path"],
            "executed_snapshot": snapshot_record,
        }
    material_source = Path(
        protocol["sources"]["volume_preserving_active"]["snapshot"]
    ).read_text()
    assert "G = F @ A_inv" in material_source
    assert material_source.count("J = func.I3(F)") == 5
    assert "func.I3(G)" not in material_source
    assert "math.square(J - F.dtype(1.0))" in material_source
    inverse_source = Path(protocol["sources"]["__main__"]["snapshot"]).read_text()
    assert "return s[:, None] * torch.stack" in inverse_source
    assert "s.clamp_min_(0)" in inverse_source
    assert "weight_decay=0" in inverse_source and "amsgrad=False" in inverse_source

    assert isinstance(trace, list) and trace
    steps = np.asarray([int(row["step"]) for row in trace], dtype=np.int64)
    assert np.array_equal(steps, np.arange(steps[-1] + 1))
    assert int(steps[-1]) == summary["last_evaluated_step"] == cfg.expected_steps
    elapsed = np.asarray([float(row["elapsed_seconds"]) for row in trace])
    assert np.isfinite(elapsed).all() and np.all(np.diff(elapsed) > 0.0)
    running_best = None
    running_step = None
    for row in trace:
        value = float(row["objective_mm2"])
        if running_best is None or value < running_best:
            running_best = value
            running_step = int(row["step"])
        assert int(row["best_step"]) == running_step
    assert running_step == summary["best_step"]
    best_row = trace[int(summary["best_step"])]
    last_row = trace[-1]
    for label, reported, row in (
        ("summary-best", summary["best_metrics"], best_row),
        ("summary-last", summary["last_metrics"], last_row),
    ):
        assert set(reported).issubset(row)
        for key, value in reported.items():
            if isinstance(value, (int, float)):
                close(float(value), float(row[key]), f"{label}/{key}")
            else:
                assert value == row[key]

    best_state = load_state(input_paths["best"], active_ids, axes_hash)
    last_state = load_state(input_paths["last"], active_ids, axes_hash)
    assert best_state["step"] == summary["best_step"]
    assert last_state["step"] == summary["last_evaluated_step"]
    best_metrics, best_mechanics = state_metrics(
        state=best_state,
        initial_s=initial_s,
        axes=axes,
        rest=rest,
        tets=tets,
        active_ids=active_ids,
        active_volume=active_volume,
        pure_muscle=pure_muscle,
        top=top,
        target=target,
        area=area,
        fixed=fixed,
        fixed_value=fixed_value,
    )
    last_metrics, last_mechanics = state_metrics(
        state=last_state,
        initial_s=initial_s,
        axes=axes,
        rest=rest,
        tets=tets,
        active_ids=active_ids,
        active_volume=active_volume,
        pure_muscle=pure_muscle,
        top=top,
        target=target,
        area=area,
        fixed=fixed,
        fixed_value=fixed_value,
    )
    best_metric_error = compare_metrics(best_metrics, best_row, "best")
    last_metric_error = compare_metrics(last_metrics, last_row, "last")

    checkpoint = torch.load(
        input_paths["optimizer_latest"], map_location="cpu", weights_only=False
    )
    assert checkpoint["parameterization"] == PARAMETERIZATION
    assert checkpoint["axes_sha256"] == axes_hash
    assert np.array_equal(checkpoint["active_ids"], active_ids)
    assert int(checkpoint["step"]) == last_state["step"]
    checkpoint_s = checkpoint["s"].detach().cpu().numpy()
    checkpoint_gradient = checkpoint["gradient"].detach().cpu().numpy()
    checkpoint_u = np.asarray(checkpoint["u"], dtype=np.float64)
    assert checkpoint_s.shape == checkpoint_gradient.shape == (len(active_ids),)
    assert checkpoint_u.shape == rest.shape
    assert np.isfinite(checkpoint_s).all() and np.min(checkpoint_s) >= 0.0
    assert np.isfinite(checkpoint_gradient).all() and np.isfinite(checkpoint_u).all()
    assert np.array_equal(checkpoint_s, last_state["s"])
    assert np.array_equal(checkpoint_u, last_state["u"])
    checkpoint_best = checkpoint["best"]
    assert int(checkpoint_best["step"]) == best_state["step"]
    assert np.array_equal(checkpoint_best["s"], best_state["s"])
    assert np.array_equal(checkpoint_best["u"], best_state["u"])

    optimizer = checkpoint["optimizer"]
    assert len(optimizer["param_groups"]) == 1 and len(optimizer["state"]) == 1
    group = optimizer["param_groups"][0]
    assert len(group["params"]) == 1
    assert group["lr"] == 0.3 and group["eps"] == 0.01
    assert tuple(group["betas"]) == (0.9, 0.999)
    assert group["weight_decay"] == 0 and group["amsgrad"] is False
    state = optimizer["state"][group["params"][0]]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert int(state["step"].item()) == last_state["step"]
    for name in ("exp_avg", "exp_avg_sq"):
        tensor = state[name]
        assert tuple(tensor.shape) == (len(active_ids),)
        assert tensor.dtype == torch.float64
        assert torch.isfinite(tensor).all()
    assert torch.all(state["exp_avg_sq"] >= 0.0)

    parent_verification = None
    if protocol["parent"] is not None:
        parent = protocol["parent"]
        parent_checkpoint_path = Path(parent["path"])
        parent_checkpoint_record = record(parent_checkpoint_path)
        assert parent_checkpoint_record["sha256"] == parent["sha256"]
        assert parent_checkpoint_record["bytes"] == parent["bytes"]
        parent_checkpoint = torch.load(
            parent_checkpoint_path, map_location="cpu", weights_only=False
        )
        parent_step = int(parent["step"])
        assert 0 <= parent_step < cfg.expected_steps
        assert int(parent_checkpoint["step"]) == parent_step
        assert parent_checkpoint["axes_sha256"] == axes_hash
        assert np.array_equal(parent_checkpoint["active_ids"], active_ids)
        parent_optimizer = parent_checkpoint["optimizer"]
        assert len(parent_optimizer["param_groups"]) == 1
        parent_group = parent_optimizer["param_groups"][0]
        assert len(parent_group["params"]) == 1
        parent_state = parent_optimizer["state"][parent_group["params"][0]]
        assert int(parent_state["step"].item()) == parent_step
        parent_trace_path = parent_checkpoint_path.parent / "trace.json"
        parent_trace = json.loads(parent_trace_path.read_text())
        parent_prefix = [row for row in parent_trace if int(row["step"]) <= parent_step]
        assert len(parent_prefix) == parent_step + 1
        assert trace[: parent_step + 1] == parent_prefix
        parent_verification = {
            "checkpoint": parent_checkpoint_record,
            "step": parent_step,
            "adam_counter": int(parent_state["step"].item()),
            "axes_sha256": parent_checkpoint["axes_sha256"],
            "trace": record(parent_trace_path),
            "inherited_trace_rows": len(parent_prefix),
            "inherited_trace_prefix_exact": True,
        }

    mapping = checkpoint_s - np.maximum(checkpoint_s - checkpoint_gradient, 0.0)
    kkt = np.where(
        checkpoint_s > 0.0,
        checkpoint_gradient,
        np.minimum(checkpoint_gradient, 0.0),
    )
    gradient_metrics = {
        "raw_gradient_rms": float(np.sqrt(np.mean(checkpoint_gradient**2))),
        "projected_gradient_rms": float(np.sqrt(np.mean(mapping**2))),
        "projected_gradient_max": float(np.max(np.abs(mapping))),
        "kkt_residual_rms": float(np.sqrt(np.mean(kkt**2))),
        "kkt_residual_max": float(np.max(np.abs(kkt))),
    }
    gradient_metric_errors = {
        key: close(gradient_metrics[key], float(last_row[key]), f"latest/{key}")
        for key in LATEST_GRADIENT_METRICS
    }
    initial_projected = float(trace[0]["projected_gradient_rms"])
    close(
        float(last_row["relative_projected_gradient"]),
        gradient_metrics["projected_gradient_rms"] / initial_projected,
        "latest/relative_projected_gradient",
    )

    source_dir = out / "sources"
    source_dir.mkdir()
    source_snapshot = source_dir / Path(__file__).name
    shutil.copy2(__file__, source_snapshot)
    receipt = {
        "status": "passed",
        "scope": "CPU-only verification of completed fixed-direction inverse artifacts; no mechanics solve or optimization",
        "inputs": {name: record(path) for name, path in input_paths.items()},
        "executed_inverse_sources": executed_sources,
        "verifier_source_snapshot": record(source_snapshot),
        "fixture_contract": {
            "points": len(rest),
            "tetrahedra": len(tets),
            "active_cells": len(active_ids),
            "target_vertices": len(top),
            "fixed_coordinate_dofs": int(fixed.sum()),
            "trainable_scalars": len(checkpoint_s),
            "scalars_per_active_cell": len(checkpoint_s) / len(active_ids),
        },
        "frozen_axis_contract": {
            "axes_sha256": axes_hash,
            "full_BBt_minus_I_vs_Z_max_abs_error": full_z_error,
            "top_eigenvector_residual_max_abs_error": top_eigen_residual,
            "stored_vs_recomputed_top_projector_max_abs_error": projector_error,
            "initial_B_vs_dominant_max_abs_error": initial_b_error,
            "initial_Z_vs_dominant_max_abs_error": initial_z_error,
            "unsupported_positive_axis_cells": int(np.sum(initial_s == 0.0)),
            "near_repeated_top_axis_cells": int(np.sum(near_repeated)),
        },
        "axial_stretch_tail_summaries": {
            "quantity": "Natural axial stretch 1/(1+s) for each active tetrahedron",
            "unweighted_percentile_convention": "NumPy percentile with its default linear interpolation",
            "weighted_percentile_convention": "Inverse weighted ECDF: inf{x: cumulative active-muscle-volume fraction at or below x >= p}; stable ascending sort",
            "threshold_convention": "Strict stretch<threshold; 0.5 and 0.65 are descriptive tail summaries, not physiological thresholds or imposed bounds",
            "initialization": axial_stretch_tail(initial_s, active_volume),
            "best": best_mechanics["axial_stretch_tail"],
            "last": last_mechanics["axial_stretch_tail"],
        },
        "gradient_validation": {
            "status": gradient_validation["status"],
            "directions": sorted(
                {row["direction"] for row in gradient_validation["checks"]}
            ),
            "epsilons": sorted(
                {float(row["epsilon"]) for row in gradient_validation["checks"]},
                reverse=True,
            ),
            "maximum_relative_error": max(
                float(row["relative_error"]) for row in gradient_validation["checks"]
            ),
        },
        "trace_contract": {
            "first_step": int(steps[0]),
            "last_step": int(steps[-1]),
            "rows": len(trace),
            "strictly_increasing_elapsed_time": True,
            "recomputed_best_step": running_step,
            "recomputed_best_objective_mm2": running_best,
            "parent": parent_verification,
        },
        "best": {
            "step": best_state["step"],
            "metrics": best_metrics,
            "mechanics": best_mechanics,
            "reported_core_metric_max_abs_error": best_metric_error,
        },
        "last": {
            "step": last_state["step"],
            "metrics": last_metrics,
            "mechanics": last_mechanics,
            "reported_core_metric_max_abs_error": last_metric_error,
        },
        "optimizer_checkpoint": {
            "step": int(checkpoint["step"]),
            "adam_counter": int(state["step"].item()),
            "parameter_shape": list(checkpoint_s.shape),
            "gradient_shape": list(checkpoint_gradient.shape),
            "first_moment_shape": list(state["exp_avg"].shape),
            "second_moment_shape": list(state["exp_avg_sq"].shape),
            "latest_gradient_metrics": gradient_metrics,
            "reported_gradient_metric_max_abs_error": max(
                gradient_metric_errors.values(), default=0.0
            ),
        },
        "material_contract": {
            "model": protocol["materials"]["muscle_model"],
            "energy": "mu/2*||F B||^2 plus determinant terms evaluated on physical det(F)",
            "skin_E_MPa": protocol["materials"]["skin_E_MPa"],
            "contact_enabled": protocol["materials"]["contact_enabled"],
            "source_snapshot_uses_physical_J": True,
        },
        "limitations": [
            "This receipt verifies stored arrays, metrics, gradients, optimizer state, and executed-source provenance; it does not rerun equilibrium or the adjoint.",
            "A completed 200-update budget is not a stationarity certificate.",
            "The frozen axes were inferred in-sample from the full fit; this verification does not establish anatomical validity or superiority to alternative axes.",
        ],
    }
    receipt_path = out / "receipt.json"
    write_json(receipt_path, receipt)
    cherries.log_output(receipt_path)
    cherries.log_output(source_snapshot)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
