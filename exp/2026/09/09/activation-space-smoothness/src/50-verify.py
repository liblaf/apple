# ruff: noqa: C901, EM102, FBT001, PLR0912, PLR0915, TRY003
"""Independently verify saved activation-study states without re-solving."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
CASES = {
    "baseline-smooth": ("raw6", True),
    "learned-axis": ("learned-axis", False),
    "learned-axis-smooth": ("learned-axis", True),
}
SMOOTH_LENGTH_M = 0.005
PSD_ROUNDOFF_FACTOR = 64.0
FLOAT_RTOL = 2.0e-10
FLOAT_ATOL = 2.0e-12


class Config(cherries.BaseConfig):
    """Final saved-state locations for the approved three-run study."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    learned_axis_dir: Path = cherries.input("24-learned-axis-256")
    learned_axis_smooth_dir: Path = cherries.input("25-learned-axis-smooth-256")
    raw6_smooth_dir: Path = cherries.input("28-raw6-smooth")
    raw6_off_dir: Path = (
        GROUP.parents[2] / "09/08/local-skin-prestrain/data/30-refit-no-skin"
    )
    learned_axis_settings: Path = cherries.input(
        "18-learned-axis-calibration-refined/summary.json"
    )
    raw6_settings: Path = cherries.input("19-raw6-smooth-calibration/summary.json")
    output_dir: Path = cherries.output("50-verification", mkdir=True)


@dataclass(frozen=True)
class Fixture:
    path: Path
    points: np.ndarray
    tets: np.ndarray
    active_ids: np.ndarray
    active_region: np.ndarray
    top: np.ndarray
    target: np.ndarray
    skin_ids: np.ndarray
    active_volume: np.ndarray
    rest_determinant: np.ndarray
    graph_i: np.ndarray
    graph_j: np.ndarray
    graph_weight: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def close(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL
    )


def require_close(left: float, right: float, message: str) -> None:
    require(close(left, right), f"{message}: {left!r} != {right!r}")


def require_array_close(
    left: np.ndarray, right: np.ndarray, message: str, *, atol: float = 1.0e-11
) -> None:
    require(
        left.shape == right.shape, f"{message}: shape {left.shape} != {right.shape}"
    )
    require(
        bool(np.allclose(left, right, rtol=2.0e-11, atol=atol)),
        f"{message}: maximum absolute difference {np.max(np.abs(left - right))}",
    )


def parse_cell(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    return float(value)


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = [
            {key: parse_cell(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    require(bool(rows), f"empty trace: {path}")
    required = {
        "step",
        "objective_mm2",
        "objective_total",
        "smoothness",
        "smoothness_C",
        "smoothness_Z",
        "smoothness_weight",
        "fit_rms_mm",
        "motion_rms_mm",
        "target_projection",
        "gradient_rms",
        "forward_success",
        "adjoint_success",
        "projection_rms",
        "projected_negative_eigenvalue_fraction",
        "best_step",
        "best_objective_step",
        "detF_min",
        "detF_max",
        "inverted_all_cells",
        "inverted_active_cells",
    }
    missing = required - rows[0].keys()
    require(not missing, f"trace {path} lacks columns {sorted(missing)}")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                require(math.isfinite(value), f"nonfinite trace value {key} at {path}")
    return rows


def compare_mapping_subset(
    expected: dict[str, Any], actual: dict[str, Any], message: str
) -> None:
    for key, value in expected.items():
        require(key in actual, f"{message}: missing key {key}")
        other = actual[key]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            require(value == other, f"{message}.{key}: {value!r} != {other!r}")
        elif isinstance(value, (int, float)):
            require_close(value, other, f"{message}.{key}")
        elif isinstance(value, dict):
            require(isinstance(other, dict), f"{message}.{key} is not a mapping")
            compare_mapping_subset(value, other, f"{message}.{key}")
        else:
            require(value == other, f"{message}.{key} differs")


def case_directories(cfg: Config) -> dict[str, Path]:
    return {
        "baseline-smooth": cfg.raw6_smooth_dir,
        "learned-axis": cfg.learned_axis_dir,
        "learned-axis-smooth": cfg.learned_axis_smooth_dir,
    }


def preflight_primary_evidence(directories: dict[str, Path]) -> None:
    common_required = {
        "config.json",
        "provenance.json",
        "summary.json",
    }
    accepted_prefix_required = {
        "trace.csv",
        "solver-receipts.jsonl",
        "optimizer-latest.pt",
        "best.npz",
        "best-objective.npz",
    }
    problems = []
    for case, directory in directories.items():
        missing = sorted(
            name for name in common_required if not (directory / name).is_file()
        )
        if missing:
            problems.append(f"{case}: missing {missing}")
            continue
        summary = json.loads((directory / "summary.json").read_text())
        status = summary.get("status")
        accepted_prefix_available = summary.get("last_evaluated_step") is not None
        if accepted_prefix_available:
            missing = sorted(
                name
                for name in accepted_prefix_required
                if not (directory / name).is_file()
            )
            if missing:
                problems.append(f"{case}: accepted prefix missing {missing}")
        else:
            problems.append(
                f"{case}: no accepted solver-valid prefix; this verifier requires "
                "at least accepted step 0"
            )
        if status == "completed_fixed_budget_not_stationarity_certified":
            if summary.get("failure") is not None:
                problems.append(f"{case}: completed status has a failure receipt")
            if not (directory / "last.npz").is_file():
                problems.append(f"{case}: completed run lacks last.npz")
            unexpected = [
                name
                for name in ("failure.json", "failure-controls.npz")
                if (directory / name).exists()
            ]
            if unexpected:
                problems.append(
                    f"{case}: completed run has failure artifacts {unexpected}"
                )
        elif status == "failed_before_completion":
            if not isinstance(summary.get("failure"), dict):
                problems.append(f"{case}: failed status lacks a failure receipt")
            missing_failure = [
                name
                for name in ("failure.json", "failure-controls.npz")
                if not (directory / name).is_file()
            ]
            if missing_failure:
                problems.append(
                    f"{case}: failed run lacks failure artifacts {missing_failure}"
                )
        else:
            problems.append(f"{case}: unsupported status {status!r}")
    require(
        not problems,
        "primary evidence preflight failed: " + "; ".join(problems),
    )


def build_graph(
    points: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    region: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = tets[active_ids]
    pattern = np.array(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    faces = np.sort(active[:, pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(active_ids)), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    pair = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    require(
        not np.any(np.diff(pair) == 1),
        "fixture has a nonmanifold active tetrahedral face",
    )
    i, j = owner[pair], owner[pair + 1]
    same = region[i] == region[j]
    i, j, faces = i[same], j[same], faces[pair[same]]
    xyz = points[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    centers = points[active].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    require(bool(np.all(distance > 0.0)), "face-sharing centroids coincide")
    active_fraction = fraction[active_ids]
    weight = (
        area
        / distance
        * (
            2.0
            * active_fraction[i]
            * active_fraction[j]
            / (active_fraction[i] + active_fraction[j])
        )
    )
    return i, j, weight


def load_fixture(path: Path) -> Fixture:
    volume = pv.read(path / "volume.vtu")
    skin = pv.read(path / "skin.vtp")
    points = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    active_ids = np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], bool))
    region = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        active_ids
    ]
    fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)
    dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], axes=(0, 2, 1))
    rest_determinant = np.linalg.det(dm)
    require(bool(np.all(rest_determinant > 0.0)), "fixture has nonpositive rest tets")
    active_volume = rest_determinant[active_ids] / 6.0 * fraction[active_ids]
    stored_volume = np.asarray(volume.cell_data["Volume"], dtype=np.float64)
    require_array_close(
        rest_determinant / 6.0,
        stored_volume,
        "fixture geometry and stored Volume",
        atol=1.0e-18,
    )
    top = np.flatnonzero(
        np.asarray(volume.point_data["IsFace"], bool)
        & np.isfinite(np.asarray(volume.point_data["Smile"])).all(axis=1)
    )
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    graph_i, graph_j, graph_weight = build_graph(
        points, tets, active_ids, region, fraction
    )
    return Fixture(
        path=path,
        points=points,
        tets=tets,
        active_ids=active_ids,
        active_region=region,
        top=top,
        target=target,
        skin_ids=skin_ids,
        active_volume=active_volume,
        rest_determinant=rest_determinant,
        graph_i=graph_i,
        graph_j=graph_j,
        graph_weight=graph_weight,
    )


def q_to_c_z(q: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Independently map optimized coordinates to C and physical Z."""
    require(q.ndim == 2, f"invalid q rank {q.shape}")
    if model == "learned-axis":
        require(q.shape[1] == 3, f"invalid learned-axis q shape {q.shape}")
        c = q[:, :, None] * q[:, None, :]
        strength = np.sum(q * q, axis=1)[:, None, None]
        return c, (2.0 + strength) * c

    require(model == "raw6", f"unknown model {model!r}")
    require(q.shape[1] == 6, f"invalid Raw6 q shape {q.shape}")
    c = np.empty((len(q), 3, 3), dtype=np.float64)
    c[:, 0, 0] = q[:, 0]
    c[:, 1, 1] = q[:, 1]
    c[:, 2, 2] = q[:, 2]
    c[:, 0, 1] = c[:, 1, 0] = q[:, 3]
    c[:, 1, 2] = c[:, 2, 1] = q[:, 4]
    c[:, 0, 2] = c[:, 2, 0] = q[:, 5]
    b = c + np.eye(3)
    return c, b @ np.swapaxes(b, 1, 2) - np.eye(3)


def recompute_smoothness(z: np.ndarray, fixture: Fixture) -> float:
    delta = z[fixture.graph_i] - z[fixture.graph_j]
    weighted = np.sum(fixture.graph_weight * np.sum(delta * delta, axis=(1, 2)))
    return float(SMOOTH_LENGTH_M**2 * weighted / fixture.active_volume.sum())


def spectrum_metrics(z: np.ndarray) -> dict[str, Any]:
    """Classify PSD violations relative to each matrix's spectral scale."""
    spectrum = np.linalg.eigvalsh(z)
    scale = np.maximum(1.0, np.max(np.abs(spectrum), axis=1))
    bound = PSD_ROUNDOFF_FACTOR * np.finfo(np.float64).eps * scale
    violation = spectrum < -bound[:, None]
    relative_negative_residual = np.maximum(0.0, -spectrum[:, 0]) / scale
    return {
        "spectrum": spectrum,
        "violation": violation,
        "z_eigen_min": float(spectrum.min()),
        "z_eigen_max": float(spectrum.max()),
        "z_psd_roundoff_factor": PSD_ROUNDOFF_FACTOR,
        "z_psd_roundoff_bound_max": float(bound.max()),
        "z_psd_relative_negative_residual_max": float(relative_negative_residual.max()),
        "z_psd_violation_eigenvalue_fraction": float(np.mean(violation)),
        "z_psd_violation_cell_fraction": float(np.mean(np.any(violation, axis=1))),
    }


def state_geometry_metrics(u: np.ndarray, fixture: Fixture) -> dict[str, float]:
    predicted = u[fixture.top]
    target = fixture.target[fixture.top]
    error = predicted - target
    objective = float(np.mean(error * error) * 1.0e6)
    return {
        "objective_mm2": objective,
        "fit_rms_mm": float(1000.0 * np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "motion_rms_mm": float(
            1000.0 * np.sqrt(np.mean(np.sum(predicted * predicted, axis=1)))
        ),
        "target_projection": float(
            np.sum(predicted * target) / np.sum(target * target)
        ),
    }


def state_volume_metrics(u: np.ndarray, fixture: Fixture) -> dict[str, float | int]:
    x = fixture.points + u
    ds = np.transpose(x[fixture.tets[:, 1:]] - x[fixture.tets[:, :1]], axes=(0, 2, 1))
    determinant = np.linalg.det(ds) / fixture.rest_determinant
    active = determinant[fixture.active_ids]
    return {
        "detF_min": float(determinant.min()),
        "detF_max": float(determinant.max()),
        "inverted_all_cells": int(np.count_nonzero(determinant <= 0.0)),
        "inverted_active_cells": int(np.count_nonzero(active <= 0.0)),
        "active_volume_weighted_rms_detF_minus_1": float(
            np.sqrt(np.average((active - 1.0) ** 2, weights=fixture.active_volume))
        ),
    }


def validate_state(
    path: Path,
    model: str,
    fixture: Fixture,
    row: dict[str, Any],
    *,
    check_volume: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_keys = {
        "step",
        "q",
        "C",
        "Z",
        "u",
        "active_ids",
        "rest_points",
        "solver_valid",
        "model",
    }
    with np.load(path, allow_pickle=False) as loaded:
        require(set(loaded.files) == expected_keys, f"state schema differs: {path}")
        state = {key: np.asarray(loaded[key]) for key in loaded.files}
    step = int(state["step"].item())
    require(step == int(row["step"]), f"state step differs from trace: {path}")
    require(bool(state["solver_valid"].item()), f"state is solver-invalid: {path}")
    require(state["model"].item() == model, f"state model differs: {path}")
    require(
        np.array_equal(state["active_ids"], fixture.active_ids),
        f"state active IDs differ: {path}",
    )
    require(
        np.array_equal(state["rest_points"], fixture.points),
        f"state rest points differ: {path}",
    )
    q = np.asarray(state["q"], dtype=np.float64)
    c = np.asarray(state["C"], dtype=np.float64)
    z = np.asarray(state["Z"], dtype=np.float64)
    u = np.asarray(state["u"], dtype=np.float64)
    expected_coordinates = 3 if model == "learned-axis" else 6
    require(
        q.shape == (len(fixture.active_ids), expected_coordinates),
        f"q shape differs: {path}",
    )
    require(c.shape == (len(fixture.active_ids), 3, 3), f"C shape differs: {path}")
    require(z.shape == (len(fixture.active_ids), 3, 3), f"Z shape differs: {path}")
    require(u.shape == fixture.points.shape, f"u shape differs: {path}")
    require(
        bool(
            np.isfinite(q).all()
            and np.isfinite(c).all()
            and np.isfinite(z).all()
            and np.isfinite(u).all()
        ),
        f"state contains nonfinite values: {path}",
    )
    expected_c, expected_z = q_to_c_z(q, model)
    require_array_close(c, expected_c, f"independent q-to-C conversion at {path}")
    require_array_close(z, expected_z, f"independent q-to-Z conversion at {path}")
    spectrum = spectrum_metrics(z)
    if model == "learned-axis":
        require(
            not np.any(spectrum["violation"]),
            "learned-axis Z violates the scale-aware PSD roundoff bound "
            f"at {path}: minimum eigenvalue {spectrum['z_eigen_min']}, "
            "maximum relative negative residual "
            f"{spectrum['z_psd_relative_negative_residual_max']}",
        )
    smooth_c = recompute_smoothness(c, fixture)
    smooth_z = recompute_smoothness(z, fixture)
    require_close(smooth_c, row["smoothness_C"], f"saved-state C smoothness at {path}")
    require_close(smooth_z, row["smoothness_Z"], f"saved-state Z smoothness at {path}")
    require_close(smooth_c, row["smoothness"], f"optimized smoothness at {path}")
    geometry = state_geometry_metrics(u, fixture)
    for key, value in geometry.items():
        require_close(value, row[key], f"saved-state {key} at {path}")
    if check_volume:
        volume = state_volume_metrics(u, fixture)
        for key, value in volume.items():
            require_close(value, row[key], f"saved-state {key} at {path}")
    receipt = {
        **record(path),
        "step": step,
        "smoothness_C": smooth_c,
        "smoothness_Z": smooth_z,
        **{
            key: value
            for key, value in spectrum.items()
            if key not in {"spectrum", "violation"}
        },
        "volume_metrics_recomputed": check_volume,
    }
    return receipt, {"q": q, "C": c, "Z": z, "u": u}


def validate_surface(
    path: Path, step: int, fixture: Fixture, full_u: np.ndarray | None
) -> str:
    with np.load(path, allow_pickle=False) as loaded:
        require(
            set(loaded.files) == {"step", "point_ids", "u"},
            f"surface schema differs: {path}",
        )
        saved_step = int(loaded["step"].item())
        point_ids = np.asarray(loaded["point_ids"])
        u = np.asarray(loaded["u"], dtype=np.float64)
    require(saved_step == step, f"surface step differs: {path}")
    require(np.array_equal(point_ids, fixture.skin_ids), f"surface IDs differ: {path}")
    require(u.shape == (len(fixture.skin_ids), 3), f"surface shape differs: {path}")
    require(bool(np.isfinite(u).all()), f"surface has nonfinite values: {path}")
    if full_u is not None:
        require(
            np.array_equal(u, full_u[fixture.skin_ids]),
            f"surface and full checkpoint differ: {path}",
        )
    return sha256(path)


def validate_failure_evidence(
    directory: Path,
    summary_failure: dict[str, Any],
    model: str,
    fixture: Fixture,
    last_accepted_step: int,
    target_step: int,
) -> dict[str, Any]:
    failure_path = directory / "failure.json"
    controls_path = directory / "failure-controls.npz"
    failure = json.loads(failure_path.read_text())
    require(failure == summary_failure, f"summary/failure receipt differs: {directory}")
    require(
        set(failure) == {"type", "message", "step", "forward"},
        f"failure receipt schema differs: {directory}",
    )
    failed_step = int(failure["step"])
    forward = failure["forward"]
    failed_forward_evaluation = (
        isinstance(forward, dict) and forward.get("success") is False
    )
    require(
        failed_step in {last_accepted_step, last_accepted_step + 1},
        f"failure step is not adjacent to accepted prefix: {directory}",
    )
    if failed_forward_evaluation:
        require(
            failed_step == last_accepted_step + 1,
            f"failed forward is not the next evaluation after the accepted prefix: {directory}",
        )
    require(
        failed_step <= target_step, f"failure step exceeds declared target: {directory}"
    )

    expected_keys = {"step", "q", "C", "Z", "active_ids", "solver_valid"}
    with np.load(controls_path, allow_pickle=False) as loaded:
        require(
            set(loaded.files) == expected_keys,
            f"failure controls schema differs: {controls_path}",
        )
        controls = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(
        int(controls["step"].item()) == failed_step,
        f"failure controls step differs: {controls_path}",
    )
    require(
        not bool(controls["solver_valid"].item()),
        f"failure controls claim solver validity: {controls_path}",
    )
    require(
        np.array_equal(controls["active_ids"], fixture.active_ids),
        f"failure controls active IDs differ: {controls_path}",
    )
    coordinates = 3 if model == "learned-axis" else 6
    q = np.asarray(controls["q"], dtype=np.float64)
    c = np.asarray(controls["C"], dtype=np.float64)
    z = np.asarray(controls["Z"], dtype=np.float64)
    require(
        q.shape == (len(fixture.active_ids), coordinates),
        f"failure q shape differs: {controls_path}",
    )
    require(
        c.shape == (len(fixture.active_ids), 3, 3),
        f"failure C shape differs: {controls_path}",
    )
    require(
        z.shape == (len(fixture.active_ids), 3, 3),
        f"failure Z shape differs: {controls_path}",
    )
    expected_c, expected_z = q_to_c_z(q, model)
    require(
        bool(np.allclose(c, expected_c, rtol=2.0e-11, atol=1.0e-11, equal_nan=True)),
        f"failure q-to-C conversion differs: {controls_path}",
    )
    require(
        bool(np.allclose(z, expected_z, rtol=2.0e-11, atol=1.0e-11, equal_nan=True)),
        f"failure q-to-Z conversion differs: {controls_path}",
    )
    finite = {
        "q": bool(np.isfinite(q).all()),
        "C": bool(np.isfinite(c).all()),
        "Z": bool(np.isfinite(z).all()),
    }
    spectrum = spectrum_metrics(z) if finite["Z"] else None
    if model == "learned-axis" and spectrum is not None:
        require(
            not np.any(spectrum["violation"]),
            f"failed learned-axis controls violate scale-aware PSD bound: {controls_path}",
        )
    return {
        "failure": record(failure_path),
        "controls": record(controls_path),
        "failed_step": failed_step,
        "last_accepted_step": last_accepted_step,
        "step_relation": (
            "same_as_last_accepted"
            if failed_step == last_accepted_step
            else "next_attempted_step"
        ),
        "failure_stage": (
            "forward_evaluation"
            if failed_forward_evaluation
            else "not_recorded_by_legacy_failure_schema"
        ),
        "failure_stage_is_explicitly_supported_by_receipt": failed_forward_evaluation,
        "solver_valid": False,
        "finite": finite,
        "control_mapping_recomputed": True,
        "mapping_check_allows_matching_nonfinite_values": True,
        "spectrum": (
            None
            if spectrum is None
            else {
                key: value
                for key, value in spectrum.items()
                if key not in {"spectrum", "violation"}
            }
        ),
        "separate_from_last_accepted_optimizer": True,
    }


def resolve_snapshot(case_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.is_file():
        return path
    parts = path.parts
    if "sources" in parts:
        index = parts.index("sources")
        candidate = case_dir.joinpath(*parts[index:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"archived source snapshot is missing: {recorded}")


def verify_sources(
    case_dir: Path,
    provenance: dict[str, Any],
    required_modules: set[str] | None = None,
) -> dict[str, Any]:
    if required_modules is None:
        required_modules = {
            "__main__",
            "activation_controls",
            "experiment_profile",
            "study_metrics",
            "study_physics",
            "study_runner",
            "tensor_active",
            "volume_preserving_active",
        }
    sources = provenance["sources"]
    require(
        required_modules <= sources.keys(), f"source archive incomplete: {case_dir}"
    )
    snapshots = {}
    drift = {}
    for module, source in sources.items():
        snapshot = resolve_snapshot(case_dir, source["snapshot"])
        snapshot_sha = sha256(snapshot)
        require(
            snapshot_sha == source["sha256"],
            f"snapshot hash differs for {module}: {snapshot}",
        )
        snapshots[module] = snapshot_sha
        current = Path(source["path"])
        current_sha = sha256(current) if current.is_file() else None
        drift[module] = {
            "recorded_path": source["path"],
            "exists": current_sha is not None,
            "current_sha256": current_sha,
            "differs_from_authoritative_snapshot": current_sha != snapshot_sha,
        }
    return {"snapshot_sha256_by_module": snapshots, "current_source": drift}


def validate_trace(
    case: str,
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    expected_weight: float,
    *,
    completed: bool,
) -> dict[str, Any]:
    start = int(provenance["optimizer"]["start_step"])
    target = int(provenance["optimizer"]["target_step"])
    steps = [int(row["step"]) for row in rows]
    last = steps[-1]
    require(steps == list(range(last + 1)), f"accepted trace prefix differs: {case}")
    require(last <= target, f"accepted trace exceeds target: {case}")
    require(
        completed == (last == target),
        f"run status and declared-target completion differ: {case}",
    )
    best_fit_step = 0
    best_fit = math.inf
    best_objective_step = 0
    best_objective = math.inf
    max_inverted = 0
    for row in rows:
        step = int(row["step"])
        require(
            row["forward_success"] is True, f"failed forward trace row: {case}/{step}"
        )
        require(
            row["adjoint_success"] is True, f"failed adjoint trace row: {case}/{step}"
        )
        require_close(
            row["smoothness_weight"], expected_weight, f"trace weight {case}/{step}"
        )
        require_close(
            row["objective_total"],
            row["objective_mm2"] + expected_weight * row["smoothness"],
            f"total objective {case}/{step}",
        )
        require_close(
            3.0 * row["objective_mm2"],
            row["fit_rms_mm"] ** 2,
            f"fit/objective identity {case}/{step}",
        )
        require_close(
            row["smoothness"], row["smoothness_C"], f"C penalty {case}/{step}"
        )
        require(row["projection_rms"] >= 0.0, f"negative projection RMS: {case}/{step}")
        require(
            0.0 <= row["projected_negative_eigenvalue_fraction"] <= 1.0,
            f"projection fraction outside [0,1]: {case}/{step}",
        )
        require(row["projection_rms"] == 0.0, f"control was projected: {case}/{step}")
        require(
            row["projected_negative_eigenvalue_fraction"] == 0.0,
            f"control has a projection fraction: {case}/{step}",
        )
        if row["objective_mm2"] < best_fit:
            best_fit, best_fit_step = row["objective_mm2"], step
        if row["objective_total"] < best_objective:
            best_objective, best_objective_step = row["objective_total"], step
        require(
            int(row["best_step"]) == best_fit_step,
            f"running best fit differs: {case}/{step}",
        )
        require(
            int(row["best_objective_step"]) == best_objective_step,
            f"running best objective differs: {case}/{step}",
        )
        max_inverted = max(max_inverted, int(row["inverted_all_cells"]))
    return {
        "start_step": start,
        "target_step": target,
        "last_accepted_step": last,
        "fixed_budget_completed": completed,
        "continuation_prefix_states": start + 1,
        "evaluated_states": len(rows),
        "best_step": best_fit_step,
        "best_objective_step": best_objective_step,
        "maximum_inverted_all_cells_diagnostic": max_inverted,
    }


def validate_solver_receipts(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    receipts = [json.loads(line) for line in path.read_text().splitlines() if line]
    require(len(receipts) == len(rows), f"solver receipt count differs: {path}")
    for receipt, row in zip(receipts, rows, strict=True):
        step = int(row["step"])
        require(int(receipt["step"]) == step, f"solver receipt step differs: {path}")
        require(
            receipt["forward"]["success"] is True,
            f"forward receipt failed: {path}/{step}",
        )
        require(
            receipt["adjoint"]["success"] is True,
            f"adjoint receipt failed: {path}/{step}",
        )
    return {
        **record(path),
        "count": len(receipts),
        "scope": "accepted_trace_prefix",
        "all_accepted_states_successful": True,
    }


def optimizer_step_values(state: dict[str, Any]) -> list[int]:
    values = []
    for item in state.values():
        step = item["step"]
        if isinstance(step, torch.Tensor):
            step = step.item()
        values.append(int(step))
    return values


def validate_parent_checkpoint(
    config: dict[str, Any],
    provenance: dict[str, Any],
    case: str,
    model: str,
    fixture: Fixture,
    settings_sha: str,
    expected_weight: float,
    start_step: int,
    expected_gradient_rms: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parent = provenance["parent"]
    if parent is None:
        require(config["resume"] is None, f"fresh config has resume path: {case}")
        return None, None

    require(config["resume"] is not None, f"continuation lacks resume path: {case}")
    require(
        set(parent) == {"checkpoint", "sha256", "step", "continuation"},
        f"parent receipt schema differs: {case}",
    )
    require(
        parent["continuation"]
        == "Restore saved gradient, controls, Adam state and solve seed; next evaluation follows one Adam update. Parent states are not re-evaluated.",
        f"continuation semantics differ: {case}",
    )
    resume = Path(config["resume"])
    parent_path = Path(parent["checkpoint"])
    require(
        str(config["resume"]) == str(parent["checkpoint"]),
        f"resume and parent checkpoint declarations differ: {case}",
    )
    require(
        resume.resolve() == parent_path.resolve(),
        f"resume and parent checkpoint paths differ: {case}",
    )
    require(parent_path.is_file(), f"parent checkpoint is missing: {parent_path}")
    require(
        sha256(parent_path) == parent["sha256"],
        f"parent checkpoint hash differs: {case}",
    )
    require(int(parent["step"]) == start_step, f"parent step differs: {case}")

    saved = torch.load(parent_path, map_location="cpu", weights_only=False)
    expected_keys = {
        "case",
        "model",
        "step",
        "q",
        "u",
        "active_ids",
        "gradient",
        "optimizer",
        "smoothness_weight",
        "settings_sha256",
        "initialization_seed",
        "best_fit_step",
        "best_objective_step",
    }
    require(set(saved) == expected_keys, f"parent checkpoint schema differs: {case}")
    require(
        saved["case"] == case and saved["model"] == model,
        f"parent checkpoint identity differs: {case}",
    )
    require(int(saved["step"]) == start_step, f"parent checkpoint step differs: {case}")
    require(
        saved["settings_sha256"] == settings_sha,
        f"parent checkpoint settings hash differs: {case}",
    )
    expected_initialization_seed = provenance["optimizer"]["initialization_seed"]
    require(
        saved["initialization_seed"] == expected_initialization_seed,
        f"parent checkpoint initialization seed differs: {case}",
    )
    require_close(
        saved["smoothness_weight"],
        expected_weight,
        f"parent checkpoint weight {case}",
    )
    active_ids = np.asarray(saved["active_ids"])
    require(
        np.array_equal(active_ids, fixture.active_ids),
        f"parent checkpoint active IDs differ: {case}",
    )
    q = saved["q"].detach().cpu().numpy()
    gradient = saved["gradient"].detach().cpu().numpy()
    u = np.asarray(saved["u"])
    expected_coordinates = 3 if model == "learned-axis" else 6
    require(
        q.shape == (len(fixture.active_ids), expected_coordinates),
        f"parent q shape differs: {case}",
    )
    require(gradient.shape == q.shape, f"parent gradient shape differs: {case}")
    require(
        u.shape == fixture.points.shape, f"parent displacement shape differs: {case}"
    )
    require(
        bool(
            np.isfinite(q).all()
            and np.isfinite(gradient).all()
            and np.isfinite(u).all()
        ),
        f"parent checkpoint contains nonfinite values: {case}",
    )
    require_close(
        float(np.sqrt(np.mean(gradient * gradient))),
        expected_gradient_rms,
        f"parent checkpoint gradient RMS {case}",
    )
    optimizer = saved["optimizer"]
    require(
        len(optimizer["param_groups"]) == 1,
        f"parent optimizer group count differs: {case}",
    )
    group = optimizer["param_groups"][0]
    require(group["params"] == [0], f"parent optimizer parameter list differs: {case}")
    expected = provenance["optimizer"]
    require_close(group["lr"], expected["lr"], f"parent optimizer lr {case}")
    require_close(group["eps"], expected["eps"], f"parent optimizer eps {case}")
    require(
        tuple(group["betas"]) == tuple(expected["betas"]),
        f"parent optimizer betas differ: {case}",
    )
    for name in ("weight_decay", "amsgrad", "maximize", "foreach", "fused"):
        require(
            group[name] == expected[name], f"parent optimizer {name} differs: {case}"
        )
    expected_state_count = int(start_step > 0)
    require(
        len(optimizer["state"]) == expected_state_count,
        f"parent optimizer state count differs: {case}",
    )
    step_values = optimizer_step_values(optimizer["state"])
    expected_steps = [] if start_step == 0 else [start_step]
    require(step_values == expected_steps, f"parent Adam step count differs: {case}")
    for item in optimizer["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = item[name].detach().cpu().numpy()
            require(
                value.shape == q.shape, f"parent optimizer {name} shape differs: {case}"
            )
            require(
                bool(np.isfinite(value).all()),
                f"parent optimizer {name} is nonfinite: {case}",
            )
    return (
        {
            **record(parent_path),
            "declared_path": str(parent["checkpoint"]),
            "step": start_step,
            "adam_step": None if not step_values else step_values[0],
        },
        {"q": q, "u": u, "gradient": gradient, "saved": saved},
    )


def validate_optimizer(
    path: Path,
    case: str,
    model: str,
    row: dict[str, Any],
    provenance: dict[str, Any],
    settings_sha: str,
    expected_weight: float,
    fixture: Fixture,
    best_step: int,
    best_objective_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "case",
        "model",
        "step",
        "q",
        "u",
        "active_ids",
        "gradient",
        "optimizer",
        "smoothness_weight",
        "settings_sha256",
        "initialization_seed",
        "best_fit_step",
        "best_objective_step",
    }
    require(set(saved) == expected_keys, f"optimizer checkpoint schema differs: {path}")
    require(
        saved["case"] == case and saved["model"] == model,
        f"optimizer identity differs: {path}",
    )
    step = int(saved["step"])
    require(step == int(row["step"]), f"optimizer evaluated step differs: {path}")
    require(
        saved["settings_sha256"] == settings_sha,
        f"optimizer settings hash differs: {path}",
    )
    require(
        saved["initialization_seed"] == provenance["optimizer"]["initialization_seed"],
        f"optimizer initialization seed differs: {path}",
    )
    require_close(
        saved["smoothness_weight"], expected_weight, f"optimizer weight at {path}"
    )
    require(
        int(saved["best_fit_step"]) == best_step,
        f"optimizer best-fit step differs: {path}",
    )
    require(
        int(saved["best_objective_step"]) == best_objective_step,
        f"optimizer best-objective step differs: {path}",
    )
    q = saved["q"].detach().cpu().numpy()
    gradient = saved["gradient"].detach().cpu().numpy()
    u = np.asarray(saved["u"])
    active_ids = np.asarray(saved["active_ids"])
    require(
        np.array_equal(active_ids, fixture.active_ids),
        f"optimizer active IDs differ: {path}",
    )
    coordinates = 3 if model == "learned-axis" else 6
    require(
        q.shape == (len(fixture.active_ids), coordinates),
        f"optimizer q shape differs: {path}",
    )
    require(u.shape == fixture.points.shape, f"optimizer u shape differs: {path}")
    require(
        bool(np.isfinite(q).all() and np.isfinite(u).all()),
        f"optimizer accepted state contains nonfinite values: {path}",
    )
    c, z = q_to_c_z(q, model)
    smooth_c = recompute_smoothness(c, fixture)
    smooth_z = recompute_smoothness(z, fixture)
    require_close(smooth_c, row["smoothness_C"], f"optimizer C smoothness at {path}")
    require_close(smooth_z, row["smoothness_Z"], f"optimizer Z smoothness at {path}")
    for key, value in state_geometry_metrics(u, fixture).items():
        require_close(value, row[key], f"optimizer saved-state {key} at {path}")
    for key, value in state_volume_metrics(u, fixture).items():
        require_close(value, row[key], f"optimizer saved-state {key} at {path}")
    spectrum = spectrum_metrics(z)
    if model == "learned-axis":
        require(
            not np.any(spectrum["violation"]),
            f"optimizer learned-axis Z violates scale-aware PSD bound: {path}",
        )
    require(gradient.shape == q.shape, f"optimizer gradient shape differs: {path}")
    require(
        bool(np.isfinite(gradient).all()), f"optimizer gradient is nonfinite: {path}"
    )
    require_close(
        float(np.sqrt(np.mean(gradient * gradient))),
        row["gradient_rms"],
        f"optimizer gradient RMS at {path}",
    )
    optimizer = saved["optimizer"]
    require(
        len(optimizer["param_groups"]) == 1, f"optimizer group count differs: {path}"
    )
    group = optimizer["param_groups"][0]
    require(group["params"] == [0], f"optimizer parameter list differs: {path}")
    expected = provenance["optimizer"]
    require_close(group["lr"], expected["lr"], f"optimizer lr at {path}")
    require_close(group["eps"], expected["eps"], f"optimizer eps at {path}")
    require(
        tuple(group["betas"]) == tuple(expected["betas"]),
        f"optimizer betas differ: {path}",
    )
    for name in ("weight_decay", "amsgrad", "maximize", "foreach", "fused"):
        require(group[name] == expected[name], f"optimizer {name} differs: {path}")
    expected_state_count = int(step > 0)
    require(
        len(optimizer["state"]) == expected_state_count,
        f"optimizer state count differs: {path}",
    )
    step_values = optimizer_step_values(optimizer["state"])
    expected_steps = [] if step == 0 else [step]
    require(
        step_values == expected_steps, f"Adam step count differs: {path}: {step_values}"
    )
    for item in optimizer["state"].values():
        for name in ("exp_avg", "exp_avg_sq"):
            value = item[name].detach().cpu().numpy()
            require(value.shape == q.shape, f"optimizer {name} shape differs: {path}")
            require(
                bool(np.isfinite(value).all()), f"optimizer {name} is nonfinite: {path}"
            )
    return (
        {
            **record(path),
            "artifact_role": "last_accepted_optimizer_checkpoint",
            "evaluated_step": step,
            "adam_step": None if not step_values else step_values[0],
            "gradient_rms": float(np.sqrt(np.mean(gradient * gradient))),
            "smoothness_C": smooth_c,
            "smoothness_Z": smooth_z,
            **{
                key: value
                for key, value in spectrum.items()
                if key not in {"spectrum", "violation"}
            },
            "matches_last_accepted_trace_and_surface": True,
        },
        {"q": q, "C": c, "Z": z, "u": u},
    )


def validate_copied_parent_prefix(
    directory: Path,
    parent_checkpoint: Path,
    rows: list[dict[str, Any]],
    start_step: int,
    parent_state: dict[str, Any],
) -> dict[str, Any]:
    parent_dir = parent_checkpoint.parent
    parent_summary = json.loads((parent_dir / "summary.json").read_text())
    require(
        int(parent_summary["last_evaluated_step"]) == start_step,
        "parent summary boundary differs",
    )
    parent_rows = read_trace(parent_dir / "trace.csv")
    require(len(parent_rows) == start_step + 1, "parent trace length differs")
    for expected, actual in zip(parent_rows, rows[: start_step + 1], strict=True):
        require(expected.keys() == actual.keys(), "copied parent trace columns differ")
        compare_mapping_subset(expected, actual, "copied parent trace row")

    copied = {}
    for pattern in ("surface-*.npz", "step-*.npz"):
        for source in parent_dir.glob(pattern):
            target = directory / source.name
            require(target.is_file(), f"copied parent artifact is missing: {target}")
            require(
                sha256(target) == sha256(source),
                f"copied parent artifact differs: {target}",
            )
            copied[source.name] = sha256(target)

    parent_receipts = [
        json.loads(line)
        for line in (parent_dir / "solver-receipts.jsonl").read_text().splitlines()
        if line
    ]
    child_receipts = [
        json.loads(line)
        for line in (directory / "solver-receipts.jsonl").read_text().splitlines()
        if line
    ]
    require(
        child_receipts[: len(parent_receipts)] == parent_receipts,
        "copied parent solver-receipt prefix differs",
    )
    boundary_path = directory / f"step-{start_step:04d}.npz"
    with np.load(boundary_path, allow_pickle=False) as boundary:
        require(
            np.array_equal(boundary["q"], parent_state["q"]),
            "copied boundary q differs from resume checkpoint",
        )
        require(
            np.array_equal(boundary["u"], parent_state["u"]),
            "copied boundary u differs from resume checkpoint",
        )
    first_new_step = None
    if len(rows) > start_step + 1:
        first_new_step = int(rows[start_step + 1]["step"])
        require(
            first_new_step == start_step + 1,
            "first continuation evaluation is not parent step plus one",
        )
    return {
        "directory": str(parent_dir.resolve()),
        "summary": record(parent_dir / "summary.json"),
        "trace": record(parent_dir / "trace.csv"),
        "copied_artifact_count": len(copied),
        "copied_artifact_manifest_sha256": hashlib.sha256(
            "\n".join(
                f"{name}\0{digest}" for name, digest in sorted(copied.items())
            ).encode()
        ).hexdigest(),
        "solver_receipt_prefix_count": len(parent_receipts),
        "first_new_step": first_new_step,
    }


def expected_learned_axis_initial(
    fixture: Fixture, settings: dict[str, Any]
) -> np.ndarray:
    labels = torch.as_tensor(fixture.active_region, dtype=torch.int64, device="cpu")
    unique = torch.unique(labels, sorted=True)
    require(
        torch.equal(
            unique,
            torch.arange(len(unique), dtype=torch.int64, device="cpu"),
        ),
        "fixture activation labels are not contiguous",
    )
    generator = torch.Generator(device="cpu").manual_seed(
        int(settings["initialization_seed"])
    )
    axes = torch.randn(
        (len(unique), 3),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    return (math.sqrt(float(settings["initial_strength"])) * axes[labels]).numpy()


def validate_initialization(
    model: str,
    settings: dict[str, Any],
    fixture: Fixture,
    initial: dict[str, np.ndarray],
    row: dict[str, Any],
) -> dict[str, Any]:
    if model == "learned-axis":
        expected = expected_learned_axis_initial(fixture, settings)
        require(
            np.array_equal(initial["q"], expected),
            "learned-axis initial controls differ from the frozen seed",
        )
        require_close(row["smoothness_C"], 0.0, "learned-axis initial S_C")
        strength = np.sum(initial["q"] * initial["q"], axis=1)
        require_array_close(
            strength,
            np.full_like(strength, float(settings["initial_strength"])),
            "learned-axis initial strength",
            atol=1.0e-15,
        )
        return {
            "mode": settings["initialization_mode"],
            "seed": int(settings["initialization_seed"]),
            "strength": float(settings["initial_strength"]),
            "q_sha256": hashlib.sha256(initial["q"].tobytes()).hexdigest(),
            "constant_same_muscle_C": True,
        }

    require(model == "raw6", f"unknown initialization model {model!r}")
    archive = settings["archive_initialization"]
    path = Path(archive["path"])
    require(path.is_file(), f"Raw6 archive is missing: {path}")
    require(sha256(path) == archive["sha256"], "Raw6 archive hash differs")
    with np.load(path, allow_pickle=False) as loaded:
        archived = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(
        set(archived)
        == {
            "step",
            "q",
            "Ainv",
            "u",
            "rest_points",
            "active_ids",
            "solver_valid",
            "physical_volume_energy",
        },
        "Raw6 archive schema differs",
    )
    require(int(archived["step"].item()) == 0, "Raw6 archive step differs")
    require(
        bool(archived["solver_valid"].item()), "Raw6 archive solver state is invalid"
    )
    require(
        bool(archived["physical_volume_energy"].item()),
        "Raw6 archive does not use physical-volume energy",
    )
    require(
        np.array_equal(archived["active_ids"], fixture.active_ids),
        "Raw6 archive IDs differ",
    )
    require(
        np.array_equal(archived["rest_points"], fixture.points),
        "Raw6 archive rest points differ",
    )
    require(
        np.array_equal(archived["q"], initial["q"]),
        "Raw6 initial q differs from archive",
    )
    archive_c, _ = q_to_c_z(np.asarray(archived["q"], dtype=np.float64), "raw6")
    require_array_close(
        np.asarray(archived["Ainv"], dtype=np.float64),
        archive_c + np.eye(3),
        "Raw6 archive q-to-B conversion",
    )
    require(
        archived["u"].shape == fixture.points.shape,
        "Raw6 archive displacement shape differs",
    )
    require(
        bool(np.isfinite(archived["q"]).all() and np.isfinite(archived["u"]).all()),
        "Raw6 archive contains nonfinite values",
    )
    require_close(
        row["smoothness_C"],
        recompute_smoothness(initial["C"], fixture),
        "Raw6 initial S_C",
    )
    return {
        "mode": settings["initialization_mode"],
        "archive": record(path),
        "archive_q_sha256": hashlib.sha256(archived["q"].tobytes()).hexdigest(),
        "archive_seed_u_sha256": hashlib.sha256(archived["u"].tobytes()).hexdigest(),
        "evaluated_u_differs_from_seed": not np.array_equal(
            initial["u"], archived["u"]
        ),
    }


def verify_case(
    case: str,
    directory: Path,
    fixture: Fixture,
    settings: dict[str, Any],
    settings_sha: str,
) -> dict[str, Any]:
    model, regularized = CASES[case]
    config = json.loads((directory / "config.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = read_trace(directory / "trace.csv")
    require(config["case"] == case, f"config case differs: {directory}")
    require(provenance["case"] == case, f"provenance case differs: {directory}")
    require(provenance["model"] == model, f"provenance model differs: {directory}")
    require(summary["case"] == case, f"summary case differs: {directory}")
    run_status = summary["status"]
    completed = run_status == "completed_fixed_budget_not_stationarity_certified"
    failed = run_status == "failed_before_completion"
    require(completed or failed, f"unsupported run status: {directory}: {run_status!r}")
    require(
        (summary["failure"] is None) == completed,
        f"run status and failure receipt differ: {directory}",
    )
    require(
        summary["parent"] == provenance["parent"], f"summary parent differs: {case}"
    )
    require(settings["smoothness_field"] == "C", f"settings do not select C: {case}")
    require_close(
        float(config["smoothness_multiplier"]),
        1.0,
        f"smoothness multiplier {case}",
    )

    require(
        sha256(Path(config["settings"])) == settings_sha,
        f"config settings differ: {directory}",
    )
    require(
        provenance["settings"]["sha256"] == settings_sha,
        f"provenance settings differ: {directory}",
    )
    require(
        sha256(Path(provenance["settings"]["path"])) == settings_sha,
        f"recorded settings path drifted: {directory}",
    )
    for name, expected_input in settings["inputs"].items():
        archived_input = provenance["inputs"][name]
        require(
            archived_input["sha256"] == expected_input["sha256"],
            f"fixture receipt differs for {case}/{name}",
        )
        require(
            sha256(Path(archived_input["path"])) == archived_input["sha256"],
            f"fixture input drifted for {case}/{name}",
        )
    validation = provenance["controls_validation"]
    require(
        validation["sha256"] == settings["controls_validation"]["sha256"],
        f"controls-validation receipt differs: {case}",
    )
    require(
        sha256(Path(validation["path"])) == validation["sha256"],
        f"controls-validation input drifted: {case}",
    )
    require(
        json.loads(Path(validation["path"]).read_text())["status"] == "passed",
        f"controls validation did not pass: {case}",
    )
    for name, diagnostic_input in provenance["diagnostics"]["inputs"].items():
        require(
            sha256(Path(diagnostic_input["path"])) == diagnostic_input["sha256"],
            f"diagnostic input drifted for {case}/{name}",
        )
    expected_weight = (
        float(settings["smoothness_weight"]) * float(config["smoothness_multiplier"])
        if regularized
        else 0.0
    )
    require_close(
        provenance["smoothness_weight"], expected_weight, f"provenance weight {case}"
    )
    require_close(
        summary["smoothness_weight"], expected_weight, f"summary weight {case}"
    )
    require(provenance["skin_enabled"] is False, f"skin enabled: {case}")
    require(provenance["magnitude_weight"] == 0.0, f"magnitude penalty enabled: {case}")
    require(provenance["rank_weight"] == 0.0, f"rank penalty enabled: {case}")
    require(provenance["upper_stress_cap"] is None, f"upper stress cap enabled: {case}")
    materials = provenance["materials"]
    require(materials["skin_E_MPa"] == 0.0, f"skin material enabled: {case}")
    require(materials["skin_prestrain"] == 0.0, f"skin prestrain enabled: {case}")
    require(materials["contact_enabled"] is False, f"contact enabled: {case}")
    expected_material = "stable-active-physical-volume"
    require(
        materials["muscle_model"] == expected_material,
        f"muscle material differs: {case}",
    )
    require_close(
        materials["muscle_mu_code_MPa"],
        0.03 / (2.0 * 1.49),
        f"muscle mu {case}",
    )
    expected_lr = float(settings["learning_rates"][model])
    require_close(provenance["optimizer"]["lr"], expected_lr, f"learning rate {case}")
    require_close(
        provenance["optimizer"]["eps"], settings["adam_eps"], f"Adam eps {case}"
    )
    require(
        tuple(provenance["optimizer"]["betas"]) == tuple(settings["betas"]),
        f"Adam betas differ: {case}",
    )
    require(provenance["optimizer"]["weight_decay"] == 0, f"Adam decay enabled: {case}")
    require(provenance["optimizer"]["amsgrad"] is False, f"AMSGrad enabled: {case}")
    require(
        provenance["optimizer"]["maximize"] is False, f"Adam maximize enabled: {case}"
    )
    require(
        int(config["steps"]) == int(provenance["optimizer"]["target_step"]),
        f"target step differs: {case}",
    )

    fresh = provenance["parent"] is None
    require(
        (config["resume"] is None) == fresh,
        f"resume/parent declaration differs: {case}",
    )
    start_step = int(provenance["optimizer"]["start_step"])
    target_step = int(provenance["optimizer"]["target_step"])
    if model == "raw6":
        require(fresh, "Raw6 smooth primary must start with fresh Adam")
        require(
            provenance["optimizer"]["initialization_seed"] is None,
            "Raw6 initialization seed must be absent",
        )
        require(start_step == 0, "Raw6 start differs")
        require(int(config["steps"]) == 200, "Raw6 update budget differs")
        require(
            int(config["checkpoint_interval"]) == 10, "Raw6 checkpoint interval differs"
        )
        require(
            settings["initialization_mode"] == "archived_controls_and_seed_fresh_adam",
            "Raw6 initialization mode differs",
        )
    else:
        require(
            int(config["initialization_seed"]) == int(settings["initialization_seed"]),
            f"learned initialization seed differs: {case}",
        )
        require(
            int(provenance["optimizer"]["initialization_seed"])
            == int(settings["initialization_seed"]),
            f"learned provenance initialization seed differs: {case}",
        )
        if fresh:
            require(start_step == 0, f"learned initial phase start differs: {case}")
            require(
                target_step == 128,
                f"learned initial phase budget differs: {case}",
            )
        else:
            require(start_step == 128, f"learned continuation start differs: {case}")
            require(
                target_step == 256,
                f"failed learned continuation budget differs: {case}",
            )
        require(
            int(config["checkpoint_interval"]) == 16,
            f"learned checkpoint interval differs: {case}",
        )
        require(
            settings["initialization_mode"] == "seeded_axis_fresh_adam",
            f"learned initialization mode differs: {case}",
        )
        require(
            settings["initialization_seed"] == 20260909, f"learned seed differs: {case}"
        )
    trace_receipt = validate_trace(
        case, rows, provenance, expected_weight, completed=completed
    )
    parent_receipt, parent_state = validate_parent_checkpoint(
        config,
        provenance,
        case,
        model,
        fixture,
        settings_sha,
        expected_weight,
        trace_receipt["start_step"],
        float(rows[trace_receipt["start_step"]]["gradient_rms"]),
    )
    parent_prefix = None
    if parent_state is not None:
        parent_prefix = validate_copied_parent_prefix(
            directory,
            Path(provenance["parent"]["checkpoint"]),
            rows,
            trace_receipt["start_step"],
            parent_state,
        )
    require(
        int(summary["last_evaluated_step"]) == int(rows[-1]["step"]),
        f"summary last step differs: {case}",
    )
    require(
        int(summary["best_step"]) == trace_receipt["best_step"],
        f"summary best fit differs: {case}",
    )
    require(
        int(summary["best_objective_step"]) == trace_receipt["best_objective_step"],
        f"summary best objective differs: {case}",
    )
    compare_mapping_subset(
        summary["last_metrics"], rows[-1], f"summary last metrics {case}"
    )
    best_row = next(row for row in rows if int(row["step"]) == summary["best_step"])
    best_objective_row = next(
        row for row in rows if int(row["step"]) == summary["best_objective_step"]
    )
    compare_mapping_subset(
        summary["best_metrics"], best_row, f"summary best metrics {case}"
    )
    compare_mapping_subset(
        summary["best_objective_metrics"],
        best_objective_row,
        f"summary best objective metrics {case}",
    )
    if not regularized:
        require(
            trace_receipt["best_step"] == trace_receipt["best_objective_step"],
            f"unregularized best selections differ: {case}",
        )

    row_by_step = {int(row["step"]): row for row in rows}
    start = trace_receipt["start_step"]
    target = trace_receipt["target_step"]
    last = trace_receipt["last_accepted_step"]
    interval = int(config["checkpoint_interval"])
    expected_steps = [
        step
        for step in range(last + 1)
        if step in {0, 1} or step % interval == 0 or (completed and step == target)
    ]
    checkpoint_paths = {
        step: directory / f"step-{step:04d}.npz" for step in expected_steps
    }
    actual_checkpoint_names = {path.name for path in directory.glob("step-*.npz")}
    require(
        actual_checkpoint_names == {path.name for path in checkpoint_paths.values()},
        f"full checkpoint set differs: {case}",
    )
    state_receipts = []
    retained_steps = {
        0,
        start,
        last,
        int(summary["best_step"]),
        int(summary["best_objective_step"]),
    }
    checkpoint_arrays: dict[int, dict[str, np.ndarray]] = {}
    for step, path in checkpoint_paths.items():
        receipt, arrays = validate_state(
            path,
            model,
            fixture,
            row_by_step[step],
            check_volume=True,
        )
        state_receipts.append(receipt)
        validate_surface(
            directory / f"surface-{step:04d}.npz",
            step,
            fixture,
            arrays["u"],
        )
        if step in retained_steps:
            checkpoint_arrays[step] = arrays

    if parent_state is not None and start in checkpoint_arrays:
        require(
            np.array_equal(parent_state["q"], checkpoint_arrays[start]["q"]),
            f"first continuation state differs from parent q: {case}",
        )
        require(
            np.array_equal(parent_state["u"], checkpoint_arrays[start]["u"]),
            f"first continuation state differs from parent u: {case}",
        )

    aliases = {
        "best": ("best.npz", best_row),
        "best_objective": ("best-objective.npz", best_objective_row),
    }
    if completed:
        aliases["last"] = ("last.npz", rows[-1])
    alias_receipts = {}
    alias_arrays = {}
    for name, (filename, row) in aliases.items():
        receipt, arrays = validate_state(
            directory / filename,
            model,
            fixture,
            row,
            check_volume=True,
        )
        alias_receipts[name] = receipt
        alias_arrays[name] = arrays
        step = int(row["step"])
        if step in checkpoint_arrays:
            for field in ("q", "C", "Z", "u"):
                require(
                    np.array_equal(arrays[field], checkpoint_arrays[step][field]),
                    f"{name} alias differs from step checkpoint: {case}/{field}",
                )
    if not regularized:
        for field in ("q", "C", "Z", "u"):
            require(
                np.array_equal(
                    alias_arrays["best"][field], alias_arrays["best_objective"][field]
                ),
                f"unregularized best aliases differ: {case}/{field}",
            )

    expected_surface_names = {f"surface-{int(row['step']):04d}.npz" for row in rows}
    actual_surface_names = {path.name for path in directory.glob("surface-*.npz")}
    require(
        actual_surface_names == expected_surface_names,
        f"surface state set differs: {case}",
    )
    surface_manifest = hashlib.sha256()
    for row in rows:
        step = int(row["step"])
        path = directory / f"surface-{step:04d}.npz"
        full = checkpoint_arrays.get(step)
        digest = validate_surface(
            path, step, fixture, None if full is None else full["u"]
        )
        surface_manifest.update(f"{path.name}\0{digest}\n".encode())
    for name, arrays in alias_arrays.items():
        step = int(alias_receipts[name]["step"])
        validate_surface(
            directory / f"surface-{step:04d}.npz", step, fixture, arrays["u"]
        )

    initialization = validate_initialization(
        model,
        settings,
        fixture,
        checkpoint_arrays[0],
        rows[0],
    )
    require(rows[0]["projection_rms"] == 0.0, f"initial projection is nonzero: {case}")

    solver = validate_solver_receipts(directory / "solver-receipts.jsonl", rows)
    optimizer, optimizer_arrays = validate_optimizer(
        directory / "optimizer-latest.pt",
        case,
        model,
        rows[-1],
        provenance,
        settings_sha,
        expected_weight,
        fixture,
        trace_receipt["best_step"],
        trace_receipt["best_objective_step"],
    )
    validate_surface(
        directory / f"surface-{last:04d}.npz",
        last,
        fixture,
        optimizer_arrays["u"],
    )
    if completed:
        optimizer_alias_comparison = {}
        for field in ("q", "u"):
            require(
                np.array_equal(optimizer_arrays[field], alias_arrays["last"][field]),
                f"optimizer and last alias differ: {case}/{field}",
            )
            optimizer_alias_comparison[field] = {"bitwise_equal": True}
        for field in ("C", "Z"):
            left = optimizer_arrays[field]
            right = alias_arrays["last"][field]
            require_array_close(
                left,
                right,
                f"optimizer reconstruction and last alias differ: {case}/{field}",
            )
            difference = left - right
            denominator = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
            optimizer_alias_comparison[field] = {
                "bitwise_equal": np.array_equal(left, right),
                "max_abs": float(np.max(np.abs(difference))),
                "relative_l2": (
                    0.0
                    if denominator == 0.0
                    else float(np.linalg.norm(difference) / denominator)
                ),
            }
        optimizer["last_alias_comparison"] = optimizer_alias_comparison
        endpoint = {
            **alias_receipts["last"],
            "artifact_role": "completed_last_full_state_alias",
            "optimizer_alias_comparison": optimizer_alias_comparison,
        }
    else:
        endpoint = optimizer
    failure_evidence = (
        None
        if completed
        else validate_failure_evidence(
            directory,
            summary["failure"],
            model,
            fixture,
            last,
            target,
        )
    )
    source_receipt = verify_sources(directory, provenance)
    study_target_step = 256 if model == "learned-axis" else 200
    study_target_reached = completed and target == study_target_step
    return {
        "directory": str(directory.resolve()),
        "case": case,
        "run_status": run_status,
        "evidence_verification_status": "passed",
        "fixed_budget_completed": completed,
        "declared_phase_status": (
            "completed_declared_phase" if completed else "failed_declared_phase"
        ),
        "study_target_step": study_target_step,
        "study_target_reached": study_target_reached,
        "model": model,
        "regularized": regularized,
        "smoothness_multiplier": float(config["smoothness_multiplier"]),
        "effective_smoothness_weight": expected_weight,
        "fresh": fresh,
        "config": record(directory / "config.json"),
        "provenance": record(directory / "provenance.json"),
        "summary": record(directory / "summary.json"),
        "trace": {**record(directory / "trace.csv"), **trace_receipt},
        "solver_receipts": solver,
        "optimizer": optimizer,
        "failure_evidence": failure_evidence,
        "endpoint": endpoint,
        "parent_checkpoint": parent_receipt,
        "parent_prefix": parent_prefix,
        "initialization": initialization,
        "full_checkpoints": state_receipts,
        "full_checkpoint_schedule": {
            "expected_steps": expected_steps,
            "schedule_complete": True,
            "last_accepted_step_has_full_checkpoint": last in expected_steps,
        },
        "aliases": alias_receipts,
        "surface_states": {
            "count": len(rows),
            "manifest_sha256": surface_manifest.hexdigest(),
        },
        "sources": source_receipt,
        "materials": materials,
        "inversions_are_diagnostics_only": True,
    }


def read_archived_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = [
            {key: parse_cell(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    require(bool(rows), f"empty archived trace: {path}")
    required = {
        "step",
        "objective_mm2",
        "fit_rms_mm",
        "target_projection",
        "gradient_rms",
        "forward_success",
        "adjoint_success",
        "best_step",
    }
    require(required <= rows[0].keys(), f"archived trace schema differs: {path}")
    return rows


def validate_archived_raw6_state(
    path: Path,
    fixture: Fixture,
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    expected_keys = {
        "step",
        "q",
        "Ainv",
        "u",
        "rest_points",
        "active_ids",
        "solver_valid",
        "physical_volume_energy",
    }
    with np.load(path, allow_pickle=False) as loaded:
        require(
            set(loaded.files) == expected_keys, f"archived state schema differs: {path}"
        )
        state = {name: np.asarray(loaded[name]) for name in loaded.files}
    require(
        int(state["step"]) == int(row["step"]), f"archived state step differs: {path}"
    )
    require(bool(state["solver_valid"]), f"archived solver state is invalid: {path}")
    require(
        bool(state["physical_volume_energy"]),
        f"archived energy law flag differs: {path}",
    )
    require(
        np.array_equal(state["active_ids"], fixture.active_ids),
        f"archived IDs differ: {path}",
    )
    require(
        np.array_equal(state["rest_points"], fixture.points),
        f"archived rest points differ: {path}",
    )
    q = np.asarray(state["q"], dtype=np.float64)
    c, z = q_to_c_z(q, "raw6")
    b = c + np.eye(3)
    require_array_close(state["Ainv"], b, f"archived q-to-B conversion at {path}")
    u = np.asarray(state["u"], dtype=np.float64)
    require(
        q.shape == (len(fixture.active_ids), 6), f"archived q shape differs: {path}"
    )
    require(u.shape == fixture.points.shape, f"archived u shape differs: {path}")
    for key, value in state_geometry_metrics(u, fixture).items():
        require_close(value, row[key], f"archived geometry {key} at {path}")
    volume = state_volume_metrics(u, fixture)
    archived_volume_key = "active_volume_weighted_RMS_detF_minus_1"
    require_close(
        volume["active_volume_weighted_rms_detF_minus_1"],
        row[archived_volume_key],
        f"archived active volume metric at {path}",
    )
    for key in ("detF_min", "detF_max", "inverted_all_cells", "inverted_active_cells"):
        require_close(volume[key], row[key], f"archived volume {key} at {path}")
    return (
        {
            **record(path),
            "step": int(state["step"]),
            "smoothness_C": recompute_smoothness(c, fixture),
            "smoothness_Z": recompute_smoothness(z, fixture),
        },
        {"q": q, "C": c, "Z": z, "u": u},
    )


def verify_archived_raw6_off(
    directory: Path,
    fixture: Fixture,
    settings: dict[str, Any],
) -> dict[str, Any]:
    reference = settings["reference"]
    require(
        directory.resolve() == Path(reference["path"]).resolve(),
        "Raw6 off reference path differs",
    )
    require(
        sha256(directory / "provenance.json") == reference["provenance_sha256"],
        "Raw6 off provenance hash differs",
    )
    require(
        sha256(directory / "trace.csv") == reference["trace_sha256"],
        "Raw6 off trace hash differs",
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = read_archived_trace(directory / "trace.csv")
    require(
        summary["status"] == "completed_200_updates_not_stationarity_certified",
        "Raw6 off status differs",
    )
    require(summary["failure"] is None, "Raw6 off has a failure")
    require(
        [int(row["step"]) for row in rows] == list(range(201)), "Raw6 off steps differ"
    )
    best_step = 0
    best_value = math.inf
    for row in rows:
        step = int(row["step"])
        require(row["forward_success"] is True, f"Raw6 off forward failed at {step}")
        require(row["adjoint_success"] is True, f"Raw6 off adjoint failed at {step}")
        require_close(
            3.0 * row["objective_mm2"],
            row["fit_rms_mm"] ** 2,
            f"Raw6 off fit identity {step}",
        )
        if row["objective_mm2"] < best_value:
            best_value, best_step = row["objective_mm2"], step
        require(
            int(row["best_step"]) == best_step,
            f"Raw6 off running best differs at {step}",
        )
    require(int(summary["best_step"]) == best_step, "Raw6 off summary best differs")

    row_by_step = {int(row["step"]): row for row in rows}
    expected_steps = list(range(0, 201, 10))
    require(best_step in expected_steps, "Raw6 off best lacks a full checkpoint")
    require(
        {path.name for path in directory.glob("step-*.npz")}
        == {f"step-{step:04d}.npz" for step in expected_steps},
        "Raw6 off checkpoint set differs",
    )
    states = {}
    receipts = []
    for step in expected_steps:
        receipt, state = validate_archived_raw6_state(
            directory / f"step-{step:04d}.npz", fixture, row_by_step[step]
        )
        receipts.append(receipt)
        if step in {0, 200, best_step}:
            states[step] = state
        validate_surface(
            directory / f"surface-{step:04d}.npz", step, fixture, state["u"]
        )
    expected_surfaces = {f"surface-{step:04d}.npz" for step in range(201)}
    require(
        {path.name for path in directory.glob("surface-*.npz")} == expected_surfaces,
        "Raw6 off surface set differs",
    )
    surface_manifest = hashlib.sha256()
    for step in range(201):
        path = directory / f"surface-{step:04d}.npz"
        digest = validate_surface(path, step, fixture, None)
        surface_manifest.update(f"{path.name}\0{digest}\n".encode())
    for name, step in (
        ("last.npz", 200),
        ("best.npz", best_step),
        ("final.npz", best_step),
    ):
        receipt, state = validate_archived_raw6_state(
            directory / name, fixture, row_by_step[step]
        )
        for field in ("q", "C", "Z", "u"):
            require(
                np.array_equal(state[field], states[step][field]),
                f"Raw6 off alias differs: {name}/{field}",
            )

    optimizer_path = directory / "optimizer-latest.pt"
    saved = torch.load(optimizer_path, map_location="cpu", weights_only=False)
    require(int(saved["step"]) == 200, "Raw6 off optimizer step differs")
    require(
        np.array_equal(saved["q"].numpy(), states[200]["q"]),
        "Raw6 off optimizer q differs",
    )
    require(
        np.array_equal(saved["u"], states[200]["u"]), "Raw6 off optimizer u differs"
    )
    require_close(
        float(saved["gradient"].square().mean().sqrt()),
        rows[-1]["gradient_rms"],
        "Raw6 off gradient RMS",
    )
    group = saved["optimizer"]["param_groups"][0]
    require_close(group["lr"], 0.3, "Raw6 off Adam lr")
    require_close(group["eps"], 0.01, "Raw6 off Adam eps")
    require(tuple(group["betas"]) == (0.9, 0.999), "Raw6 off Adam betas differ")
    require(
        optimizer_step_values(saved["optimizer"]["state"]) == [200],
        "Raw6 off Adam count differs",
    )
    solver_receipts = [
        json.loads(line)
        for line in (directory / "solver-receipts.jsonl").read_text().splitlines()
        if line
    ]
    require(len(solver_receipts) == 201, "Raw6 off solver receipt count differs")
    for step, receipt in enumerate(solver_receipts):
        require(
            int(receipt["step"]) == step,
            f"Raw6 off solver receipt step differs: {step}",
        )
        require(
            receipt["forward"]["success"] is True,
            f"Raw6 off forward receipt failed: {step}",
        )
        require(
            receipt["adjoint"]["success"] is True,
            f"Raw6 off adjoint receipt failed: {step}",
        )
    sources = verify_sources(
        directory,
        provenance,
        {
            "__main__",
            "local_physics",
            "experiment_io",
            "continuation_metrics",
            "volume_preserving_active",
        },
    )
    return {
        "directory": str(directory.resolve()),
        "status": summary["status"],
        "trace": {
            **record(directory / "trace.csv"),
            "evaluated_states": len(rows),
            "best_step": best_step,
        },
        "full_checkpoints": receipts,
        "optimizer": {**record(optimizer_path), "adam_step": 200},
        "solver_receipts": {
            **record(directory / "solver-receipts.jsonl"),
            "count": len(solver_receipts),
            "all_successful": True,
        },
        "surface_states": {
            "count": 201,
            "manifest_sha256": surface_manifest.hexdigest(),
        },
        "initial": receipts[0],
        "last": receipts[-1],
        "materials": summary["materials"],
        "sources": sources,
    }


def verify_cross_case(
    results: dict[str, dict[str, Any]],
    directories: dict[str, Path],
    fixture: Fixture,
    raw6_off_dir: Path,
) -> dict[str, Any]:
    require(
        results["learned-axis"]["trace"]["target_step"] in {128, 256},
        "learned off target differs",
    )
    require(
        results["learned-axis-smooth"]["trace"]["target_step"] in {128, 256},
        "learned smooth target differs",
    )
    require(
        results["baseline-smooth"]["trace"]["target_step"] == 200,
        "Raw6 smooth target differs",
    )
    source_maps = [
        result["sources"]["snapshot_sha256_by_module"] for result in results.values()
    ]
    require(
        all(value == source_maps[0] for value in source_maps[1:]),
        "source snapshots differ across cases",
    )
    base_material = results["learned-axis"]["materials"]
    for case, result in results.items():
        require(
            result["materials"] == base_material,
            f"material specification differs: {case}",
        )

    learned_directories = {
        "off": directories["learned-axis"],
        "smooth": directories["learned-axis-smooth"],
    }
    matched_steps = {}
    paired_thresholds = {
        "field_max_abs": 1.0e-6,
        "field_relative_l2": 1.0e-4,
        "geometry_rms_mm": 1.0e-6,
    }
    paired_passed = True
    common_last_accepted_step = min(
        results[case]["trace"]["last_accepted_step"]
        for case in ("learned-axis", "learned-axis-smooth")
    )
    compared_steps = [step for step in (0, 1) if step <= common_last_accepted_step]
    for step in compared_steps:
        states = {}
        for name, directory in learned_directories.items():
            with np.load(
                directory / f"step-{step:04d}.npz", allow_pickle=False
            ) as loaded:
                states[name] = {
                    field: np.asarray(loaded[field]) for field in ("q", "C", "Z", "u")
                }
        field_errors = {}
        for field in ("q", "C", "Z"):
            off = states["off"][field]
            smooth = states["smooth"][field]
            difference = off - smooth
            denominator = max(float(np.linalg.norm(off)), float(np.linalg.norm(smooth)))
            require(denominator > 0.0, f"learned matched {field} norm is zero")
            max_abs = float(np.max(np.abs(difference)))
            relative_l2 = float(np.linalg.norm(difference) / denominator)
            field_errors[field] = {
                "bitwise_equal": np.array_equal(off, smooth),
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "max_abs_passed": max_abs <= paired_thresholds["field_max_abs"],
                "relative_l2_passed": (
                    relative_l2 <= paired_thresholds["field_relative_l2"]
                ),
            }
        geometry_rms_mm = float(
            1000.0
            * np.linalg.norm(states["off"]["u"] - states["smooth"]["u"])
            / math.sqrt(len(fixture.points))
        )
        if step == 0:
            for field in ("q", "C", "Z"):
                require(
                    np.array_equal(states["off"][field], states["smooth"][field]),
                    f"learned step-0 {field} differs",
                )
        geometry_passed = geometry_rms_mm <= paired_thresholds["geometry_rms_mm"]
        step_passed = geometry_passed and all(
            error["max_abs_passed"] and error["relative_l2_passed"]
            for error in field_errors.values()
        )
        paired_passed = paired_passed and step_passed
        matched_steps[str(step)] = {
            "field_errors": field_errors,
            "u_bitwise_equal": np.array_equal(
                states["off"]["u"], states["smooth"]["u"]
            ),
            "geometry_rms_mm": geometry_rms_mm,
            "geometry_rms_mm_passed": geometry_passed,
            "passed": step_passed,
        }
    first_update_available = 1 in compared_steps
    paired_result = paired_passed if first_update_available else None

    require(
        results["learned-axis"]["effective_smoothness_weight"] == 0.0,
        "learned off weight is nonzero",
    )
    require(
        results["learned-axis-smooth"]["effective_smoothness_weight"] > 0.0,
        "learned smooth weight is not positive",
    )
    require(
        results["baseline-smooth"]["effective_smoothness_weight"] > 0.0,
        "Raw6 smooth weight is not positive",
    )
    with np.load(
        directories["baseline-smooth"] / "step-0000.npz", allow_pickle=False
    ) as smooth:
        smooth_q = np.asarray(smooth["q"])
        smooth_u = np.asarray(smooth["u"])
    with np.load(raw6_off_dir / "step-0000.npz", allow_pickle=False) as off:
        off_q = np.asarray(off["q"])
        off_u = np.asarray(off["u"])
    require(np.array_equal(smooth_q, off_q), "Raw6 off/smooth initial q differs")
    raw6_geometry_rms_mm = float(
        1000.0 * np.linalg.norm(smooth_u - off_u) / math.sqrt(len(fixture.points))
    )
    require(raw6_geometry_rms_mm <= 1.0e-5, "Raw6 off/smooth initial geometry differs")
    return {
        "learned_declared_target_steps": {
            "off": results["learned-axis"]["trace"]["target_step"],
            "smooth": results["learned-axis-smooth"]["trace"]["target_step"],
        },
        "learned_both_fixed_budget_completed": all(
            results[case]["fixed_budget_completed"]
            for case in ("learned-axis", "learned-axis-smooth")
        ),
        "learned_both_declared_phases_completed": all(
            results[case]["declared_phase_status"] == "completed_declared_phase"
            for case in ("learned-axis", "learned-axis-smooth")
        ),
        "learned_completed_common_128_pair": all(
            results[case]["fixed_budget_completed"]
            and results[case]["trace"]["last_accepted_step"] >= 128
            for case in ("learned-axis", "learned-axis-smooth")
        ),
        "learned_study_target_256_reached": all(
            results[case]["study_target_reached"]
            for case in ("learned-axis", "learned-axis-smooth")
        ),
        "raw6_target_step": 200,
        "source_snapshots_identical": True,
        "materials_identical": True,
        "learned_matched_initial_and_first_update": {
            "status": (
                "unavailable_first_update_not_accepted_by_both_runs"
                if not first_update_available
                else (
                    "passed"
                    if paired_passed
                    else "failed_predeclared_first_update_threshold"
                )
            ),
            "available": first_update_available,
            "passed": paired_result,
            "predeclared_thresholds": paired_thresholds,
            "steps": matched_steps,
            "interpretation": (
                "A failed threshold is reported as numerical trajectory sensitivity; "
                "it does not invalidate independently verified saved-state evidence."
            ),
        },
        "raw6_matched_initial": {
            "q_bitwise_equal": True,
            "u_bitwise_equal": np.array_equal(smooth_u, off_u),
            "geometry_rms_mm": raw6_geometry_rms_mm,
        },
        "weights_are_model_specific": {
            "learned_axis": results["learned-axis-smooth"][
                "effective_smoothness_weight"
            ],
            "raw6": results["baseline-smooth"]["effective_smoothness_weight"],
        },
        "endpoint_smoothness_diagnostics": {
            case: {
                "S_C": result["endpoint"]["smoothness_C"],
                "S_Z": result["endpoint"]["smoothness_Z"],
            }
            for case, result in results.items()
        },
    }


def main(cfg: Config) -> None:
    torch.set_default_device("cpu")
    directories = case_directories(cfg)
    preflight_primary_evidence(directories)
    settings_paths = {
        "learned-axis": cfg.learned_axis_settings,
        "baseline-smooth": cfg.raw6_settings,
    }
    settings = {
        name: json.loads(path.read_text()) for name, path in settings_paths.items()
    }
    for name, value in settings.items():
        require(
            value["status"] == "frozen_before_primary_runs",
            f"settings are not frozen: {name}",
        )
        require(value["smoothness_field"] == "C", f"settings do not select C: {name}")
        require_close(
            value["smooth_length_m"], SMOOTH_LENGTH_M, f"smoothness length {name}"
        )
        validation = value["controls_validation"]
        require(
            sha256(Path(validation["path"])) == validation["sha256"],
            f"validation drifted: {name}",
        )
        require(
            json.loads(Path(validation["path"]).read_text())["status"] == "passed",
            f"validation failed: {name}",
        )
    require(
        settings["learned-axis"]["inputs"] == settings["baseline-smooth"]["inputs"],
        "calibrations use different fixtures",
    )
    fixture_paths = {
        Path(json.loads((directory / "config.json").read_text())["fixture"]).resolve()
        for directory in directories.values()
    }
    require(len(fixture_paths) == 1, f"case fixtures differ: {fixture_paths}")
    fixture_path = fixture_paths.pop()
    common_inputs = settings["learned-axis"]["inputs"]
    for name, source in common_inputs.items():
        path = fixture_path / name
        require(sha256(path) == source["sha256"], f"fixture input hash differs: {path}")
    fixture = load_fixture(fixture_path)
    settings_by_case = {
        "learned-axis": settings["learned-axis"],
        "learned-axis-smooth": settings["learned-axis"],
        "baseline-smooth": settings["baseline-smooth"],
    }
    results = {
        case: verify_case(
            case,
            directories[case],
            fixture,
            settings_by_case[case],
            sha256(
                cfg.learned_axis_settings
                if case.startswith("learned-axis")
                else cfg.raw6_settings
            ),
        )
        for case in CASES
    }
    archived_raw6_off = verify_archived_raw6_off(
        cfg.raw6_off_dir,
        fixture,
        settings["baseline-smooth"],
    )
    for key in (
        "fat_E_MPa",
        "fat_nu",
        "fat_mu_code_MPa",
        "fat_lambda_code_MPa",
        "muscle_E_MPa",
        "muscle_nu",
        "muscle_mu_code_MPa",
        "muscle_lambda_code_MPa",
        "aponeurosis_E_MPa",
        "aponeurosis_nu",
        "skin_E_MPa",
        "contact_enabled",
    ):
        require(
            archived_raw6_off["materials"][key]
            == results["baseline-smooth"]["materials"][key],
            f"reused/new Raw6 material differs: {key}",
        )
    cross_case = verify_cross_case(results, directories, fixture, cfg.raw6_off_dir)
    failed_cases = [
        case for case, result in results.items() if not result["fixed_budget_completed"]
    ]
    paired_check_passed = cross_case["learned_matched_initial_and_first_update"][
        "passed"
    ]
    output = {
        "status": (
            "passed_retained_evidence_with_failed_primary_runs"
            if failed_cases
            else (
                "passed"
                if paired_check_passed
                else "passed_evidence_with_failed_predeclared_paired_check"
            )
        ),
        "evidence_verification_status": "passed",
        "all_primary_fixed_budgets_completed": not failed_cases,
        "all_study_targets_reached": all(
            result["study_target_reached"] for result in results.values()
        ),
        "failed_primary_cases": failed_cases,
        "predeclared_paired_check_passed": paired_check_passed,
        "study_checks_passed": not failed_cases and paired_check_passed,
        "scope": "CPU-only saved-state verification; no equilibrium solve or optimizer update",
        "settings": {name: record(path) for name, path in settings_paths.items()},
        "fixture": {
            "path": str(fixture.path.resolve()),
            "active_cells": len(fixture.active_ids),
            "same_muscle_edges": len(fixture.graph_i),
            "active_volume_m3": float(fixture.active_volume.sum()),
            "inputs": {name: record(fixture.path / name) for name in common_inputs},
        },
        "cross_case": cross_case,
        "cases": results,
        "reused_raw6_off": archived_raw6_off,
        "policy": {
            "source_snapshots": "Archived snapshots and their recorded hashes are authoritative. Current-source drift is reported but does not invalidate a run.",
            "inversions": "Recorded and recomputed as diagnostics; never used as a rejection gate.",
            "learned_axis_psd": (
                "Analytic rank-one PSD is checked using a per-cell eigensolver "
                f"roundoff bound of {PSD_ROUNDOFF_FACTOR} * float64 epsilon * "
                "max(1, spectral scale)."
            ),
            "failed_trajectories": (
                "Only the accepted solver-valid prefix is treated as trajectory evidence. "
                "Solver-invalid failure controls are verified and reported separately; "
                "a retained-evidence pass is not fixed-budget completion."
            ),
        },
    }
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    require(
        not any(cfg.output_dir.iterdir()),
        f"refuse to overwrite verifier output: {cfg.output_dir}",
    )
    write_json(cfg.output_dir / "summary.json", output)
    cherries.log_metrics(
        {
            "verification/cases": len(results) + 1,
            "verification/evaluated_states": sum(
                result["trace"]["evaluated_states"] for result in results.values()
            )
            + archived_raw6_off["trace"]["evaluated_states"],
            "verification/full_checkpoints": sum(
                len(result["full_checkpoints"]) for result in results.values()
            ),
            "verification/current_source_drifts": sum(
                item["differs_from_authoritative_snapshot"]
                for result in results.values()
                for item in result["sources"]["current_source"].values()
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
