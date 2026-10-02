# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""CPU-only posthoc validation of saved full-face comparison endpoints."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.sparse as sp

from liblaf import cherries

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[6]
        / "exp/2026/09/14/dominant-activation-ablation/src"
    ),
)
from study_metrics import StudyMetrics


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = cherries.input("10-comparison")
    output: Path = cherries.output("20-verification/checks.json", mkdir=True)


def _load_state(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as state:
        required = {"q", "u", "step", "solver_valid", "physical_volume_energy"}
        absent = required - set(state.files)
        if absent:
            raise ValueError(f"{path} lacks {sorted(absent)}")
        if not bool(state["solver_valid"]) or not bool(state["physical_volume_energy"]):
            raise ValueError(f"{path} is not a valid physical-volume state")
        return (
            np.asarray(state["q"], dtype=np.float64),
            np.asarray(state["u"], dtype=np.float64),
            int(state["step"]),
        )


def _stiffness(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, sp.csr_matrix]:
    p0, p1, p2 = (points[triangles[:, column]] for column in range(3))
    double_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    if np.any(double_area <= 0.0):
        raise ValueError("degenerate reference skin triangle")
    cot0 = np.einsum("ij,ij->i", p1 - p0, p2 - p0) / double_area
    cot1 = np.einsum("ij,ij->i", p2 - p1, p0 - p1) / double_area
    cot2 = np.einsum("ij,ij->i", p0 - p2, p1 - p2) / double_area
    i = np.concatenate((triangles[:, 1], triangles[:, 2], triangles[:, 0]))
    j = np.concatenate((triangles[:, 2], triangles[:, 0], triangles[:, 1]))
    weight = 0.5 * np.concatenate((cot0, cot1, cot2))
    stiffness = sp.coo_matrix(
        (
            np.concatenate((-weight, -weight, weight, weight)),
            (np.concatenate((i, j, i, j)), np.concatenate((j, i, i, j))),
        ),
        shape=(len(points), len(points)),
    ).tocsr()
    return 0.5 * double_area, stiffness


def _direct_gradient_energy(
    residual: np.ndarray, points: np.ndarray, triangles: np.ndarray
) -> float:
    vertices = points[triangles]
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    normal_squared = np.sum(normal**2, axis=1)
    gradient = (
        np.stack(
            (
                np.cross(normal, vertices[:, 2] - vertices[:, 1]),
                np.cross(normal, vertices[:, 0] - vertices[:, 2]),
                np.cross(normal, vertices[:, 1] - vertices[:, 0]),
            ),
            axis=1,
        )
        / normal_squared[:, None, None]
    )
    field_gradient = np.einsum("tvi,tvj->tij", residual[triangles], gradient)
    area = 0.5 * np.sqrt(normal_squared)
    return float(np.sum(area * np.sum(field_gradient**2, axis=(1, 2))) / area.sum())


def _activation(q: np.ndarray) -> np.ndarray:
    B = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    B[:, (0, 1, 2), (0, 1, 2)] += q[:, :3]
    B[:, 0, 1] = B[:, 1, 0] = q[:, 3]
    B[:, 1, 2] = B[:, 2, 1] = q[:, 4]
    B[:, 0, 2] = B[:, 2, 0] = q[:, 5]
    return B


def _close(actual: float, reported: float, *, tolerance: float = 1e-9) -> float:
    error = abs(actual - reported)
    if error > tolerance * max(1.0, abs(actual), abs(reported)):
        raise AssertionError((actual, reported, error, tolerance))
    return error


def _endpoint_metrics(
    q: np.ndarray,
    u: np.ndarray,
    *,
    rest: np.ndarray,
    skin_ids: np.ndarray,
    target_skin: np.ndarray,
    weights: np.ndarray,
    tets: np.ndarray,
    dm_inv: np.ndarray,
    active_ids: np.ndarray,
    fixed: np.ndarray,
    prescribed: np.ndarray,
    gradient_energy: float,
) -> dict[str, float | int]:
    prediction = u[skin_ids]
    residual = prediction - target_skin
    mean = np.sum(weights[:, None] * residual, axis=0)
    B = _activation(q)
    J = np.linalg.det(
        np.transpose((rest + u)[tets[:, 1:]] - (rest + u)[tets[:, :1]], (0, 2, 1))
        @ dm_inv
    )
    eigenvalues = np.linalg.eigvalsh(B)
    return {
        "position_loss_component_mm2": float(
            1e6 * np.sum(weights[:, None] * residual**2) / 3
        ),
        "surface_gradient_loss": gradient_energy,
        "surface_gradient_rms": float(math.sqrt(gradient_energy)),
        "fit_rms_mm": float(1000 * math.sqrt(np.sum(weights[:, None] * residual**2))),
        "motion_rms_mm": float(
            1000 * math.sqrt(np.sum(weights[:, None] * prediction**2))
        ),
        "centered_fit_rms_mm": float(
            1000 * math.sqrt(np.sum(weights[:, None] * (residual - mean) ** 2))
        ),
        "mean_error_norm_mm": float(1000 * np.linalg.norm(mean)),
        "target_projection": float(
            np.sum(weights[:, None] * prediction * target_skin)
            / np.sum(weights[:, None] * target_skin**2)
        ),
        "detF_min": float(J.min()),
        "detF_max": float(J.max()),
        "inverted_all_cells": int(np.count_nonzero(J <= 0)),
        "inverted_active_cells": int(np.count_nonzero(J[active_ids] <= 0)),
        "non_spd_active_cells": int(np.count_nonzero(eigenvalues[:, 0] <= 0)),
        "activation_eigen_min": float(eigenvalues.min()),
        "activation_eigen_max": float(eigenvalues.max()),
        "fixed_displacement_error": float(np.max(np.abs(u[fixed] - prescribed[fixed]))),
    }


def main(cfg: Config) -> None:
    source = cfg.comparison_dir
    validation_dir = source.parent / "06-validation"
    validation_protocol = json.loads((validation_dir / "protocol.json").read_text())
    comparison_protocol = json.loads((source / "protocol.json").read_text())
    validation_sources = {
        name: details["sha256"]
        for name, details in validation_protocol["sources"].items()
    }
    comparison_sources = {
        name: details["sha256"]
        for name, details in comparison_protocol["sources"].items()
    }
    if validation_sources != comparison_sources:
        raise AssertionError("validation and comparison numerical source hashes differ")
    if validation_protocol["fixture"] != comparison_protocol["fixture"]:
        raise AssertionError("validation and comparison fixture receipts differ")
    for name in ("learning_rate", "adam_eps"):
        if validation_protocol["config"][name] != comparison_protocol["config"][name]:
            raise AssertionError(f"validation and comparison {name} differ")
    validation_calibration = json.loads(
        (validation_dir / "calibration.json").read_text()
    )
    comparison_calibration = json.loads((source / "calibration.json").read_text())
    if validation_calibration != comparison_calibration:
        raise AssertionError(
            "comparison calibration differs from validation calibration"
        )
    if comparison_calibration != comparison_protocol["gradient_scale_calibration"]:
        raise AssertionError("comparison protocol has a different gradient calibration")
    for name in ("learning_rate", "adam_eps"):
        if comparison_calibration[name] != comparison_protocol["config"][name]:
            raise AssertionError(f"calibration {name} differs from comparison config")
    expected_steps = int(comparison_protocol["config"]["steps"])
    with np.load(source / "mesh.npz", allow_pickle=False) as data:
        rest = np.asarray(data["rest_points"], dtype=np.float64)
        skin_ids = np.asarray(data["skin_ids"], dtype=np.int64)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        target_skin = np.asarray(data["target_displacement_skin"], dtype=np.float64)
        weights = np.asarray(data["skin_vertex_weights"], dtype=np.float64)
        initial_u = np.asarray(data["initial_u"], dtype=np.float64)
        active_ids = np.asarray(data["active_ids"], dtype=np.int64)
    volume = pv.read(
        Path(__file__).resolve().parents[6]
        / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
    )
    tets = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)[:, 1:]
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    fixed = np.asarray(volume.point_data["FixedMask"], dtype=bool)
    prescribed = np.asarray(volume.point_data["FixedValue"], dtype=np.float64)
    area, stiffness = _stiffness(rest[skin_ids], triangles)
    if np.max(np.abs(np.asarray(stiffness.sum(axis=1)).ravel())) > 1e-10:
        raise AssertionError("independent stiffness does not preserve constants")

    checks: dict[str, object] = {
        "provenance": {
            "numerical_source_sha256_match": True,
            "numerical_source_count": len(comparison_sources),
            "fixture_receipts_match": True,
            "learning_rate": comparison_protocol["config"]["learning_rate"],
            "adam_eps": comparison_protocol["config"]["adam_eps"],
            "gradient_scale": comparison_calibration["gradient_scale"],
        },
        "branches": {},
    }
    first_q: dict[str, np.ndarray] = {}
    first_u: dict[str, np.ndarray] = {}
    metrics_engine = StudyMetrics()
    for kind in ("l2", "gradient"):
        folder = source / kind
        q0, u0, step0 = _load_state(folder / "step-0000.npz")
        q, u, step = _load_state(folder / "last.npz")
        if step != expected_steps:
            raise AssertionError(
                f"{kind} endpoint is step {step}, expected {expected_steps}"
            )
        if step0 != 0 or not np.array_equal(q0, np.zeros_like(q0)):
            raise AssertionError(f"{kind} lacks the neutral raw6 checkpoint")
        if np.max(np.abs(u0 - initial_u)) > 1e-14:
            raise AssertionError(f"{kind} neutral displacement differs from zero")
        first_q[kind], first_u[kind] = q0, u0
        residual = u[skin_ids] - target_skin
        direct = _direct_gradient_energy(residual, rest[skin_ids], triangles)
        sparse = float(
            sum(
                residual[:, component] @ stiffness @ residual[:, component]
                for component in range(3)
            )
            / area.sum()
        )
        sparse_direct_error = _close(sparse, direct, tolerance=1e-11)
        summary = json.loads((folder / "summary.json").read_text())
        if summary["status"] != "completed_budget_not_convergence_certified":
            raise AssertionError(f"{kind} completion status is {summary['status']!r}")
        reported = summary["last_metrics"]
        recomputed = _endpoint_metrics(
            q,
            u,
            rest=rest,
            skin_ids=skin_ids,
            target_skin=target_skin,
            weights=weights,
            tets=tets,
            dm_inv=dm_inv,
            active_ids=active_ids,
            fixed=fixed,
            prescribed=prescribed,
            gradient_energy=direct,
        )
        metric_errors = {
            name: _close(float(value), float(reported[name]))
            for name, value in recomputed.items()
        }
        surface_errors = {
            name: _close(float(value), float(reported[name]))
            for name, value in metrics_engine.evaluate_surface(u[skin_ids]).items()
        }
        checks["branches"][kind] = {
            "endpoint_step": step,
            "initial_displacement_inf": float(np.max(np.abs(u0 - initial_u))),
            "direct_surface_gradient_energy": direct,
            "sparse_stiffness_surface_gradient_energy": sparse,
            "sparse_direct_absolute_error": sparse_direct_error,
            "exported_metric_absolute_errors": metric_errors,
            "exported_surface_metric_absolute_errors": surface_errors,
        }

    zero_u_difference = float(np.max(np.abs(first_u["l2"] - first_u["gradient"])))
    if zero_u_difference > 1e-14 or not np.array_equal(
        first_q["l2"], first_q["gradient"]
    ):
        raise AssertionError("branches do not share the same neutral checkpoint")
    checks["neutral_checkpoint"] = {
        "q_equal": True,
        "u_l2_gradient_max_absolute_difference": zero_u_difference,
    }
    checks["passed"] = True
    cfg.output.write_text(json.dumps(checks, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    cherries.main(main)
