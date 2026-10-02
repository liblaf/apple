"""Independently verify saved dominant-activation forward-ablation states."""

# ruff: noqa: EM102, PLR0915, PT018, TRY003

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
FORWARD = GROUP / "data/10-forward"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
BASELINE = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/step-0200.npz"
)
BASELINE_SHA256 = "21b7e546f04c5566629a1d3634659ec0c5738df215699ba23e24eb7ac856abd7"
STAGES = (
    ("baseline-replay", 0.0),
    ("removal-025", 0.25),
    ("removal-050", 0.50),
    ("removal-075", 0.75),
    ("dominant-only", 1.0),
)
CORE_METRICS = (
    "fit_rms_mm",
    "motion_rms_mm",
    "area_weighted_fit_rms_mm",
    "area_weighted_motion_rms_mm",
    "area_weighted_change_from_saved_rms_mm",
    "detF_min",
    "inverted_all_cells",
    "inverted_active_cells",
    "inverted_pure_muscle_cells",
    "active_volume_weighted_rms_detF_minus_one",
    "fixed_max_error_m",
)
EXCLUDED_HELPER_METRICS = {
    "z_update_frobenius_rms": (
        "The forward script did not pass previous_z to StudyMetrics.evaluate, so "
        "this helper field is self-relative and always zero. Use the independently "
        "verified baseline-relative Z-change fields instead."
    )
}


class Config(cherries.BaseConfig):
    output_dir: Path = cherries.output("30-verification", mkdir=True)


def record(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def unpack(q: np.ndarray) -> np.ndarray:
    x, y, z, xy, yz, xz = q.T
    return np.stack((x, xy, xz, xy, y, yz, xz, yz, z), axis=-1).reshape(
        -1, 3, 3
    ) + np.eye(3)


def detf(rest: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    ds = np.transpose((rest + u)[tets[:, 1:]] - (rest + u)[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ np.linalg.inv(dm))


def close(actual: float, expected: float, label: str) -> float:
    error = abs(actual - expected)
    tolerance = 5e-12 * max(1.0, abs(actual), abs(expected))
    if error > tolerance:
        raise AssertionError(
            f"{label}: {actual!r} != {expected!r}; error {error} > {tolerance}"
        )
    return error


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output must be empty: {out}"
    assert record(BASELINE)["sha256"] == BASELINE_SHA256

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
    pure_muscle = (muscle == 1.0) & (fat == 0.0) & (aponeurosis == 0.0)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    active_volume = volume[active_ids] * muscle[active_ids]
    is_face = np.asarray(mesh.point_data["IsFace"], dtype=bool)
    target = np.asarray(mesh.point_data["Smile"], dtype=np.float64)
    top = np.flatnonzero(is_face & np.isfinite(target).all(axis=1))
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    triangles = skin_ids[np.asarray(skin.faces).reshape(-1, 4)[:, 1:]]
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
    fixed = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(mesh.point_data["FixedValue"], dtype=np.float64)
    assert len(rest) == 228_660 and len(tets) == 1_146_517
    assert len(active_ids) == 288_235 and len(top) == 15_302
    assert int(fixed.sum()) == 81_108
    assert np.all(fixed_value[fixed] == 0.0)

    with np.load(BASELINE, allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert int(saved["step"]) == 200
        assert np.array_equal(saved["rest_points"], rest)
        assert np.array_equal(saved["active_ids"], active_ids)
        q0 = np.asarray(saved["q"], dtype=np.float64)
        b0 = np.asarray(saved["Ainv"], dtype=np.float64)
        u0 = np.asarray(saved["u"], dtype=np.float64)
    assert np.max(np.abs(unpack(q0) - b0)) < 1e-14
    z0 = b0 @ b0.swapaxes(-1, -2) - np.eye(3)
    eigenvalues0, eigenvectors0 = np.linalg.eigh(z0)
    top0 = eigenvalues0[:, -1]
    strength = np.maximum(top0, 0.0)
    axis = eigenvectors0[:, :, -1]
    projector = axis[:, :, None] * axis[:, None, :]
    z1 = strength[:, None, None] * projector
    b1 = np.eye(3) + (np.sqrt(1.0 + strength) - 1.0)[:, None, None] * projector
    spectral_scale = np.maximum(np.max(np.abs(eigenvalues0), axis=1), 1.0)
    spectral_tolerance = 64.0 * np.finfo(np.float64).eps * spectral_scale
    gap = eigenvalues0[:, -1] - eigenvalues0[:, -2]
    positive = top0 > spectral_tolerance
    weak_positive = (top0 > 0.0) & ~positive
    near_repeated = gap <= 1e-6 * np.maximum(1.0, np.abs(top0))
    baseline_b_values = np.linalg.eigvalsh(b0)

    diagnostic = StudyMetrics(FIXTURE)
    summary_path = FORWARD / "summary.json"
    protocol_path = FORWARD / "protocol.json"
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    assert summary["status"] == "complete"
    assert len(summary["stages"]) == len(STAGES)
    assert protocol["baseline"]["sha256"] == BASELINE_SHA256
    assert protocol["materials"]["muscle_model"] == "stable-active-physical-volume"
    assert protocol["materials"]["skin_E_MPa"] == 0.0
    assert protocol["materials"]["contact_enabled"] is False
    executed_sources = {}
    for name in (
        "__main__",
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
    assert (
        protocol["sources"]["volume_preserving_active"]["sha256"]
        == "aa31c2d9bd8f5abde2471b28c86881829496fcbca77328883523265cebe7f2c9"
    )

    rows = []
    for (name, t), reported in zip(STAGES, summary["stages"], strict=True):
        assert reported["name"] == name
        assert reported["removal_fraction"] == t
        assert reported["forward"]["success"] is True
        checkpoint = FORWARD / f"{name}.npz"
        assert reported["checkpoint"]["sha256"] == record(checkpoint)["sha256"]
        with np.load(checkpoint, allow_pickle=False) as state:
            assert bool(state["solver_valid"])
            assert bool(state["physical_volume_energy"])
            assert float(state["removal_fraction"]) == t
            assert np.array_equal(state["rest_points"], rest)
            assert np.array_equal(state["active_ids"], active_ids)
            u = np.asarray(state["u"], dtype=np.float64)
            b = np.asarray(state["B"], dtype=np.float64)
            z = np.asarray(state["Z"], dtype=np.float64)
            q = np.asarray(state["q"], dtype=np.float64)
        expected_z = (1.0 - t) * z0 + t * z1
        if t == 0.0:
            expected_b = b0
        elif t == 1.0:
            expected_b = b1
        else:
            values, vectors = np.linalg.eigh(np.eye(3) + expected_z)
            expected_b = (vectors * np.sqrt(values)[:, None, :]) @ vectors.swapaxes(
                -1, -2
            )
        z_expected_error = float(np.max(np.abs(z - expected_z)))
        b_expected_error = float(np.max(np.abs(b - expected_b)))
        b_z_error = float(np.max(np.abs(b @ b.swapaxes(-1, -2) - np.eye(3) - z)))
        q_b_error = float(np.max(np.abs(unpack(q) - b)))
        assert z_expected_error < 2e-13
        assert b_expected_error < 2e-12
        assert b_z_error < 2e-12
        assert q_b_error < 2e-14

        # The retained dyad is constant; all omitted tensor content scales by 1-t.
        retained = strength[:, None, None] * projector
        omitted_scaling_error = float(
            np.max(np.abs((z - retained) - (1.0 - t) * (z0 - retained)))
        )
        stage_eigenvalues = np.linalg.eigvalsh(z)
        dominant_strength_error = float(
            np.max(np.abs(stage_eigenvalues[positive, -1] - strength[positive]))
        )
        assert omitted_scaling_error < 3e-13
        assert dominant_strength_error < 3e-12

        pred = u[top]
        target_top = target[top]
        error = pred - target_top
        determinant = detf(rest, tets, u)
        independent = {
            "fit_rms_mm": float(1000.0 * np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "motion_rms_mm": float(1000.0 * np.sqrt(np.mean(np.sum(pred**2, axis=1)))),
            "area_weighted_fit_rms_mm": float(
                1000.0 * np.sqrt(np.sum(area * np.sum(error**2, axis=1)))
            ),
            "area_weighted_motion_rms_mm": float(
                1000.0 * np.sqrt(np.sum(area * np.sum(pred**2, axis=1)))
            ),
            "area_weighted_change_from_saved_rms_mm": float(
                1000.0 * np.sqrt(np.sum(area * np.sum((pred - u0[top]) ** 2, axis=1)))
            ),
            "detF_min": float(determinant.min()),
            "detF_max": float(determinant.max()),
            "inverted_all_cells": int(np.sum(determinant <= 0.0)),
            "inverted_active_cells": int(np.sum(determinant[active_ids] <= 0.0)),
            "inverted_pure_muscle_cells": int(
                np.sum((determinant <= 0.0) & pure_muscle)
            ),
            "active_volume_weighted_rms_detF_minus_one": float(
                np.sqrt(
                    np.average(
                        (determinant[active_ids] - 1.0) ** 2,
                        weights=active_volume,
                    )
                )
            ),
            "fixed_max_error_m": float(np.max(np.abs(u[fixed] - fixed_value[fixed]))),
            "z_change_from_baseline_frobenius_rms": float(
                np.sqrt(np.mean(np.sum((z - z0) ** 2, axis=(1, 2))))
            ),
            "z_change_from_baseline_volume_weighted_frobenius_rms": float(
                np.sqrt(
                    np.average(
                        np.sum((z - z0) ** 2, axis=(1, 2)),
                        weights=active_volume,
                    )
                )
            ),
        }
        assert independent["fixed_max_error_m"] < 1e-14
        core_errors = {
            key: close(
                float(reported["metrics"][key]),
                float(independent[key]),
                f"{name}/{key}",
            )
            for key in CORE_METRICS
        }
        helper = diagnostic.evaluate(u, z, previous_u=u0)
        helper_errors = {}
        for key, value in helper.items():
            if key in EXCLUDED_HELPER_METRICS:
                continue
            helper_errors[key] = close(
                float(reported["metrics"][key]), float(value), f"{name}/{key}"
            )
        rows.append(
            {
                "name": name,
                "removal_fraction": t,
                "checkpoint": record(checkpoint),
                "projection": {
                    "Z_expected_max_abs_error": z_expected_error,
                    "B_expected_max_abs_error": b_expected_error,
                    "B_Bt_minus_I_vs_Z_max_abs_error": b_z_error,
                    "packed_q_vs_B_max_abs_error": q_b_error,
                    "omitted_mode_scaling_max_abs_error": omitted_scaling_error,
                    "retained_dominant_strength_max_abs_error_material_positive_cells": dominant_strength_error,
                },
                "metrics": independent,
                "reported_core_metric_max_abs_error": max(core_errors.values()),
                "reported_helper_metric_max_abs_error_excluding_declared_fields": max(
                    helper_errors.values(), default=0.0
                ),
                "solver": reported["forward"],
            }
        )

    replay = rows[0]["metrics"]
    saved_metrics = summary["saved_baseline_metrics"]
    replay_fit_error = abs(
        replay["area_weighted_fit_rms_mm"]
        - float(saved_metrics["area_weighted_fit_rms_mm"])
    )
    replay_motion_error = abs(
        replay["area_weighted_motion_rms_mm"]
        - float(saved_metrics["area_weighted_motion_rms_mm"])
    )
    replay_checks = {
        "area_weighted_shape_change_mm": replay[
            "area_weighted_change_from_saved_rms_mm"
        ],
        "saved_vs_replay_area_fit_abs_error_mm": replay_fit_error,
        "saved_vs_replay_area_motion_abs_error_mm": replay_motion_error,
        "reverse_triangle_bound_mm": replay["area_weighted_change_from_saved_rms_mm"]
        + 1e-12,
    }
    assert replay_checks["area_weighted_shape_change_mm"] < 0.01
    assert replay_fit_error <= replay_checks["reverse_triangle_bound_mm"]
    assert replay_motion_error <= replay_checks["reverse_triangle_bound_mm"]

    receipt = {
        "status": "passed",
        "scope": "Read-only verification of saved forward states; no mechanics solve or optimization",
        "inputs": {
            "baseline": record(BASELINE),
            "forward_summary": record(summary_path),
            "forward_protocol": record(protocol_path),
            "executed_sources": executed_sources,
            "fixture": {
                name: record(FIXTURE / name)
                for name in ("volume.vtu", "skin.vtp", "summary.json")
            },
        },
        "fixture_contract": {
            "points": len(rest),
            "tetrahedra": len(tets),
            "active_cells": len(active_ids),
            "target_vertices": len(top),
            "fixed_vertices": int(np.any(fixed, axis=1).sum()),
            "fixed_coordinate_dofs": int(fixed.sum()),
            "pure_muscle_rule": "MuscleFraction == 1 and FatFraction == 0 and AponeurosisFraction == 0",
            "pure_muscle_cells": int(pure_muscle.sum()),
        },
        "source_spectrum": {
            "B_nonpositive_minimum_eigenvalue_cells": int(
                np.sum(baseline_b_values[:, 0] <= 0.0)
            ),
            "no_strictly_positive_top_Z_mode_cells": int(np.sum(top0 <= 0.0)),
            "weak_positive_top_Z_mode_cells_within_roundoff_tolerance": int(
                np.sum(weak_positive)
            ),
            "material_positive_top_Z_mode_cells": int(np.sum(positive)),
            "near_repeated_top_Z_modes_cells": int(np.sum(near_repeated)),
            "positive_mode_tolerance": "64*eps*max(||Z||_2, 1) per cell",
            "near_repeated_gap_rule": "lambda_max-lambda_2 <= 1e-6*max(1, abs(lambda_max))",
        },
        "projection_contract": {
            "definition": "Z1=max(lambda_max(Z0),0) n n^T; B1 is the positive square root of I+Z1",
            "continuation": "The retained dyad stays fixed and all omitted Z content is multiplied by 1-t",
            "mechanics": "For the corrected physical-J law, B enters equilibrium through B B^T; replacing its sign gauge by the positive square root is exact",
        },
        "baseline_replay": replay_checks,
        "excluded_forward_helper_metrics": EXCLUDED_HELPER_METRICS,
        "metric_limitations": [
            "Fit and motion are endpoint outcome metrics; this run performs no inverse refit.",
            "The continuation stages are solver aids and sensitivity context, not separately optimized models.",
            "Z is an effective activation tensor; its dominant eigenvector is not an anatomical fiber or observed tissue strain.",
            "The per-cell omitted fraction and global norm ratio are different weighting summaries and must retain distinct names.",
        ],
        "stages": rows,
    }
    write_json(out / "summary.json", receipt)
    cherries.log_metrics(
        {
            "verified_stages": len(rows),
            "final_area_fit_rms_mm": rows[-1]["metrics"]["area_weighted_fit_rms_mm"],
            "final_area_motion_rms_mm": rows[-1]["metrics"][
                "area_weighted_motion_rms_mm"
            ],
            "final_inverted_all_cells": rows[-1]["metrics"]["inverted_all_cells"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
