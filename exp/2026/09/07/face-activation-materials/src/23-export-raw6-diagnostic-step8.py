# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Export and verify the accepted Raw6 step-8 diagnostic endpoint on the CPU."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    source_dir: Path = ROOT / "data" / "23-raw6-fat049"
    fixture: Path = ROOT / "data" / "10-fixture"
    output_dir: Path = ROOT / "data" / "23-raw6-diagnostic-step8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    if encoded.shape[0] != mesh.n_cells or np.any(encoded[:, 0] != 4):
        raise ValueError("fixture must be an all-tetrahedron UnstructuredGrid")
    return encoded[:, 1:]


def triangles(mesh: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.shape[0] != mesh.n_cells or np.any(encoded[:, 0] != 3):
        raise ValueError("fixture skin must be an all-triangle PolyData")
    return encoded[:, 1:]


def compute_detf(
    rest: np.ndarray,
    current: np.ndarray,
    tets: np.ndarray,
    *,
    chunk_size: int = 50_000,
) -> np.ndarray:
    result = np.empty(len(tets), dtype=np.float64)
    for start in range(0, len(tets), chunk_size):
        stop = min(start + chunk_size, len(tets))
        ids = tets[start:stop]
        dm = np.transpose(rest[ids[:, 1:]] - rest[ids[:, :1]], (0, 2, 1))
        ds = np.transpose(current[ids[:, 1:]] - current[ids[:, :1]], (0, 2, 1))
        det_dm = np.linalg.det(dm)
        if np.any(det_dm <= 0.0):
            raise ValueError("fixture contains a nonpositive rest tetrahedron")
        result[start:stop] = np.linalg.det(ds @ np.linalg.inv(dm))
    return result


def surface_weights(
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
    top_mask: np.ndarray,
) -> np.ndarray:
    volume_ids = np.asarray(volume.point_data["GlobalPointId"], dtype=np.int64)
    if np.unique(volume_ids).size != volume.n_points:
        raise ValueError("volume GlobalPointId is not one-to-one")
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    skin_triangles = triangles(skin)
    triangle_global_ids = skin_ids[skin_triangles]
    order = np.argsort(volume_ids)
    locations = np.searchsorted(volume_ids[order], triangle_global_ids)
    if np.any(locations >= len(order)):
        raise ValueError("skin GlobalPointId is absent from volume")
    tri = order[locations]
    if not np.array_equal(volume_ids[tri], triangle_global_ids):
        raise ValueError("skin-to-volume GlobalPointId mapping failed")
    rest = np.asarray(volume.points, dtype=np.float64)
    xyz = rest[tri]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2.0
    )
    weights = np.zeros(volume.n_points, dtype=np.float64)
    np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
    selected = weights[top_mask]
    if np.any(selected < 0.0) or selected.sum() <= 0.0:
        raise ValueError("target surface weights are invalid")
    return selected / selected.sum()


def independent_metrics(
    volume: pv.UnstructuredGrid,
    skin: pv.PolyData,
    config: dict[str, Any],
    u: np.ndarray,
    ainv: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    rest = np.asarray(volume.points, dtype=np.float64)
    current = rest + u
    tets = tetrahedra(volume)
    detf = compute_detf(rest, current, tets)
    active = np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], dtype=bool))
    if ainv.shape != (len(active), 3, 3):
        raise ValueError("checkpoint Ainv does not match fixture active cells")
    det_ainv = np.linalg.det(ainv)
    eig_ainv = np.linalg.eigvalsh(ainv)
    detg = detf.copy()
    detg[active] *= det_ainv

    raw_target = np.asarray(
        volume.point_data[config["target_name"]], dtype=np.float64
    ) * float(config["target_scale"])
    top_mask = np.asarray(volume.point_data["IsFace"], dtype=bool) & np.isfinite(
        raw_target
    ).all(axis=1)
    target = np.zeros_like(rest)
    target[top_mask] = raw_target[top_mask]
    weights = surface_weights(volume, skin, top_mask)
    target_top = target[top_mask]
    pred = u[top_mask]
    scale = float(np.sqrt(np.sum(weights[:, None] * target_top**2)))
    fit = float(np.sum(weights[:, None] * (pred - target_top) ** 2) / scale**2)
    amplitude = float(np.sum(weights[:, None] * pred * target_top) / scale**2)
    residual = pred - amplitude * target_top
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    metrics = {
        "target_rms_m": scale,
        "target_rms_mm": 1000.0 * scale,
        "fit_objective_raw6": fit,
        "fit_rms_over_D": math.sqrt(fit),
        "fit_rms_mm": 1000.0 * scale * math.sqrt(fit),
        "motion_rms_mm": 1000.0 * math.sqrt(float(np.sum(weights[:, None] * pred**2))),
        "target_projection_amplitude": amplitude,
        "target_projection_residual_over_D": float(
            np.sqrt(np.sum(weights[:, None] * residual**2)) / scale
        ),
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
        "inverted_tets": int(np.count_nonzero(detf <= 0.0)),
        "detG_min": float(detg.min()),
        "activation_eigen_min": float(eig_ainv.min()),
        "activation_eigen_max": float(eig_ainv.max()),
        "activation_det_min": float(det_ainv.min()),
        "activation_det_max": float(det_ainv.max()),
        "fixed_readback_max_abs_m": float(np.abs(u[fixed]).max()),
        "target_vertices": int(np.count_nonzero(top_mask)),
        "surface_weight_sum": float(weights.sum()),
    }
    return metrics, detf, target, active


def material_spec(config: dict[str, Any]) -> dict[str, Any]:
    def lame(young: float, nu: float, model: str) -> tuple[float, float, float]:
        mu = young / (2.0 * (1.0 + nu))
        classical = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        code = classical + mu if model == "stable" else classical
        return mu, classical, code

    fat_young = 0.003 * float(config["fat_factor"])
    fat_nu = float(config["fat_nu"] or config["soft_nu"])
    fat_mu, fat_classical, fat_code = lame(fat_young, fat_nu, config["fat_model"])
    muscle_young = 0.03 * float(config["muscle_factor"])
    muscle_nu = float(config["soft_nu"])
    muscle_mu, muscle_classical, muscle_code = lame(muscle_young, muscle_nu, "stable")
    apo_mu, apo_classical, apo_code = lame(0.1, 0.35, "stable")
    return {
        "fat_E_MPa": fat_young,
        "fat_model": config["fat_model"],
        "fat_nu": fat_nu,
        "fat_mu_code_MPa": fat_mu,
        "fat_lambda_classical_MPa": fat_classical,
        "fat_lambda_code_MPa": fat_code,
        "muscle_E_MPa": muscle_young,
        "muscle_model": "stable-active",
        "muscle_nu": muscle_nu,
        "muscle_mu_code_MPa": muscle_mu,
        "muscle_lambda_classical_MPa": muscle_classical,
        "muscle_lambda_code_MPa": muscle_code,
        "aponeurosis_E_MPa": 0.1,
        "aponeurosis_model": "stable",
        "aponeurosis_nu": 0.35,
        "aponeurosis_mu_code_MPa": apo_mu,
        "aponeurosis_lambda_classical_MPa": apo_classical,
        "aponeurosis_lambda_code_MPa": apo_code,
        "skin_E_MPa": 0.2 * float(config["skin_factor"]),
        "skin_nu": float(config["skin_nu"]),
        "skin_thickness_m": 0.001,
        "skin_plane_stress": True,
        "skin_prestrain": 0.0,
        "contact_enabled": False,
        "volume_lame_convention": (
            "Stable uses lambda_code=lambda_classical+mu; logarithmic Neo uses "
            "lambda_code=lambda_classical; reported E,nu are actual infinitesimal "
            "constants"
        ),
    }


def compare_metrics(
    computed: dict[str, Any], trace_row: dict[str, str]
) -> dict[str, dict[str, Any]]:
    keys = (
        "fit_rms_over_D",
        "fit_rms_mm",
        "motion_rms_mm",
        "target_projection_amplitude",
        "target_projection_residual_over_D",
        "detF_min",
        "detF_max",
        "inverted_tets",
        "detG_min",
        "activation_eigen_min",
        "activation_eigen_max",
        "activation_det_min",
        "activation_det_max",
    )
    comparison = {}
    for key in keys:
        observed = computed[key]
        saved = int(trace_row[key]) if key == "inverted_tets" else float(trace_row[key])
        if key == "inverted_tets":
            matched = observed == saved
            difference = observed - saved
        else:
            matched = bool(np.isclose(observed, saved, rtol=5e-12, atol=1e-12))
            difference = observed - saved
        if not matched:
            raise ValueError(f"independent metric mismatch for {key}")
        comparison[key] = {
            "computed": observed,
            "trace": saved,
            "difference": difference,
            "matched": True,
        }
    return comparison


def save_endpoint(
    path: Path,
    fixture: pv.UnstructuredGrid,
    u: np.ndarray,
    ainv: np.ndarray,
    detf: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, Any]:
    rest = np.asarray(fixture.points, dtype=np.float64)
    current = rest + u
    mesh = pv.UnstructuredGrid(fixture.cells, fixture.celltypes, current)
    point_keys = (
        "IsFace",
        "IsFixed",
        "IsLip",
        "FixedMask",
        "FixedValue",
        "CutBoundary",
        "ArtificialCutIncident",
        "CutBoundaryAddedFixed",
        "GlobalPointId",
    )
    cell_keys = (
        "MuscleId",
        "MuscleFraction",
        "FatFraction",
        "AponeurosisFraction",
        "ActivationMask",
        "ActivationControlId",
        "ActivationFiber",
    )
    for key in point_keys:
        if key in fixture.point_data:
            mesh.point_data[key] = fixture.point_data[key]
    for key in cell_keys:
        mesh.cell_data[key] = fixture.cell_data[key]
    mesh.point_data["RestPosition"] = rest
    mesh.point_data["Displacement"] = u
    mesh.point_data["TargetDisplacement"] = target
    mesh.cell_data["DetF"] = detf
    full_ainv = np.broadcast_to(np.eye(3), (fixture.n_cells, 3, 3)).copy()
    full_ainv[active] = ainv
    mesh.cell_data["ActivationInverseMatrix"] = full_ainv.reshape(-1, 9)
    det_ainv = np.ones(fixture.n_cells, dtype=np.float64)
    det_ainv[active] = np.linalg.det(ainv)
    mesh.cell_data["DetAinv"] = det_ainv
    mesh.save(path)

    loaded = pv.read(path)
    if not isinstance(loaded, pv.UnstructuredGrid):
        raise TypeError("exported endpoint did not read as UnstructuredGrid")
    if (
        loaded.n_points != fixture.n_points
        or loaded.n_cells != fixture.n_cells
        or not np.array_equal(loaded.cells, fixture.cells)
        or not np.array_equal(loaded.celltypes, fixture.celltypes)
        or not np.array_equal(np.asarray(loaded.points), current)
    ):
        raise ValueError(
            "exported endpoint topology or coordinates changed on round trip"
        )
    for key in mesh.point_data:
        if not np.array_equal(
            np.asarray(loaded.point_data[key]), np.asarray(mesh.point_data[key])
        ):
            raise ValueError(f"point array changed on round trip: {key}")
    for key in mesh.cell_data:
        if not np.array_equal(
            np.asarray(loaded.cell_data[key]), np.asarray(mesh.cell_data[key])
        ):
            raise ValueError(f"cell array changed on round trip: {key}")
    return {
        "points": loaded.n_points,
        "cells": loaded.n_cells,
        "point_arrays": sorted(loaded.point_data.keys()),
        "cell_arrays": sorted(loaded.cell_data.keys()),
        "round_trip_exact": True,
    }


def manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {"files": files}


def main(cfg: Config) -> None:
    source = cfg.source_dir.resolve()
    fixture_dir = cfg.fixture.resolve()
    output = cfg.output_dir.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic export: {output}")

    checkpoint = source / "diagnostic-stop-checkpoint.npz"
    stop_path = source / "diagnostic-stop.json"
    trace_path = source / "trace.csv"
    trials_path = source / "trials.json"
    config_path = source / "config.json"
    provenance_path = source / "provenance.json"
    fixture_path = fixture_dir / "volume.vtu"
    skin_path = fixture_dir / "skin.vtp"
    fixture_summary_path = fixture_dir / "summary.json"
    source_paths = (
        checkpoint,
        stop_path,
        trace_path,
        trials_path,
        config_path,
        provenance_path,
        fixture_path,
        skin_path,
        fixture_summary_path,
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        cherries.log_input(path)

    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    with trace_path.open(newline="", encoding="utf-8") as stream:
        trace = list(csv.DictReader(stream))
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != stop["checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint hash differs from stop receipt")
    if stop["signal"] != "SIGINT" or config["method"] != "Raw6":
        raise ValueError("source is not the declared interrupted Raw6 run")
    if stop["last_persisted_trace_row"] != trace[-1]:
        raise ValueError("stop receipt does not reproduce the last trace row exactly")

    with np.load(checkpoint, allow_pickle=False) as saved:
        expected_keys = {"q", "u", "Ainv", "step"}
        if set(saved.files) != expected_keys:
            raise ValueError("diagnostic checkpoint schema changed")
        q = np.asarray(saved["q"]).copy()
        u = np.asarray(saved["u"]).copy()
        ainv = np.asarray(saved["Ainv"]).copy()
        checkpoint_step = int(np.asarray(saved["step"]).item())
    if checkpoint_step != int(trace[-1]["step"]):
        raise ValueError("checkpoint step does not match the last trace row")
    if q.shape != (120_020, 6) or u.shape != (228_660, 3):
        raise ValueError("Raw6 checkpoint tensor shape changed")
    if ainv.shape != (120_020, 3, 3):
        raise ValueError("Raw6 checkpoint Ainv shape changed")

    accepted_trial_index = len(trials) - 1
    accepted = trials[accepted_trial_index]
    if (
        accepted_trial_index != 12
        or accepted["step"] != checkpoint_step - 1
        or accepted["trial"] != 3
        or not accepted["admissible"]
        or not accepted["armijo"]
        or not accepted["forward"]["success"]
        or accepted["objective"] != float(trace[-1]["objective"])
        or accepted["forward"]["steps"] != int(trace[-1]["forward_steps"])
        or accepted["forward"]["grad_norm"] != float(trace[-1]["forward_grad_norm"])
    ):
        raise ValueError("last accepted trial does not explain checkpoint step 8")

    for name, expected in provenance["inputs"].items():
        if sha256(fixture_dir / name) != expected:
            raise ValueError(f"fixture provenance hash mismatch: {name}")
    original_sources = source / "sources"
    for name in ("20-run-face-inverse.py", "face_physics.py"):
        if sha256(original_sources / name) != provenance["sources"][name]:
            raise ValueError(f"source provenance hash mismatch: {name}")

    volume = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(volume, pv.UnstructuredGrid) or not isinstance(skin, pv.PolyData):
        raise TypeError("fixture volume/skin types are invalid")
    computed, detf, target, active = independent_metrics(volume, skin, config, u, ainv)
    comparison = compare_metrics(computed, trace[-1])
    if not np.isclose(
        computed["fit_objective_raw6"],
        float(trace[-1]["objective"]),
        rtol=5e-12,
        atol=1e-12,
    ):
        raise ValueError("independent Raw6 fit does not equal the saved objective")
    if computed["fixed_readback_max_abs_m"] != 0.0:
        raise ValueError("fixed degrees of freedom moved in checkpoint")
    if float(trace[-1]["kkt"]) <= float(config["kkt_tol"]):
        raise ValueError("diagnostic endpoint unexpectedly satisfies KKT tolerance")

    temporary.mkdir(parents=True)
    evidence_dir = temporary / "source-run"
    evidence_dir.mkdir()
    for path in (stop_path, trace_path, trials_path, config_path, provenance_path):
        shutil.copy2(path, evidence_dir / path.name)
    output_sources = temporary / "sources"
    output_sources.mkdir()
    for name in ("20-run-face-inverse.py", "face_physics.py"):
        shutil.copy2(original_sources / name, output_sources / name)
    shutil.copy2(Path(__file__), output_sources / Path(__file__).name)

    final_npz = temporary / "final.npz"
    shutil.copyfile(checkpoint, final_npz)
    if sha256(final_npz) != checkpoint_hash:
        raise AssertionError("final.npz is not an exact checkpoint byte copy")
    endpoint_receipt = save_endpoint(
        temporary / "final.vtu", volume, u, ainv, detf, target, active
    )
    input_hashes = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    }
    summary = {
        "schema_version": 1,
        "status": "diagnostic_stop_not_converged",
        "optimization_complete": False,
        "accepted_equilibrium": True,
        "physics_resolve_performed": False,
        "fresh_rest_branch_check_performed": False,
        "description": (
            "Exact export of the last accepted equilibrium from an adaptively "
            "interrupted Raw6 run; this is not a completed optimization."
        ),
        "checkpoint": {
            "step": checkpoint_step,
            "source": checkpoint.relative_to(ROOT).as_posix(),
            "source_sha256": checkpoint_hash,
            "final_npz_is_byte_copy": True,
            "final_npz_sha256": sha256(final_npz),
            "schema": {
                "q": {"shape": list(q.shape), "dtype": str(q.dtype)},
                "u": {"shape": list(u.shape), "dtype": str(u.dtype)},
                "Ainv": {"shape": list(ainv.shape), "dtype": str(ainv.dtype)},
                "step": {"value": checkpoint_step},
            },
        },
        "stop": {
            "reason": stop["reason"],
            "signal": stop["signal"],
            "requested_steps": stop["requested_steps"],
            "kkt": float(trace[-1]["kkt"]),
            "kkt_tolerance": float(config["kkt_tol"]),
            "kkt_criterion_met": False,
        },
        "source_config": config,
        "materials": material_spec(config),
        "evidence": {
            "trace": {
                "path": trace_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(trace_path),
                "data_row_index_zero_based": len(trace) - 1,
                "physical_csv_line_one_based": len(trace) + 1,
                "row": trace[-1],
                "matches_stop_receipt_exactly": True,
            },
            "accepted_trial": {
                "path": trials_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(trials_path),
                "json_index_zero_based": accepted_trial_index,
                "receipt": accepted,
                "relation_to_checkpoint": (
                    "outer step 7 trial 3 was accepted, then persisted as trace and "
                    "checkpoint step 8"
                ),
            },
            "input_hashes": input_hashes,
            "source_hashes_match_original_provenance": True,
            "fixture_hashes_match_original_provenance": True,
        },
        "independent_recomputation": {
            "implementation": (
                "NumPy/PyVista CPU reconstruction from fixture rest geometry, saved "
                "u/Ainv, fixture target, and rest triangle-area weights; no FacePhysics, "
                "torch, Warp, forward solve, or GPU"
            ),
            "metrics": computed,
            "trace_comparison": comparison,
            "all_compared_metrics_match": True,
        },
        "final_vtu": endpoint_receipt,
    }
    write_json(temporary / "summary.json", summary)
    write_json(temporary / "artifact-manifest.json", manifest(temporary))
    temporary.rename(output)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "diagnostic/step": checkpoint_step,
            "diagnostic/fit_rms_mm": computed["fit_rms_mm"],
            "diagnostic/min_detF": computed["detF_min"],
            "diagnostic/min_detG": computed["detG_min"],
            "diagnostic/kkt": float(trace[-1]["kkt"]),
            "diagnostic/physics_resolves": 0,
        }
    )
    logger.info("Wrote verified diagnostic endpoint export to %s", output)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
