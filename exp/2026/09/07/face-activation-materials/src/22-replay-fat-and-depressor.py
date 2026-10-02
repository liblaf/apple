# ruff: noqa: EM101, EM102, TRY003
"""Bounded forward replays of saved FiberRegion activation from fresh rest."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import ROOT, FacePhysics, configure

from liblaf import cherries

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
DEPRESSOR_LABII_IDS = (162, 163)


class Config(cherries.BaseConfig):
    """Pinned inputs and output for the forward-only replay batch."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    fixture: Path = cherries.input("10-fixture")
    checkpoint: Path = cherries.input("21-fiber-region-B-v2/final.npz")
    output_dir: Path = cherries.output("22-replays", mkdir=True)
    include_neo_49: bool = True


@dataclass(frozen=True)
class Case:
    """One independent forward replay with bounded changes from baseline B."""

    name: str
    depressor_scale: float
    fat_model: str
    fat_nu: float


CASES = (
    Case("baseline-B", 1.0, "stable", 0.46),
    Case("depressor-labii-50pct", 0.5, "stable", 0.46),
    Case("depressor-labii-0pct", 0.0, "stable", 0.46),
    Case("fat-stable-nu049", 1.0, "stable", 0.49),
    Case("fat-neo-nu046", 1.0, "neo", 0.46),
    Case("fat-neo-nu049", 1.0, "neo", 0.49),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: (
                item.item() if isinstance(item, np.generic) else str(item)
            ),
        )
        + "\n",
        encoding="utf-8",
    )


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(
        (0.0, 0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999, 1.0)
    )
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    result = np.interp(probabilities, cumulative, sorted_values)
    return {
        f"q{probability:g}": float(value)
        for probability, value in zip(probabilities, result, strict=True)
    }


def metrics(
    physics: FacePhysics,
    u: np.ndarray,
    ainv: np.ndarray,
) -> dict[str, Any]:
    target = physics.target[physics.top]
    prediction = u[physics.top]
    residual = prediction - target
    detf = physics.detf(u)
    det_ainv = np.linalg.det(ainv)
    detg = detf.copy()
    detg[physics.ids] *= det_ainv
    projection = float(
        np.sum(physics.weights[:, None] * prediction * target) / physics.D**2
    )
    orthogonal = prediction - projection * target
    return {
        "target_rms_mm": 1000.0 * physics.D,
        "fit_rms_mm": 1000.0
        * math.sqrt(float(np.sum(physics.weights[:, None] * residual**2))),
        "motion_rms_mm": 1000.0
        * math.sqrt(float(np.sum(physics.weights[:, None] * prediction**2))),
        "target_projection_amplitude": projection,
        "projection_orthogonal_rms_over_target": math.sqrt(
            float(np.sum(physics.weights[:, None] * orthogonal**2))
        )
        / physics.D,
        "J": {
            "volume_weighted_quantiles": weighted_quantiles(detf, physics.volumes_all),
            "counts": {
                f"below_{threshold:g}": int(np.count_nonzero(detf < threshold))
                for threshold in (0.0, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9)
            },
        },
        "detG": {
            "volume_weighted_quantiles": weighted_quantiles(detg, physics.volumes_all),
            "nonpositive": int(np.count_nonzero(detg <= 0.0)),
        },
        "activation": {
            "eigen_min": float(np.linalg.eigvalsh(ainv).min()),
            "eigen_max": float(np.linalg.eigvalsh(ainv).max()),
            "det_min": float(det_ainv.min()),
            "det_max": float(det_ainv.max()),
        },
        "fixed_readback_max_abs_m": float(
            np.max(
                np.abs(u[np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)])
            )
        ),
    }


def case_config(case: Case) -> dict[str, Any]:
    return {
        "experiment_role": "forward counterfactual replay; not inverse optimization",
        "depressor_labii_saved_amplitude_scale": case.depressor_scale,
        "fat_model": case.fat_model,
        "fat_nu": case.fat_nu,
        "fat_factor": 1.0,
        "muscle_factor": 0.8,
        "muscle_nu": 0.46,
        "aponeurosis_nu": 0.35,
        "skin_factor": 0.12,
        "skin_nu": 0.46,
        "forward_rtol": 1e-6,
        "forward_atol": 1e-13,
        "forward_max_steps": 10000,
        "line_search_max_steps": 30,
        "line_search_failure_policy": "raise; do not commit exhausted trial",
        "seed": "exact zero displacement (fresh rest)",
    }


def validate_saved_ainv(ainv: np.ndarray, saved_ainv: np.ndarray) -> bool:
    if not np.array_equal(ainv, saved_ainv):
        max_abs = float(np.max(np.abs(ainv - saved_ainv)))
        raise ValueError(
            "saved activation tensor is not reproduced exactly "
            f"(max_abs={max_abs:.17g})"
        )
    return True


def validate_endpoint(physics: FacePhysics, u: np.ndarray) -> None:
    if not np.isfinite(u).all():
        raise FloatingPointError("forward displacement contains nonfinite values")
    detf = physics.detf(u)
    if not np.isfinite(detf).all():
        raise FloatingPointError("endpoint J contains nonfinite values")
    if np.any(detf <= 0.0):
        raise FloatingPointError(
            f"endpoint contains {np.count_nonzero(detf <= 0.0)} nonpositive J"
        )


def run_case(
    cfg: Config,
    case: Case,
    saved_q: np.ndarray,
    saved_ainv: np.ndarray,
    depressor_controls: np.ndarray,
) -> dict[str, Any]:
    output = cfg.output_dir / case.name
    output.mkdir(parents=True, exist_ok=False)
    spec = case_config(case)
    write_json(output / "case.json", spec)
    start = time.perf_counter()
    physics: FacePhysics | None = None
    try:
        physics = FacePhysics(
            cfg.fixture,
            skin_factor=0.12,
            fat_factor=1.0,
            muscle_factor=0.8,
            rtol=1e-6,
            atol=1e-13,
            adjoint_rtol=1e-7,
            soft_nu=0.46,
            fat_nu=case.fat_nu,
            fat_model=case.fat_model,
            skin_nu=0.46,
            target_name="Smile",
            target_scale=1.0,
            line_search_max_steps=30,
        )
        q = saved_q.copy()
        q[depressor_controls] *= case.depressor_scale
        q_tensor = torch.as_tensor(q)
        local_q = q_tensor[physics.region_t]
        ainv_tensor, _ = am.matrices(local_q, "F", physics.fibers)
        ainv = ainv_tensor.detach().cpu().numpy().copy()
        baseline_ainv_exact = None
        if case.depressor_scale == 1.0:
            baseline_ainv_exact = validate_saved_ainv(ainv, saved_ainv)
        u_tensor = physics.solve(am.packed(ainv_tensor), np.zeros_like(physics.points))
        u = u_tensor.detach().cpu().numpy().copy()
        validate_endpoint(physics, u)
        result_metrics = metrics(physics, u, ainv)
        physics.save_mesh(output / "final.vtu", u, ainv)
        np.savez_compressed(output / "final.npz", q=q, u=u, Ainv=ainv)
        receipt = {
            "status": "success",
            "case": spec,
            "materials_actual": physics.material_spec,
            "solver": physics.last_forward,
            "saved_Ainv_reproduced_exactly": baseline_ainv_exact,
            "metrics": result_metrics,
            "wall_s": time.perf_counter() - start,
            "outputs": {
                name: {
                    "sha256": sha256(output / name),
                    "bytes": (output / name).stat().st_size,
                }
                for name in ("final.vtu", "final.npz")
            },
        }
        write_json(output / "receipt.json", receipt)
    except Exception as error:  # each material case is an independent probe
        receipt = {
            "status": "failure",
            "case": spec,
            "materials_actual": physics.material_spec if physics is not None else None,
            "solver": (
                getattr(physics, "last_forward", None) if physics is not None else None
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "wall_s": time.perf_counter() - start,
        }
        write_json(output / "receipt.json", receipt)
        LOG.exception("Case %s failed", case.name)
        return receipt
    else:
        return receipt


def main(cfg: Config) -> None:
    configure()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {cfg.output_dir}")
    fixture_mesh = pv.read(cfg.fixture / "volume.vtu")
    if not isinstance(fixture_mesh, pv.UnstructuredGrid):
        raise TypeError("fixture volume is not an UnstructuredGrid")
    region_ids = np.asarray(
        fixture_mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
    )
    depressor_controls = np.flatnonzero(np.isin(region_ids, DEPRESSOR_LABII_IDS))
    if len(depressor_controls) != 2:
        raise ValueError("fixture must contain both Depressor labii regions")
    with np.load(cfg.checkpoint, allow_pickle=False) as saved:
        saved_q = np.asarray(saved["q"], dtype=np.float64).copy()
        saved_ainv = np.asarray(saved["Ainv"], dtype=np.float64).copy()
    if saved_q.shape != (len(region_ids), 1):
        raise ValueError("checkpoint is not a 35-control FiberRegion state")

    source_dir = cfg.output_dir / "sources"
    source_dir.mkdir()
    source_hashes = {}
    for source in (
        Path(__file__),
        HERE / "activation_models.py",
        HERE / "face_physics.py",
        HERE / "experiment_profile.py",
    ):
        shutil.copy2(source, source_dir / source.name)
        source_hashes[source.name] = sha256(source)
    inputs = {
        "checkpoint": {
            "path": str(cfg.checkpoint.resolve()),
            "sha256": sha256(cfg.checkpoint),
        },
        "fixture": {
            name: {
                "path": str((cfg.fixture / name).resolve()),
                "sha256": sha256(cfg.fixture / name),
            }
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
    }
    provenance = {
        "experiment_role": "forward counterfactual replay; not inverse optimization",
        "inputs": inputs,
        "sources": source_hashes,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "saved_controls": {
            "shape": list(saved_q.shape),
            "depressor_labii_muscle_ids": list(DEPRESSOR_LABII_IDS),
            "depressor_labii_control_ids": depressor_controls.tolist(),
            "q_sha256": hashlib.sha256(saved_q.tobytes()).hexdigest(),
            "Ainv_sha256": hashlib.sha256(saved_ainv.tobytes()).hexdigest(),
        },
    }
    write_json(cfg.output_dir / "provenance.json", provenance)

    cases = [
        case for case in CASES if cfg.include_neo_49 or case.name != "fat-neo-nu049"
    ]
    results = []
    for step, case in enumerate(cases):
        cherries.set_step(step)
        LOG.info("Starting independent case %s", case.name)
        receipt = run_case(cfg, case, saved_q, saved_ainv, depressor_controls)
        results.append(receipt)
        scalars = {"success": float(receipt["status"] == "success")}
        if receipt["status"] == "success":
            scalars.update(
                fit_rms_mm=receipt["metrics"]["fit_rms_mm"],
                motion_rms_mm=receipt["metrics"]["motion_rms_mm"],
                min_J=receipt["metrics"]["J"]["volume_weighted_quantiles"]["q0"],
            )
        cherries.log_metrics(
            {f"replay/{case.name}/{key}": value for key, value in scalars.items()}
        )
        LOG.info("Finished %s with status=%s", case.name, receipt["status"])

    summary = {
        "schema_version": 1,
        "experiment_role": "forward counterfactual replay; not inverse optimization",
        "config": cfg.model_dump(mode="json"),
        "provenance": provenance,
        "cases": results,
        "successful_cases": sum(case["status"] == "success" for case in results),
        "failed_cases": sum(case["status"] == "failure" for case in results),
    }
    write_json(cfg.output_dir / "summary.json", summary)
    files = [
        path
        for path in cfg.output_dir.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    ]
    write_json(
        cfg.output_dir / "artifact-manifest.json",
        {
            "files": {
                path.relative_to(cfg.output_dir).as_posix(): {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(files)
            }
        },
    )
    LOG.info(
        "Completed forward-only replay batch: %d success, %d failure",
        summary["successful_cases"],
        summary["failed_cases"],
    )
    if summary["failed_cases"]:
        raise RuntimeError(
            f"{summary['failed_cases']} independent forward replay case(s) failed"
        )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
