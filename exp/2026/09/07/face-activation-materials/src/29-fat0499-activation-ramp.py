# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Check the fat-nu=.499 FiberModes equilibrium by activation continuation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure

from liblaf import cherries

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = logging.getLogger(__name__)
INCREMENTS = 10


class Config(cherries.BaseConfig):
    """Pinned inputs and output for the activation-continuation check."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = cherries.input("10-fixture")
    controls: Path = cherries.input("12-controls.npz")
    checkpoint: Path = cherries.input("24-fiber-modes-fat049/final.npz")
    warm_reference: Path = cherries.input("28-fiber-fat0499-replay/final.npz")
    output_dir: Path = cherries.output("29-fat0499-ramp", mkdir=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stage_metrics(
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
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
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
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
        "inverted_tets": int(np.count_nonzero(detf <= 0.0)),
        "detF_below_0.5": int(np.count_nonzero(detf < 0.5)),
        "detF_below_0.8": int(np.count_nonzero(detf < 0.8)),
        "detG_min": float(detg.min()),
        "detG_nonpositive": int(np.count_nonzero(detg <= 0.0)),
        "activation_eigen_min": float(np.linalg.eigvalsh(ainv).min()),
        "activation_eigen_max": float(np.linalg.eigvalsh(ainv).max()),
        "activation_det_min": float(det_ainv.min()),
        "activation_det_max": float(det_ainv.max()),
        "fixed_readback_max_abs_m": float(np.abs(u[fixed]).max()),
    }


def validate_endpoint(
    physics: FacePhysics,
    u: np.ndarray,
    metrics: dict[str, Any],
    *,
    require_solver_success: bool,
) -> None:
    if not np.isfinite(u).all():
        raise FloatingPointError("forward displacement contains nonfinite values")
    if metrics["inverted_tets"] or metrics["detG_nonpositive"]:
        raise FloatingPointError("stage contains a nonpositive deformation determinant")
    if metrics["activation_eigen_min"] <= 0.0:
        raise FloatingPointError("stage activation tensor is not positive definite")
    if metrics["fixed_readback_max_abs_m"] != 0.0:
        raise FloatingPointError("fixed degrees of freedom moved")
    if require_solver_success and not physics.last_forward["success"]:
        raise RuntimeError("strict forward solve did not report success")


def validate_stage_files(
    npz_path: Path,
    vtu_path: Path,
    q: np.ndarray,
    u: np.ndarray,
    ainv: np.ndarray,
) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as saved:
        if set(saved.files) not in (
            {"q", "u", "Ainv", "stage", "alpha"},
            {"q", "u", "Ainv"},
        ):
            raise ValueError("saved stage NPZ schema changed")
        if (
            not np.array_equal(saved["q"], q)
            or not np.array_equal(saved["u"], u)
            or not np.array_equal(saved["Ainv"], ainv)
        ):
            raise ValueError("saved stage NPZ arrays changed on readback")
    mesh = pv.read(vtu_path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("saved stage VTU is not an UnstructuredGrid")
    expected_points = np.asarray(mesh.point_data["RestPosition"]) + u
    if not np.array_equal(np.asarray(mesh.points), expected_points):
        raise ValueError("saved stage VTU points changed on readback")
    if not np.array_equal(np.asarray(mesh.point_data["Displacement"]), u):
        raise ValueError("saved stage VTU displacement changed on readback")
    full_ainv = np.asarray(mesh.cell_data["ActivationInverseMatrix"]).reshape(-1, 3, 3)
    active = np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], dtype=bool))
    if not np.array_equal(full_ainv[active], ainv):
        raise ValueError("saved stage VTU Ainv changed on readback")
    return {
        "npz_roundtrip_exact": True,
        "vtu_points_displacement_Ainv_roundtrip_exact": True,
        "npz": {"bytes": npz_path.stat().st_size, "sha256": sha256(npz_path)},
        "vtu": {"bytes": vtu_path.stat().st_size, "sha256": sha256(vtu_path)},
    }


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def artifact_manifest(output: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact-manifest.json":
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return {"files": files}


def displacement_comparison(
    physics: FacePhysics,
    ramp: np.ndarray,
    warm: np.ndarray,
) -> dict[str, Any]:
    delta = ramp - warm
    point_norm = np.linalg.norm(delta, axis=1)
    weighted = math.sqrt(
        float(np.sum(physics.weights[:, None] * delta[physics.top] ** 2))
    )
    return {
        "array_equal": bool(np.array_equal(ramp, warm)),
        "ramp_u_sha256": array_sha256(ramp),
        "warm_reference_u_sha256": array_sha256(warm),
        "max_abs_coordinate_mm": 1000.0 * float(np.abs(delta).max()),
        "max_point_norm_mm": 1000.0 * float(point_norm.max()),
        "all_points_vector_rms_mm": 1000.0
        * math.sqrt(float(np.mean(np.sum(delta**2, axis=1)))),
        "target_surface_weighted_rms_mm": 1000.0 * weighted,
        "target_surface_weighted_rms_over_D": weighted / physics.D,
        "interpretation": (
            "observational comparison between two converged seeds at identical q/Ainv; "
            "bitwise u equality is not required"
        ),
    }


def main(cfg: Config) -> None:
    configure()
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")

    reference_dir = cfg.warm_reference.resolve().parent
    reference_paths = (
        reference_dir / "config.json",
        reference_dir / "provenance.json",
        reference_dir / "trace.csv",
    )
    input_paths = (
        cfg.checkpoint.resolve(),
        cfg.controls.resolve(),
        cfg.warm_reference.resolve(),
        cfg.fixture.resolve() / "volume.vtu",
        cfg.fixture.resolve() / "skin.vtp",
        cfg.fixture.resolve() / "summary.json",
        *reference_paths,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        cherries.log_input(path)

    reference_config = json.loads(reference_paths[0].read_text(encoding="utf-8"))
    pinned_reference = {
        "method": "FiberModes",
        "skin_factor": 0.12,
        "fat_factor": 1.0,
        "muscle_factor": 0.8,
        "soft_nu": 0.46,
        "skin_nu": 0.46,
        "fat_model": "stable",
        "fat_nu": 0.499,
        "forward_rtol": 1e-5,
        "forward_atol": 1e-12,
        "adjoint_rtol": 1e-7,
        "target_name": "Smile",
        "target_scale": 1.0,
    }
    if any(reference_config[key] != value for key, value in pinned_reference.items()):
        raise ValueError("run 28 material or tolerance configuration changed")

    with np.load(cfg.checkpoint, allow_pickle=False) as saved:
        if set(saved.files) != {"q", "u", "Ainv"}:
            raise ValueError("run 24 checkpoint schema changed")
        q_final = np.asarray(saved["q"], dtype=np.float64).copy()
        checkpoint_ainv = np.asarray(saved["Ainv"], dtype=np.float64).copy()
    with np.load(cfg.warm_reference, allow_pickle=False) as saved:
        if set(saved.files) != {"q", "u", "Ainv"}:
            raise ValueError("run 28 reference schema changed")
        warm_q = np.asarray(saved["q"], dtype=np.float64).copy()
        warm_u = np.asarray(saved["u"], dtype=np.float64).copy()
        warm_ainv = np.asarray(saved["Ainv"], dtype=np.float64).copy()
    if q_final.shape != (140, 1) or checkpoint_ainv.shape != (120_020, 3, 3):
        raise ValueError("run 24 is not the expected FiberModes checkpoint")
    if not np.array_equal(q_final, warm_q) or not np.array_equal(
        checkpoint_ainv, warm_ainv
    ):
        raise ValueError("run 28 does not preserve run 24 q/Ainv exactly")

    with np.load(cfg.controls, allow_pickle=False) as basis:
        control_indices = np.asarray(basis["control_indices"], dtype=np.int64).copy()
        control_weights = np.asarray(basis["weights"], dtype=np.float64).copy()
        basis_active_ids = np.asarray(basis["active_ids"], dtype=np.int64).copy()
        basis_region_ids = np.asarray(basis["active_region_ids"], dtype=np.int64).copy()
        basis_active_mass = np.asarray(basis["active_mass"], dtype=np.float64).copy()
    if control_indices.shape != (120_020, 4) or control_weights.shape != (
        120_020,
        4,
    ):
        raise ValueError("FiberModes control interpolation schema changed")
    if not np.allclose(control_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-15):
        raise ValueError("FiberModes interpolation weights are not a partition")

    physics = FacePhysics(
        cfg.fixture,
        skin_factor=0.12,
        fat_factor=1.0,
        muscle_factor=0.8,
        rtol=1e-5,
        atol=1e-12,
        adjoint_rtol=1e-7,
        soft_nu=0.46,
        fat_nu=0.499,
        fat_model="stable",
        skin_nu=0.46,
        target_name="Smile",
        target_scale=1.0,
        line_search_max_steps=30,
    )
    if not np.array_equal(basis_active_ids, physics.ids):
        raise ValueError("control active IDs differ from FacePhysics")
    if not np.array_equal(basis_region_ids, physics.region):
        raise ValueError("control region IDs differ from FacePhysics")
    if not np.allclose(basis_active_mass, physics.volumes, rtol=1e-12, atol=0.0):
        raise ValueError("control active masses differ from FacePhysics")
    control_indices_t = torch.as_tensor(control_indices)
    control_weights_t = torch.as_tensor(control_weights)

    source_dir = output / "sources"
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
    reference_copy = output / "reference-28"
    reference_copy.mkdir()
    for path in reference_paths:
        shutil.copy2(path, reference_copy / path.name)

    input_receipts = {
        path.relative_to(ROOT).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in input_paths
    }
    provenance = {
        "schema_version": 1,
        "experiment_role": (
            "independent quasistatic activation-continuation check; forward equilibria "
            "only; not inverse optimization or physical time"
        ),
        "increments": INCREMENTS,
        "stages": INCREMENTS + 1,
        "inputs": input_receipts,
        "sources": source_hashes,
        "saved_controls": {
            "q_shape": list(q_final.shape),
            "q_sha256": array_sha256(q_final),
            "Ainv_shape": list(checkpoint_ainv.shape),
            "Ainv_sha256": array_sha256(checkpoint_ainv),
            "q_source": cfg.checkpoint.resolve().relative_to(ROOT).as_posix(),
        },
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write_json(output / "provenance.json", provenance)

    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seed = np.zeros_like(physics.points)
    previous_u_hash: str | None = None
    start_all = time.perf_counter()
    final_u: np.ndarray | None = None
    final_ainv: np.ndarray | None = None
    for stage in range(INCREMENTS + 1):
        cherries.set_step(stage)
        alpha = stage / INCREMENTS
        q = q_final.copy() if stage == INCREMENTS else alpha * q_final
        if stage == INCREMENTS and not np.array_equal(q, q_final):
            raise AssertionError("final ramp q is not the exact saved q")
        q_tensor = torch.as_tensor(q)
        local_q = (control_weights_t[..., None] * q_tensor[control_indices_t]).sum(
            dim=1
        )
        ainv_tensor, _ = am.matrices(local_q, "F", physics.fibers)
        ainv = ainv_tensor.detach().cpu().numpy().copy()
        if stage == 0 and (
            np.any(q != 0.0)
            or not np.array_equal(ainv, np.broadcast_to(np.eye(3), ainv.shape))
        ):
            raise AssertionError("stage zero is not exact zero activation")
        if stage == INCREMENTS and (
            not np.array_equal(q, warm_q)
            or not np.array_equal(ainv, warm_ainv)
            or not np.array_equal(ainv, checkpoint_ainv)
        ):
            raise ValueError("final ramp q/Ainv differs from runs 24 or 28")

        seed_receipt = (
            {"kind": "exact_zero_displacement_rest"}
            if stage == 0
            else {
                "kind": "previous_accepted_equilibrium",
                "stage": stage - 1,
                "u_sha256": previous_u_hash,
            }
        )
        if stage:
            LOG.info(
                "Solving activation stage %d/%d, alpha=%.1f",
                stage,
                INCREMENTS,
                alpha,
            )
        else:
            LOG.info("Recording exact q=0, u=0 rest reference")
        stage_start = time.perf_counter()
        try:
            if stage == 0:
                u = np.zeros_like(physics.points)
                solver_receipt = {
                    "performed": False,
                    "reason": (
                        "q=0 gives identity Ainv and u=0 is the exact unloaded rest "
                        "reference; strict solves begin at the first nonzero increment"
                    ),
                }
            else:
                u_tensor = physics.solve(am.packed(ainv_tensor), seed)
                u = u_tensor.detach().cpu().numpy().copy()
                solver_receipt = {"performed": True, **physics.last_forward}
            metrics = stage_metrics(physics, u, ainv)
            validate_endpoint(physics, u, metrics, require_solver_success=bool(stage))
            npz_path = output / f"stage-{stage:02d}.npz"
            vtu_path = output / f"stage-{stage:02d}.vtu"
            np.savez_compressed(
                npz_path,
                q=q,
                u=u,
                Ainv=ainv,
                stage=np.asarray(stage, dtype=np.int64),
                alpha=np.asarray(alpha, dtype=np.float64),
            )
            physics.save_mesh(vtu_path, u, ainv)
            outputs = validate_stage_files(npz_path, vtu_path, q, u, ainv)
        except Exception as error:
            failure = {
                "status": "failure",
                "stage": stage,
                "alpha": alpha,
                "seed": seed_receipt,
                "solver": getattr(physics, "last_forward", None),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "policy": "halt immediately; no fallback or tolerance relaxation",
            }
            write_json(output / f"stage-{stage:02d}-failure.json", failure)
            raise

        receipt = {
            "status": "success",
            "stage": stage,
            "alpha": alpha,
            "meaning": (
                "exact unloaded rest reference"
                if stage == 0
                else (
                    "strictly converged forward equilibrium at a prescribed activation "
                    "fraction; not an inverse step or physical-time sample"
                )
            ),
            "seed": seed_receipt,
            "q": {
                "shape": list(q.shape),
                "min": float(q.min()),
                "max": float(q.max()),
                "sha256": array_sha256(q),
            },
            "Ainv": {
                "shape": list(ainv.shape),
                "sha256": array_sha256(ainv),
            },
            "u": {"shape": list(u.shape), "sha256": array_sha256(u)},
            "materials_actual": physics.material_spec,
            "solver": solver_receipt,
            "metrics": metrics,
            "wall_s": time.perf_counter() - stage_start,
            "outputs": outputs,
        }
        if stage == INCREMENTS:
            receipt["exact_reference_checks"] = {
                "q_equals_run24": bool(np.array_equal(q, q_final)),
                "q_equals_run28": bool(np.array_equal(q, warm_q)),
                "Ainv_equals_run24": bool(np.array_equal(ainv, checkpoint_ainv)),
                "Ainv_equals_run28": bool(np.array_equal(ainv, warm_ainv)),
            }
        write_json(output / f"stage-{stage:02d}-receipt.json", receipt)
        receipts.append(receipt)
        row = {
            "stage": stage,
            "alpha": alpha,
            "forward_steps": solver_receipt.get("steps"),
            "forward_grad_norm": solver_receipt.get("grad_norm"),
            "fit_rms_mm": metrics["fit_rms_mm"],
            "motion_rms_mm": metrics["motion_rms_mm"],
            "detF_min": metrics["detF_min"],
            "detF_max": metrics["detF_max"],
            "detG_min": metrics["detG_min"],
            "activation_eigen_min": metrics["activation_eigen_min"],
            "stage_wall_s": receipt["wall_s"],
        }
        rows.append(row)
        write_trace(output / "trace.csv", rows)
        logged_metrics = {
            "ramp/alpha": alpha,
            "ramp/fit_rms_mm": metrics["fit_rms_mm"],
            "ramp/min_detF": metrics["detF_min"],
        }
        if stage:
            logged_metrics["ramp/forward_steps"] = physics.last_forward["steps"]
        cherries.log_metrics(logged_metrics, step=stage)
        if stage:
            LOG.info(
                "Accepted stage %d: steps=%d fit=%.6f mm minJ=%.6f",
                stage,
                physics.last_forward["steps"],
                metrics["fit_rms_mm"],
                metrics["detF_min"],
            )
        else:
            LOG.info(
                "Recorded exact rest reference with minJ=%.6f", metrics["detF_min"]
            )
        seed = u
        previous_u_hash = array_sha256(u)
        final_u = u
        final_ainv = ainv

    if final_u is None or final_ainv is None:
        raise AssertionError("ramp produced no final equilibrium")
    final_npz = output / "final.npz"
    final_vtu = output / "final.vtu"
    np.savez_compressed(final_npz, q=q_final, u=final_u, Ainv=final_ainv)
    physics.save_mesh(final_vtu, final_u, final_ainv)
    final_roundtrip = validate_stage_files(
        final_npz,
        final_vtu,
        q_final,
        final_u,
        final_ainv,
    )
    comparison = displacement_comparison(physics, final_u, warm_u)
    summary = {
        "schema_version": 1,
        "status": "success",
        "experiment_role": (
            "independent quasistatic activation-continuation check; not inverse "
            "optimization or physical time"
        ),
        "continuation": {
            "stages": INCREMENTS + 1,
            "activation_increments": INCREMENTS,
            "alphas": [stage / INCREMENTS for stage in range(INCREMENTS + 1)],
            "initial_seed": "exact zero displacement at q=0",
            "subsequent_seed": "previous accepted equilibrium",
            "failure_policy": "halt visibly; no fallback or tolerance relaxation",
        },
        "materials_actual": physics.material_spec,
        "forward_tolerance": physics.forward_tolerance,
        "exact_reference_checks": {
            "q_equals_run24": True,
            "q_equals_run28": True,
            "Ainv_equals_run24": True,
            "Ainv_equals_run28": True,
            "q_sha256": array_sha256(q_final),
            "Ainv_sha256": array_sha256(final_ainv),
        },
        "final_metrics": receipts[-1]["metrics"],
        "final_u_vs_run28_warm_equilibrium": comparison,
        "stages": [
            {
                "stage": receipt["stage"],
                "alpha": receipt["alpha"],
                "solver": receipt["solver"],
                "metrics": receipt["metrics"],
                "receipt": f"stage-{receipt['stage']:02d}-receipt.json",
            }
            for receipt in receipts
        ],
        "outputs": {
            "final_npz": final_roundtrip["npz"],
            "final_vtu": final_roundtrip["vtu"],
            "roundtrip": {
                key: value
                for key, value in final_roundtrip.items()
                if key not in {"npz", "vtu"}
            },
        },
        "wall_s": time.perf_counter() - start_all,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "artifact-manifest.json", artifact_manifest(output))
    LOG.info("Completed activation continuation: %s", summary)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
