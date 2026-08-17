from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from _common import resolve_recorded_path, slugify, toy

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input("10-prepare-manifest.json")
    output_manifest: Path = cherries.output("20-forward-manifest.json", mkdir=True)

    activation_inv: tuple[float, float, float, float, float, float] = (
        0.25,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    skin_energy_enabled: bool = False
    skin_prestrain_enabled: bool = False
    skin_e: float = toy.SKIN_E
    skin_thickness: float = toy.SKIN_THICKNESS
    skin_prestrain: float = 0.0
    require_convergence: bool = True


def activation_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 0] += values[0]
    matrix[1, 1] += values[1]
    matrix[2, 2] += values[2]
    matrix[0, 1] = matrix[1, 0] = values[3]
    matrix[1, 2] = matrix[2, 1] = values[4]
    matrix[0, 2] = matrix[2, 0] = values[5]
    return matrix


def validate_config(cfg: Config) -> np.ndarray:
    values = np.asarray(cfg.activation_inv, dtype=np.float64)
    if values.shape != (6,):
        msg = f"activation_inv must be a 6-vector, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "activation_inv must contain only finite values"
        raise ValueError(msg)
    eigenvalues = np.linalg.eigvalsh(activation_matrix(values))
    if eigenvalues.min() <= 0.0:
        msg = (
            "I + symmetric(activation_inv) must be positive definite, got "
            f"eigenvalues {eigenvalues.tolist()}"
        )
        raise ValueError(msg)
    if cfg.skin_prestrain_enabled and not cfg.skin_energy_enabled:
        msg = "skin prestrain requires skin energy"
        raise ValueError(msg)
    if cfg.skin_e <= 0.0:
        msg = f"skin_e must be positive, got {cfg.skin_e}"
        raise ValueError(msg)
    if cfg.skin_thickness <= 0.0:
        msg = f"skin_thickness must be positive, got {cfg.skin_thickness}"
        raise ValueError(msg)
    if cfg.skin_prestrain < 0.0:
        msg = f"skin_prestrain must be non-negative, got {cfg.skin_prestrain}"
        raise ValueError(msg)
    return values


def setup_label(cfg: Config) -> str:
    if not cfg.skin_energy_enabled:
        return "no-skin"
    if cfg.skin_prestrain_enabled:
        return "skin-prestrain"
    return "skin-no-prestrain"


def solve_case(
    *,
    mesh_path: Path,
    label: str,
    source_case: dict[str, Any],
    activation_inv: np.ndarray,
    cfg: Config,
    output_dir: Path,
) -> dict[str, Any]:
    mesh = pv.read(mesh_path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    resolution = toy.mesh_resolution(mesh)
    variant = toy.LossVariant(
        name="l2",
        skin_energy=cfg.skin_energy_enabled,
        skin_prestrain=cfg.skin_prestrain_enabled,
        activation_mode="shared",
    )
    case = toy.ToyCase(
        resolution=resolution,
        mode="squash",
        variant=variant,
        target_y=0.0,
    )
    forward, skin = toy.build_forward(
        mesh,
        case,
        skin_e=cfg.skin_e,
        skin_thickness=cfg.skin_thickness,
        skin_prestrain=cfg.skin_prestrain,
    )

    active_ids = np.flatnonzero(
        np.asarray(mesh.cell_data[toy.ACTIVE_FRACTION], dtype=np.float64)
        > toy.ACTIVE_FRACTION_TOL
    )
    if active_ids.size == 0:
        msg = f"{label} has no active muscle tetrahedra"
        raise RuntimeError(msg)
    active_ids_t = torch.as_tensor(active_ids, dtype=torch.long, device="cuda")
    active_activation_inv = torch.as_tensor(
        np.broadcast_to(activation_inv, (active_ids.size, 6)).copy(),
        dtype=torch.float64,
        device="cuda",
    )
    materials = toy.material_tree(
        forward.model.get_materials(),
        active_activation_inv,
        active_ids_t,
        mesh.n_cells,
    )
    forward.model.set_materials(materials)

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = toy.to_numpy(forward.state.u)
    target = toy.target_displacement(mesh, -toy.SQUASH_TARGET_MAGNITUDE)
    full_activation_inv = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    full_activation_inv[active_ids] = activation_inv

    metrics: dict[str, Any] = {
        "label": label,
        "source_mesh_path": str(mesh_path),
        "fat_thickness/min": float(source_case["fat_thickness/min"]),
        "fat_thickness/center": float(source_case["fat_thickness/center"]),
        "forward/elapsed_s": float(elapsed_s),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active_ids.size),
        "skin/energy_enabled": bool(cfg.skin_energy_enabled),
        "skin/prestrain_enabled": bool(cfg.skin_prestrain_enabled),
        "skin/E_MPa": float(cfg.skin_e) if cfg.skin_energy_enabled else 0.0,
        "skin/thickness": float(cfg.skin_thickness) if cfg.skin_energy_enabled else 0.0,
        "skin/prestrain": float(cfg.skin_prestrain)
        if cfg.skin_prestrain_enabled
        else 0.0,
        "skin/n_triangles": int(0 if skin is None else skin.n_cells),
        **{
            f"activation_inv/{name}": float(value)
            for name, value in zip(
                ("x", "y", "z", "xy", "yz", "xz"),
                activation_inv,
                strict=True,
            )
        },
        **toy.forward_solution_metrics(solution),
        **toy.displacement_summary(mesh, displacement, active_ids),
        **toy.geometry_summary(mesh),
        **toy.top_area_metrics(mesh, displacement, target),
        **toy.bumpiness_metrics(mesh, displacement, target),
        **toy.near_muscle_top_metrics(mesh, displacement, target),
    }

    case_dir = output_dir / slugify(label)
    result_path = case_dir / "result.vtu"
    summary_path = case_dir / "forward-summary.json"
    result = toy.make_result_mesh(
        mesh,
        target,
        displacement,
        full_activation_inv,
        metrics,
    )
    result.point_data["ContractedPoint"] = result.points + displacement
    result_path.parent.mkdir(parents=True, exist_ok=True)
    melon.save(result, result_path)
    metrics["result_path"] = str(result_path)
    metrics["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_input(mesh_path)
    cherries.log_output(result_path)
    cherries.log_output(summary_path)
    logger.info(
        "Solved %s: min fat %.6g, %s, %.2fs",
        label,
        metrics["fat_thickness/min"],
        metrics["forward/result"],
        elapsed_s,
    )
    if cfg.require_convergence and not metrics["forward/success"]:
        msg = f"{label} forward solve did not converge: {metrics['forward/result']}"
        raise RuntimeError(msg)
    return metrics


def main(cfg: Config) -> None:
    activation_inv = validate_config(cfg)
    cfg.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(cfg.input_manifest.read_text(encoding="utf-8"))
    source_cases = manifest.get("cases")
    if not isinstance(source_cases, list) or not source_cases:
        msg = "prepare manifest must contain a non-empty cases list"
        raise ValueError(msg)

    toy.configure_runtime()
    setup = setup_label(cfg)
    output_dir = cfg.output_manifest.parent / "20-forward" / setup
    rows: list[dict[str, Any]] = []
    for step, source_case in enumerate(source_cases):
        label = str(source_case["label"])
        mesh_path = resolve_recorded_path(
            cfg.input_manifest, str(source_case["mesh_path"])
        )
        row = solve_case(
            mesh_path=mesh_path,
            label=label,
            source_case=source_case,
            activation_inv=activation_inv,
            cfg=cfg,
            output_dir=output_dir,
        )
        rows.append(row)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{label}/fat_thickness_min": row["fat_thickness/min"],
                f"{label}/elapsed_s": row["forward/elapsed_s"],
                f"{label}/forward_success": float(row["forward/success"]),
                f"{label}/top_y_std": row["bumpiness/top_y_std"],
                f"{label}/displacement_laplacian_rms": row[
                    "bumpiness/displacement_laplacian_rms"
                ],
            }
        )

    output_manifest = {
        "schema_version": 1,
        "kind": "fat-thickness-forward-results",
        "source_manifest": str(cfg.input_manifest),
        "setup": setup,
        "activation_inv": activation_inv.tolist(),
        "activation_inv_matrix_eigenvalues": np.linalg.eigvalsh(
            activation_matrix(activation_inv)
        ).tolist(),
        "skin_energy_enabled": cfg.skin_energy_enabled,
        "skin_prestrain_enabled": cfg.skin_prestrain_enabled,
        "skin_e": cfg.skin_e,
        "skin_thickness": cfg.skin_thickness,
        "skin_prestrain": cfg.skin_prestrain,
        "require_convergence": cfg.require_convergence,
        "cases": rows,
    }
    cfg.output_manifest.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Wrote %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(main)
