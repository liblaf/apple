from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from _common import slugify, toy

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_manifest: Path = cherries.output("10-prepare-manifest.json", mkdir=True)

    labels: tuple[str, ...] = ("thin", "current", "thick")
    rim_heights: tuple[float, ...] = (0.07, 0.10, 0.14)
    tetwild_lr: float = 0.02
    parabolic_rest_amplitude: float = 0.12
    parabolic_target_amplitude: float = 0.02
    parabolic_grid: int = 32


def validate_config(cfg: Config) -> list[tuple[str, float]]:
    if len(cfg.labels) != len(cfg.rim_heights):
        msg = (
            "labels and rim_heights must have the same length, got "
            f"{len(cfg.labels)} and {len(cfg.rim_heights)}"
        )
        raise ValueError(msg)
    if not cfg.labels:
        msg = "at least one thickness case is required"
        raise ValueError(msg)
    labels = [slugify(label) for label in cfg.labels]
    if len(labels) != len(set(labels)):
        msg = f"case labels are not unique after slugification: {labels}"
        raise ValueError(msg)
    if len(cfg.rim_heights) != len(set(cfg.rim_heights)):
        msg = f"rim heights must be unique, got {cfg.rim_heights}"
        raise ValueError(msg)
    muscle_top = float(toy.SMAS_BOUNDS[3])
    if any(rim <= muscle_top for rim in cfg.rim_heights):
        msg = (
            "every rim height must lie above the SMAS/muscle layer top "
            f"{muscle_top:g}, got {cfg.rim_heights}"
        )
        raise ValueError(msg)
    if cfg.tetwild_lr <= 0.0:
        msg = f"tetwild_lr must be positive, got {cfg.tetwild_lr}"
        raise ValueError(msg)
    if cfg.parabolic_grid < 2:
        msg = f"parabolic_grid must be at least 2, got {cfg.parabolic_grid}"
        raise ValueError(msg)
    return list(zip(labels, cfg.rim_heights, strict=True))


def prepare_case(
    *,
    label: str,
    rim_height: float,
    cfg: Config,
    output_dir: Path,
) -> dict[str, Any]:
    resolution = toy.ResolutionSpec(
        name=toy.label_lr(cfg.tetwild_lr), lr=cfg.tetwild_lr
    )
    mesh = toy.make_tetwild_mesh(
        resolution,
        geometry="parabolic",
        parabolic_rim_height=rim_height,
        parabolic_rest_amplitude=cfg.parabolic_rest_amplitude,
        parabolic_target_amplitude=cfg.parabolic_target_amplitude,
        parabolic_grid=cfg.parabolic_grid,
    )
    toy.add_material_and_boundary_fields(mesh, resolution)

    case_dir = output_dir / label
    mesh_path = case_dir / "mesh.vtu"
    summary_path = case_dir / "prepare-summary.json"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, mesh_path)

    target_mask = np.asarray(mesh.point_data[toy.TARGET_SURFACE_MASK], dtype=bool)
    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    target = toy.target_displacement(mesh, -toy.SQUASH_TARGET_MAGNITUDE)
    zero = np.zeros_like(target)
    muscle_top = float(toy.SMAS_BOUNDS[3])
    min_fat_thickness = rim_height - muscle_top
    center_fat_thickness = min_fat_thickness + cfg.parabolic_rest_amplitude
    summary: dict[str, Any] = {
        "label": label,
        "mesh_path": str(mesh_path),
        "summary_path": str(summary_path),
        "resolution": resolution.name,
        "tetwild/lr": float(resolution.lr),
        "tetwild/lr_interpretation": "relative_edge_length_fac",
        "fat_thickness/min": float(min_fat_thickness),
        "fat_thickness/center": float(center_fat_thickness),
        "fat_thickness/offset_from_current": float(rim_height - 0.10),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active_mask.sum()),
        "n_target_points": int(target_mask.sum()),
        "fat/E_MPa": float(toy.FAT_E),
        "fat/nu": float(toy.FAT_NU),
        "muscle/E_MPa": float(toy.MUSCLE_E),
        "muscle/nu": float(toy.MUSCLE_NU),
        "aponeurosis/E_MPa": float(toy.APONEUROSIS_E),
        "aponeurosis/nu": float(toy.APONEUROSIS_NU),
        **toy.geometry_summary(mesh),
        **toy.top_area_metrics(mesh, zero, target),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(mesh_path)
    cherries.log_output(summary_path)
    logger.info(
        "Prepared %s: min fat %.6g, center fat %.6g, %d tets",
        label,
        min_fat_thickness,
        center_fat_thickness,
        mesh.n_cells,
    )
    return summary


def main(cfg: Config) -> None:
    cases = validate_config(cfg)
    cfg.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir = cfg.output_manifest.parent / "10-meshes"
    rows: list[dict[str, Any]] = []
    for step, (label, rim_height) in enumerate(cases):
        row = prepare_case(
            label=label,
            rim_height=rim_height,
            cfg=cfg,
            output_dir=output_dir,
        )
        rows.append(row)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{label}/fat_thickness_min": row["fat_thickness/min"],
                f"{label}/fat_thickness_center": row["fat_thickness/center"],
                f"{label}/n_tets": row["n_tets"],
                f"{label}/n_active_tets": row["n_active_tets"],
                f"{label}/target_area_over_rest": row[
                    "surface/top_area_target_over_rest"
                ],
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "fat-thickness-prepared-meshes",
        "tetwild_lr": cfg.tetwild_lr,
        "parabolic_rest_amplitude": cfg.parabolic_rest_amplitude,
        "parabolic_target_amplitude": cfg.parabolic_target_amplitude,
        "parabolic_grid": cfg.parabolic_grid,
        "muscle_layer_top": float(toy.SMAS_BOUNDS[3]),
        "cases": rows,
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Wrote %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(main)
