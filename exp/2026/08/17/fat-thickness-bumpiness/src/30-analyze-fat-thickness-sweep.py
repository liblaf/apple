from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv

mpl.use("Agg")
import matplotlib.pyplot as plt
from _common import resolve_recorded_path, toy
from _surface_metrics import (
    ResampledSurface,
    common_xz_bounds,
    finite_scalar_metrics,
    gaussian_smooth,
    resample_top_surface,
    surface_metrics,
)

from liblaf import cherries

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input("20-forward-manifest.json")
    output_csv: Path = cherries.output("30-cross-grid-metrics.csv", mkdir=True)
    output_json: Path = cherries.output("30-cross-grid-metrics.json", mkdir=True)
    output_metric_plot: Path = cherries.output("30-cross-grid-metrics.png", mkdir=True)
    output_field_plot: Path = cherries.output("30-cross-grid-fields.png", mkdir=True)

    grid_size: int = 129
    high_frequency_cutoff_cycles: float = 8.0
    laplacian_smoothing_length: float = 0.04


def case_label(case: dict[str, Any]) -> str:
    for key in ("label", "case", "name"):
        if key in case:
            return str(case[key])
    msg = "manifest case has no label, case, or name"
    raise KeyError(msg)


def case_result_path(case: dict[str, Any]) -> str:
    for key in ("result_path", "result/path", "output_mesh"):
        if key in case:
            return str(case[key])
    msg = f"case {case_label(case)!r} has no result mesh path"
    raise KeyError(msg)


def case_min_fat_thickness(case: dict[str, Any], mesh: pv.DataSet) -> float:
    if "fat_thickness/min" in case:
        return float(case["fat_thickness/min"])
    rim_height = toy.field_float(
        mesh,
        toy.PARABOLIC_RIM_HEIGHT_FIELD,
        toy.PARABOLIC_RIM_HEIGHT,
    )
    return float(rim_height - toy.SMAS_BOUNDS[3])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["fat_thickness/min"]))
    thickness = np.asarray(
        [float(row["fat_thickness/min"]) for row in ordered], dtype=np.float64
    )
    panels = (
        ("grid/displacement_y_rms", "vertical response RMS"),
        (
            "grid/displacement_y_laplacian_over_rms",
            "normalized displacement roughness",
        ),
        (
            "grid/displacement_y_highpass_rms",
            "vertical high-pass RMS",
        ),
        (
            "grid/displacement_y_high_frequency_power_fraction",
            "displacement high-frequency power fraction",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    for axis, (key, title) in zip(axes.ravel(), panels, strict=True):
        values = np.asarray(
            [float(row.get(key, math.nan)) for row in ordered], dtype=np.float64
        )
        axis.plot(thickness, values, marker="o")
        for x, y, row in zip(thickness, values, ordered, strict=True):
            if math.isfinite(y):
                axis.annotate(
                    str(row["label"]), (x, y), xytext=(4, 4), textcoords="offset points"
                )
        axis.set_xlabel("minimum fat thickness")
        axis.set_ylabel(key.removeprefix("grid/"))
        axis.set_title(title)
        axis.grid(alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def symmetric_limit(fields: list[np.ndarray]) -> float:
    limit = max(float(np.nanmax(np.abs(field))) for field in fields)
    return limit if limit > 0.0 else 1.0


def plot_resampled_fields(
    path: Path,
    rows: list[dict[str, Any]],
    samples: list[ResampledSurface],
    *,
    smoothing_length: float,
) -> None:
    displacement_fields = [sample.displacement[..., 1] for sample in samples]
    highpass_fields = []
    for sample in samples:
        dx = float(sample.x[1, 0] - sample.x[0, 0])
        dz = float(sample.z[0, 1] - sample.z[0, 0])
        smooth = gaussian_smooth(
            sample.displacement[..., 1],
            dx=dx,
            dz=dz,
            smoothing_length=smoothing_length,
        )
        highpass_fields.append(sample.displacement[..., 1] - smooth)
    residual_fields = [sample.residual[..., 1] for sample in samples]
    displacement_limit = symmetric_limit(displacement_fields)
    highpass_limit = symmetric_limit(highpass_fields)
    residual_limit = symmetric_limit(residual_fields)
    fig, axes = plt.subplots(
        3,
        len(rows),
        figsize=(4.2 * len(rows), 10.2),
        squeeze=False,
        constrained_layout=True,
    )
    displacement_image = None
    highpass_image = None
    residual_image = None
    for column, (row, sample, highpass) in enumerate(
        zip(rows, samples, highpass_fields, strict=True)
    ):
        extent = (
            float(sample.z.min()),
            float(sample.z.max()),
            float(sample.x.min()),
            float(sample.x.max()),
        )
        displacement_image = axes[0, column].imshow(
            sample.displacement[..., 1],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-displacement_limit,
            vmax=displacement_limit,
            aspect="equal",
        )
        highpass_image = axes[1, column].imshow(
            highpass,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-highpass_limit,
            vmax=highpass_limit,
            aspect="equal",
        )
        residual_image = axes[2, column].imshow(
            sample.residual[..., 1],
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            vmin=-residual_limit,
            vmax=residual_limit,
            aspect="equal",
        )
        axes[0, column].set_title(
            f"{row['label']}\nmin fat={float(row['fat_thickness/min']):g}"
        )
        axes[2, column].set_xlabel("z")
        axes[0, column].set_ylabel("x")
        axes[1, column].set_ylabel("x")
        axes[2, column].set_ylabel("x")
    if displacement_image is not None:
        fig.colorbar(
            displacement_image,
            ax=axes[0, :].tolist(),
            label="vertical displacement",
            shrink=0.85,
        )
    if residual_image is not None:
        fig.colorbar(
            residual_image,
            ax=axes[2, :].tolist(),
            label="vertical residual",
            shrink=0.85,
        )
    if highpass_image is not None:
        fig.colorbar(
            highpass_image,
            ax=axes[1, :].tolist(),
            label=f"vertical high-pass (length={smoothing_length:g})",
            shrink=0.85,
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_config(cfg: Config) -> None:
    if cfg.grid_size < 5:
        msg = f"grid_size must be at least 5, got {cfg.grid_size}"
        raise ValueError(msg)
    if cfg.high_frequency_cutoff_cycles <= 0.0:
        msg = (
            "high_frequency_cutoff_cycles must be positive, got "
            f"{cfg.high_frequency_cutoff_cycles}"
        )
        raise ValueError(msg)
    if cfg.laplacian_smoothing_length <= 0.0:
        msg = (
            "laplacian_smoothing_length must be positive, got "
            f"{cfg.laplacian_smoothing_length}"
        )
        raise ValueError(msg)


def read_manifest_cases(path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, dict) for case in cases)
    ):
        msg = "input manifest must contain a non-empty list of case objects"
        raise ValueError(msg)
    failed = [
        case_label(case) for case in cases if case.get("forward/success") is False
    ]
    if failed:
        msg = "input manifest contains failed forward cases: " + ", ".join(failed)
        raise ValueError(msg)
    return cases


def main(cfg: Config) -> None:
    validate_config(cfg)
    for path in (
        cfg.output_csv,
        cfg.output_json,
        cfg.output_metric_plot,
        cfg.output_field_plot,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    cases = read_manifest_cases(cfg.input_manifest)

    meshes: list[pv.UnstructuredGrid] = []
    paths: list[Path] = []
    for case in cases:
        path = resolve_recorded_path(cfg.input_manifest, case_result_path(case))
        mesh = pv.read(path)
        if not isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
        paths.append(path)
        meshes.append(mesh)
        cherries.log_input(path)
    bounds = common_xz_bounds(meshes)

    rows: list[dict[str, Any]] = []
    samples: list[ResampledSurface] = []
    for step, (case, mesh, path) in enumerate(zip(cases, meshes, paths, strict=True)):
        label = case_label(case)
        sample = resample_top_surface(mesh, bounds=bounds, grid_size=cfg.grid_size)
        metrics = surface_metrics(
            sample,
            high_frequency_cutoff_cycles=cfg.high_frequency_cutoff_cycles,
            laplacian_smoothing_length=cfg.laplacian_smoothing_length,
            muscle_bounds=toy.mesh_muscle_bounds(mesh),
        )
        row: dict[str, Any] = {
            "label": label,
            "result_path": str(path),
            "fat_thickness/min": case_min_fat_thickness(case, mesh),
            "fat_thickness/center": float(
                case.get(
                    "fat_thickness/center",
                    case_min_fat_thickness(case, mesh)
                    + toy.field_float(
                        mesh,
                        toy.PARABOLIC_REST_AMPLITUDE_FIELD,
                        toy.PARABOLIC_REST_AMPLITUDE,
                    ),
                )
            ),
            **metrics,
        }
        rows.append(row)
        samples.append(sample)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{label}/{key.removeprefix('grid/')}": value
                for key, value in finite_scalar_metrics(metrics).items()
            }
        )
        logger.info(
            "Analyzed %s on %dx%d grid: residual RMS %.6g, residual lap %.6g",
            label,
            cfg.grid_size,
            cfg.grid_size,
            metrics["grid/residual_rms"],
            metrics["grid/residual_y_laplacian_rms"],
        )

    write_csv(cfg.output_csv, rows)
    payload = {
        "schema_version": 1,
        "kind": "fat-thickness-cross-grid-analysis",
        "source_manifest": str(cfg.input_manifest),
        "grid_size": cfg.grid_size,
        "common_xz_bounds": list(bounds),
        "high_frequency_cutoff_cycles_per_unit": cfg.high_frequency_cutoff_cycles,
        "laplacian_smoothing_length": cfg.laplacian_smoothing_length,
        "cases": rows,
    }
    cfg.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_metric_comparison(cfg.output_metric_plot, rows)
    plot_resampled_fields(
        cfg.output_field_plot,
        rows,
        samples,
        smoothing_length=cfg.laplacian_smoothing_length,
    )
    for path in (
        cfg.output_csv,
        cfg.output_json,
        cfg.output_metric_plot,
        cfg.output_field_plot,
    ):
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    cherries.main(main)
