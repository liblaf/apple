from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


def load_base_module() -> Any:
    path = Path(__file__).with_name("20-toy-unreachable-inverse.py")
    spec = importlib.util.spec_from_file_location("toy_unreachable_inverse", path)
    if spec is None or spec.loader is None:
        msg = f"could not load base toy experiment from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


@dataclass(frozen=True)
class ForwardCase:
    resolution: Any

    @property
    def stem(self) -> str:
        return f"40-toy-forward-activation-{self.resolution.name}"


class Config(BASE.Config):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("40-toy-forward-activation-summary.json")
    output_csv: Path = cherries.output("40-toy-forward-activation-cases.csv")
    output_table: Path = cherries.output("40-toy-forward-activation-table.md")

    resolutions: tuple[str, ...] = ("coarse", "medium", "fine")
    activation: tuple[float, float, float, float, float, float] = (
        -0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    forward_max_steps: int = 5000


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def relative_value(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def activation_inv_from_activation(
    activation: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    xx, yy, zz, xy, xz, yz = activation
    A = np.asarray(
        [
            [1.0 + xx, xy, xz],
            [xy, 1.0 + yy, yz],
            [xz, yz, 1.0 + zz],
        ],
        dtype=np.float64,
    )
    A_inv = np.linalg.inv(A) - np.eye(3, dtype=np.float64)
    return (
        float(A_inv[0, 0]),
        float(A_inv[1, 1]),
        float(A_inv[2, 2]),
        float(A_inv[0, 1]),
        float(A_inv[0, 2]),
        float(A_inv[1, 2]),
    )


def apply_activation(mesh: Any, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_inv = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_value = np.asarray(cfg.activation, dtype=np.float64)
    activation_inv_value = np.asarray(
        activation_inv_from_activation(cfg.activation), dtype=np.float64
    )
    activation[active] = activation_value
    activation_inv[active] = activation_inv_value
    mesh.cell_data[ACTIVATION.vtk] = activation
    mesh.cell_data[ACTIVATION_INV.vtk] = activation_inv
    mesh.cell_data["ActivationNorm"] = np.linalg.norm(activation, axis=1)
    mesh.cell_data["ActivationInvNorm"] = np.linalg.norm(activation_inv, axis=1)
    return activation_inv


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "forward/result": "missing",
            "forward/success": False,
            "forward/steps": math.nan,
            "forward/grad_norm": math.nan,
            "forward/relative_grad_norm": math.nan,
            "forward/grad_norm_first": math.nan,
            "forward/line_search_ok": False,
            "forward/line_search_steps": math.nan,
            "forward/stagnation_count": math.nan,
        }
    convergence_state = solution.state.convergence_state
    line_search_state = solution.state.line_search_state
    grad_norm = to_float(convergence_state.grad_norm)
    grad_norm_first = to_float(convergence_state.grad_norm_first)
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": relative_value(grad_norm, grad_norm_first),
        "forward/grad_norm_first": grad_norm_first,
        "forward/line_search_ok": bool(line_search_state.ok),
        "forward/line_search_steps": int(line_search_state.step),
        "forward/stagnation_count": int(convergence_state.stagnation_count),
    }


def make_result_mesh(
    mesh: Any,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    metrics: dict[str, Any],
) -> Any:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    target = np.zeros_like(displacement)
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["TargetPoint"] = result.points
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["ForwardActivationInv"] = activation_inv
    result.cell_data["ForwardActivationInvNorm"] = np.linalg.norm(
        activation_inv, axis=1
    )
    BASE.add_tetra_volume_change_fields(result, target, displacement)
    result.cell_data["VolumeForward"] = np.asarray(result.cell_data["VolumeInverse"])
    result.cell_data["VolumeForwardRelChange"] = np.asarray(
        result.cell_data["VolumeInverseRelChange"]
    )
    result.cell_data["SignedVolumeForward"] = np.asarray(
        result.cell_data["SignedVolumeInverse"]
    )
    result.cell_data["SignedVolumeForwardRelChange"] = np.asarray(
        result.cell_data["SignedVolumeInverseRelChange"]
    )
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        result.field_data[name] = np.asarray([value])
    return result


def displacement_stats(mesh: Any, displacement: np.ndarray) -> dict[str, float]:
    values = np.linalg.norm(displacement, axis=1)
    top = np.asarray(mesh.point_data[BASE.TARGET_SURFACE_MASK], dtype=bool)
    active_cells = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    tets = BASE.tetra_cells(mesh)
    active_points = np.zeros(mesh.n_points, dtype=bool)
    active_points[np.unique(tets[active_cells].reshape(-1))] = True
    stats = {
        "displacement/mean": float(values.mean()),
        "displacement/rms": float(
            np.linalg.norm(displacement) / math.sqrt(mesh.n_points)
        ),
        "displacement/max": float(values.max()),
        "top/displacement_y_mean": float(displacement[top, 1].mean()),
        "top/displacement_y_min": float(displacement[top, 1].min()),
        "top/displacement_y_max": float(displacement[top, 1].max()),
        "top/displacement_rms": float(
            np.linalg.norm(displacement[top]) / math.sqrt(int(top.sum()))
        ),
        "active_point/displacement_rms": float(
            np.linalg.norm(displacement[active_points])
            / math.sqrt(max(1, int(active_points.sum())))
        ),
        "active_point/displacement_max": float(values[active_points].max()),
    }
    stats.update(BASE.top_roughness(mesh, displacement))
    return stats


def solve_case(case: ForwardCase, cfg: Config) -> dict[str, Any]:
    start = time.perf_counter()
    mesh = BASE.make_tet_box(case.resolution)
    BASE.add_material_and_boundary_fields(mesh, cfg)
    activation_inv = apply_activation(mesh, cfg)
    data_dir = cfg.output_summary.parent
    input_path = data_dir / f"{case.stem}-input.vtu"
    output_path = data_dir / f"{case.stem}.vtu"
    melon.save(input_path, mesh)

    forward = BASE.build_forward(mesh, cfg)
    solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = to_numpy(forward.state.u)
    target_mask = np.asarray(mesh.point_data[BASE.TARGET_SURFACE_MASK], dtype=bool)
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)

    row: dict[str, Any] = {
        "case": case.stem,
        "resolution": case.resolution.name,
        "x_segments": int(case.resolution.x_segments),
        "y_levels": list(case.resolution.y_levels),
        "z_segments": int(case.resolution.z_segments),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active.sum()),
        "n_target_points": int(target_mask.sum()),
        "elapsed_s": float(elapsed_s),
        "activation/local_xx": float(cfg.activation[0]),
        "activation/local_yy": float(cfg.activation[1]),
        "activation/local_zz": float(cfg.activation[2]),
        "activation/local_xy": float(cfg.activation[3]),
        "activation/local_xz": float(cfg.activation[4]),
        "activation/local_yz": float(cfg.activation[5]),
        "activation_inv/local_xx": float(
            activation_inv_from_activation(cfg.activation)[0]
        ),
        "activation_inv/max_abs": float(np.abs(activation_inv[active]).max()),
    }
    row.update(forward_solution_metrics(solution))
    row.update(displacement_stats(mesh, displacement))
    row.update(
        {
            f"forward/{key}": value
            for key, value in BASE.geometry_change(
                mesh, displacement, target_mask
            ).items()
        }
    )

    numeric_metrics = {
        key: value
        for key, value in row.items()
        if isinstance(value, int | float | bool)
    }
    melon.save(
        output_path,
        make_result_mesh(mesh, displacement, activation_inv, numeric_metrics),
    )
    cherries.log_output(input_path)
    cherries.log_output(output_path)
    cherries.log_metrics(
        {f"{case.stem}/{key}": value for key, value in numeric_metrics.items()}
    )
    logger.info(
        "%s displacement rms %.6g max %.6g signed dV %.6g forward %s",
        case.stem,
        row["displacement/rms"],
        row["displacement/max"],
        row["forward/volume/rel_change"],
        row["forward/result"],
    )
    return row


def selected_cases(cfg: Config) -> list[ForwardCase]:
    cases: list[ForwardCase] = []
    for resolution_name in cfg.resolutions:
        if resolution_name not in BASE.RESOLUTION_SPECS:
            msg = (
                f"unknown resolution {resolution_name!r}; "
                f"choose from {sorted(BASE.RESOLUTION_SPECS)}"
            )
            raise ValueError(msg)
        cases.append(ForwardCase(resolution=BASE.RESOLUTION_SPECS[resolution_name]))
    return cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"y_levels"}
    keys = sorted({key for row in rows for key in row if key not in excluded})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_float(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | points | tets | active tets | forward result | steps | signed volume change | abs volume change | inverted tets | top y mean | top y std | top edge RMS | displacement RMS | displacement max |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["n_points"]),
                    str(row["n_tets"]),
                    str(row["n_active_tets"]),
                    str(row["forward/result"]),
                    format_float(row["forward/steps"]),
                    format_float(row["forward/volume/rel_change"]),
                    format_float(row["forward/volume/abs_rel_change"]),
                    format_float(row["forward/volume/inverted_fraction"]),
                    format_float(row["top/displacement_y_mean"]),
                    format_float(row["top_y/std"]),
                    format_float(row["top_y/edge_rms"]),
                    format_float(row["displacement/rms"]),
                    format_float(row["displacement/max"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    BASE.configure_runtime()
    rows = [solve_case(case, cfg) for case in selected_cases(cfg)]
    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
