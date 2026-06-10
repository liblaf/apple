from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    name: str
    input_path: Path
    target_path: Path
    inverse_path: Path
    target_mask: str
    target_volume_is_physical: bool


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("10-real-mesh-area-volume-summary.json")
    output_csv: Path = cherries.output("10-real-mesh-area-volume.csv")
    output_table: Path = cherries.output("10-real-mesh-area-volume-table.md")
    output_vtu_dir: Path = cherries.output("10-real-mesh-area-volume-vtu", mkdir=True)


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = f"could not find repo root from {path}"
    raise RuntimeError(msg)


def cases(root: Path) -> list[Case]:
    return [
        Case(
            name="3152k-expression001",
            input_path=root
            / "exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-input.vtu",
            target_path=root
            / "exp/2026/05/27/inverse-face/data/10-inverse-face-3152k-target.vtu",
            inverse_path=root
            / "exp/2026/05/27/inverse-face/data/20-inverse-face-3152k.vtu",
            target_mask="IsFace",
            target_volume_is_physical=False,
        ),
        Case(
            name="515k-nosmas",
            input_path=root
            / "exp/2026/05/27/forward-face/data/10-forward-face-515k-nosmas-input.vtu",
            target_path=root
            / "exp/2026/05/27/forward-face/data/20-forward-face-515k-nosmas.vtu",
            inverse_path=root
            / "exp/2026/05/27/forward-face/data/30-inverse-face-515k-nosmas.vtu",
            target_mask="IsFace",
            target_volume_is_physical=True,
        ),
    ]


def require_path(path: Path) -> None:
    if path.exists():
        cherries.log_input(path)
        return
    msg = f"missing input: {path}"
    raise FileNotFoundError(msg)


def read_grid(path: Path) -> pv.UnstructuredGrid:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    return mesh


def tetra_cells(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if pv.CellType.TETRA not in mesh.cells_dict:
        msg = f"expected tetrahedra in {mesh}"
        raise ValueError(msg)
    return np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)


def tetra_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.abs(tetra_signed_volumes(points, tets))


def volume_arrays(
    *,
    points: np.ndarray,
    displacement: np.ndarray,
    tets: np.ndarray,
    rest_signed_volume: np.ndarray,
    rest_volume: np.ndarray,
) -> dict[str, np.ndarray]:
    deformed = points + displacement
    deformed_signed = tetra_signed_volumes(deformed, tets)
    deformed_abs = np.abs(deformed_signed)
    signed_delta = deformed_signed - rest_signed_volume
    abs_delta = deformed_abs - rest_volume
    return {
        "SignedVolume": deformed_signed,
        "SignedVolumeDelta": signed_delta,
        "SignedVolumeRelChange": signed_delta / rest_signed_volume,
        "AbsVolume": deformed_abs,
        "AbsVolumeDelta": abs_delta,
        "AbsVolumeRelChange": abs_delta / rest_volume,
        "Inverted": (deformed_signed <= 0.0).astype(np.int8),
    }


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def surface_triangles(mesh: pv.UnstructuredGrid) -> np.ndarray:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "surface extraction did not preserve vtkOriginalPointIds"
        raise KeyError(msg)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = "surface triangulation produced non-triangle faces"
        raise ValueError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return original_ids[faces[:, 1:]]


def rel_change(new: float, old: float) -> float:
    if old == 0.0:
        return math.nan
    return new / old - 1.0


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}/min": math.nan,
            f"{prefix}/p01": math.nan,
            f"{prefix}/p05": math.nan,
            f"{prefix}/p50": math.nan,
            f"{prefix}/p95": math.nan,
            f"{prefix}/p99": math.nan,
            f"{prefix}/max": math.nan,
        }
    return {
        f"{prefix}/min": float(np.min(values)),
        f"{prefix}/p01": float(np.quantile(values, 0.01)),
        f"{prefix}/p05": float(np.quantile(values, 0.05)),
        f"{prefix}/p50": float(np.quantile(values, 0.50)),
        f"{prefix}/p95": float(np.quantile(values, 0.95)),
        f"{prefix}/p99": float(np.quantile(values, 0.99)),
        f"{prefix}/max": float(np.max(values)),
    }


def mask_from_mesh(mesh: pv.UnstructuredGrid, name: str) -> np.ndarray:
    if name in mesh.point_data:
        return np.asarray(mesh.point_data[name], dtype=bool)
    if "TargetSurfaceMask" in mesh.point_data:
        return np.asarray(mesh.point_data["TargetSurfaceMask"], dtype=bool)
    if "IsFace" in mesh.point_data:
        return np.asarray(mesh.point_data["IsFace"], dtype=bool)
    return np.ones(mesh.n_points, dtype=bool)


def displacement_from(mesh: pv.UnstructuredGrid, *, preferred: str = "Displacement") -> np.ndarray:
    if preferred in mesh.point_data:
        return np.asarray(mesh.point_data[preferred], dtype=np.float64)
    if "TargetDisplacement" in mesh.point_data:
        return np.asarray(mesh.point_data["TargetDisplacement"], dtype=np.float64)
    msg = f"mesh has neither {preferred!r} nor 'TargetDisplacement'"
    raise KeyError(msg)


def restrict_displacement(displacement: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros_like(displacement)
    result[mask] = displacement[mask]
    return result


def displacement_stats(
    displacement: np.ndarray, mask: np.ndarray, prefix: str
) -> dict[str, float]:
    outside = ~mask
    values = np.linalg.norm(displacement, axis=1)
    row: dict[str, float] = {
        f"{prefix}/inside_nonzero_points": float(np.sum((values > 1.0e-12) & mask)),
        f"{prefix}/outside_nonzero_points": float(np.sum((values > 1.0e-12) & outside)),
    }
    if np.any(outside):
        row[f"{prefix}/outside_rms"] = float(
            np.linalg.norm(displacement[outside]) / math.sqrt(int(outside.sum()))
        )
        row[f"{prefix}/outside_max"] = float(values[outside].max())
    else:
        row[f"{prefix}/outside_rms"] = math.nan
        row[f"{prefix}/outside_max"] = math.nan
    return row


def validate_same_topology(
    base: pv.UnstructuredGrid, other: pv.UnstructuredGrid, path: Path
) -> None:
    if base.n_points != other.n_points or base.n_cells != other.n_cells:
        msg = (
            f"topology mismatch for {path}: "
            f"points {base.n_points} != {other.n_points}, "
            f"cells {base.n_cells} != {other.n_cells}"
        )
        raise ValueError(msg)
    if not np.allclose(base.points, other.points):
        msg = f"rest points differ for {path}"
        raise ValueError(msg)


def measure_state(
    *,
    case_name: str,
    state: str,
    base: pv.UnstructuredGrid,
    displacement: np.ndarray,
    mask: np.ndarray,
    tets: np.ndarray,
    rest_volume: np.ndarray,
    surface: np.ndarray,
    rest_surface_area: np.ndarray,
    rest_signed_volume: np.ndarray,
    volume_is_physical: bool,
    volume_scope: str,
) -> dict[str, Any]:
    points = np.asarray(base.points, dtype=np.float64)
    deformed = points + displacement
    deformed_signed_volume = tetra_signed_volumes(deformed, tets)
    deformed_volume = tetra_volumes(deformed, tets)
    volume_ratio = deformed_volume / rest_volume
    surface_mask_all = np.all(mask[surface], axis=1)
    surface_mask_any = np.any(mask[surface], axis=1)
    deformed_surface_area = triangle_areas(deformed, surface)

    rest_volume_sum = float(np.sum(rest_volume))
    deformed_volume_sum = float(np.sum(deformed_volume))
    rest_signed_volume_sum = float(np.sum(rest_signed_volume))
    deformed_signed_volume_sum = float(np.sum(deformed_signed_volume))
    rest_area_sum = float(np.sum(rest_surface_area))
    deformed_area_sum = float(np.sum(deformed_surface_area))

    rest_target_area_all = float(np.sum(rest_surface_area[surface_mask_all]))
    def_target_area_all = float(np.sum(deformed_surface_area[surface_mask_all]))
    rest_target_area_any = float(np.sum(rest_surface_area[surface_mask_any]))
    def_target_area_any = float(np.sum(deformed_surface_area[surface_mask_any]))
    norm = np.linalg.norm(displacement, axis=1)
    target_norm = norm[mask]

    row: dict[str, Any] = {
        "case": case_name,
        "state": state,
        "volume_is_physical": volume_is_physical,
        "volume_scope": volume_scope,
        "n_points": int(base.n_points),
        "n_tets": int(tets.shape[0]),
        "n_mask_points": int(mask.sum()),
        "rest_volume": rest_volume_sum,
        "deformed_volume": deformed_volume_sum,
        "volume_abs_rel_change": rel_change(deformed_volume_sum, rest_volume_sum),
        "rest_signed_volume": rest_signed_volume_sum,
        "deformed_signed_volume": deformed_signed_volume_sum,
        "volume_rel_change": rel_change(
            deformed_signed_volume_sum, rest_signed_volume_sum
        ),
        "volume_inverted_tets": int(np.sum(deformed_signed_volume <= 0.0)),
        "volume_inverted_fraction": float(np.mean(deformed_signed_volume <= 0.0)),
        "rest_surface_area": rest_area_sum,
        "deformed_surface_area": deformed_area_sum,
        "surface_area_rel_change": rel_change(deformed_area_sum, rest_area_sum),
        "rest_mask_area_all": rest_target_area_all,
        "deformed_mask_area_all": def_target_area_all,
        "mask_area_all_rel_change": rel_change(
            def_target_area_all, rest_target_area_all
        ),
        "rest_mask_area_any": rest_target_area_any,
        "deformed_mask_area_any": def_target_area_any,
        "mask_area_any_rel_change": rel_change(
            def_target_area_any, rest_target_area_any
        ),
        "displacement_mean": float(norm.mean()),
        "displacement_rms": float(np.linalg.norm(displacement) / math.sqrt(norm.size)),
        "displacement_max": float(norm.max()),
        "mask_displacement_mean": float(target_norm.mean()),
        "mask_displacement_rms": float(
            np.linalg.norm(displacement[mask]) / math.sqrt(int(mask.sum()))
        ),
        "mask_displacement_max": float(target_norm.max()),
        "volume_ratio_weighted_mean": float(
            np.average(volume_ratio, weights=rest_volume)
        ),
        "volume_ratio_weighted_std": float(
            math.sqrt(np.average((volume_ratio - 1.0) ** 2, weights=rest_volume))
        ),
    }
    row.update(quantiles(volume_ratio - 1.0, "cell_volume_rel_change"))
    return row


def add_volume_prefix(
    mesh: pv.UnstructuredGrid,
    prefix: str,
    arrays: dict[str, np.ndarray],
) -> None:
    for name, values in arrays.items():
        mesh.cell_data[f"{prefix}{name}"] = values


def write_diagnostic_vtu(
    *,
    path: Path,
    case: Case,
    base: pv.UnstructuredGrid,
    mask: np.ndarray,
    target_raw: np.ndarray,
    target_face_only: np.ndarray,
    inverse: np.ndarray,
    tets: np.ndarray,
    rest_signed_volume: np.ndarray,
    rest_volume: np.ndarray,
) -> None:
    points = np.asarray(base.points, dtype=np.float64)
    result = base.copy(deep=True)
    mask_count = np.sum(mask[tets], axis=1).astype(np.int8)
    result.point_data["TargetValidPoint"] = mask.astype(np.int8)
    result.point_data["TargetRawDisplacement"] = target_raw
    result.point_data["TargetFaceOnlyDisplacement"] = target_face_only
    result.point_data["InverseDisplacement"] = inverse
    result.point_data["InverseMinusTargetOnValidPoints"] = inverse - target_face_only
    result.cell_data["TargetValidPointCount"] = mask_count
    result.cell_data["TargetValidTetAny"] = (mask_count > 0).astype(np.int8)
    result.cell_data["TargetValidTetAll"] = (mask_count == 4).astype(np.int8)
    result.cell_data["TargetValidTetFraction"] = mask_count.astype(np.float64) / 4.0
    result.cell_data["RestSignedVolume"] = rest_signed_volume
    result.cell_data["RestAbsVolume"] = rest_volume
    add_volume_prefix(
        result,
        "TargetRaw",
        volume_arrays(
            points=points,
            displacement=target_raw,
            tets=tets,
            rest_signed_volume=rest_signed_volume,
            rest_volume=rest_volume,
        ),
    )
    add_volume_prefix(
        result,
        "TargetFaceOnly",
        volume_arrays(
            points=points,
            displacement=target_face_only,
            tets=tets,
            rest_signed_volume=rest_signed_volume,
            rest_volume=rest_volume,
        ),
    )
    add_volume_prefix(
        result,
        "Inverse",
        volume_arrays(
            points=points,
            displacement=inverse,
            tets=tets,
            rest_signed_volume=rest_signed_volume,
            rest_volume=rest_volume,
        ),
    )
    result.field_data["TargetVolumeIsPhysical"] = np.asarray(
        [int(case.target_volume_is_physical)]
    )
    result.field_data["TargetMaskName"] = np.asarray([case.target_mask])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save(path)
    cherries.log_output(path)


def error_row(
    *,
    case_name: str,
    target_displacement: np.ndarray,
    inverse_displacement: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    error = inverse_displacement - target_displacement
    error_norm = np.linalg.norm(error, axis=1)
    mask_error = error_norm[mask]
    target_norm = np.linalg.norm(target_displacement[mask], axis=1)
    return {
        "case": case_name,
        "state": "inverse-minus-target",
        "n_points": int(error.shape[0]),
        "n_mask_points": int(mask.sum()),
        "mask_error_mean": float(mask_error.mean()),
        "mask_error_rms": float(np.linalg.norm(error[mask]) / math.sqrt(int(mask.sum()))),
        "mask_error_max": float(mask_error.max()),
        "mask_target_displacement_rms": float(
            np.linalg.norm(target_displacement[mask]) / math.sqrt(int(mask.sum()))
        ),
        "mask_error_rms_fraction_of_target": float(
            np.linalg.norm(error[mask]) / np.linalg.norm(target_displacement[mask])
        )
        if np.linalg.norm(target_displacement[mask]) > 0.0
        else math.nan,
        "mask_target_displacement_max": float(target_norm.max()),
    }


def flatten_for_csv(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    keys = sorted({key for row in rows for key in row})
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {}
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float) and math.isnan(value):
                flat[key] = "nan"
            elif isinstance(value, int | float | str | bool):
                flat[key] = value
            else:
                flat[key] = json.dumps(value)
        flat_rows.append(flat)
    return keys, flat_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys, flat_rows = flatten_for_csv(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_rows)


def format_float(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    metric_rows = [
        row
        for row in rows
        if row.get("state")
        in {"target", "target-face-only-diagnostic", "inverse"}
    ]
    error_rows = {
        row["case"]: row for row in rows if row.get("state") == "inverse-minus-target"
    }
    lines = [
        "| case | state | volume scope | physical volume? | signed volume change | abs volume change | inverted tets | mask area change | mask disp RMS | mask error RMS | mask error/target RMS |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        err = error_rows.get(row["case"], {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["state"]),
                    str(row["volume_scope"]),
                    str(bool(row["volume_is_physical"])),
                    format_float(row["volume_rel_change"]),
                    format_float(row["volume_abs_rel_change"]),
                    format_float(row["volume_inverted_fraction"]),
                    format_float(row["mask_area_all_rel_change"]),
                    format_float(row["mask_displacement_rms"]),
                    format_float(err.get("mask_error_rms")),
                    format_float(err.get("mask_error_rms_fraction_of_target")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    root = repo_root()
    rows: list[dict[str, Any]] = []
    log_metrics: dict[str, float] = {}
    for case in cases(root):
        for path in (case.input_path, case.target_path, case.inverse_path):
            require_path(path)
        base = read_grid(case.input_path)
        target = read_grid(case.target_path)
        inverse = read_grid(case.inverse_path)
        validate_same_topology(base, target, case.target_path)
        validate_same_topology(base, inverse, case.inverse_path)

        mask = mask_from_mesh(target, case.target_mask)
        if not np.any(mask):
            msg = f"{case.name} selected no points with {case.target_mask}"
            raise ValueError(msg)
        target_displacement_raw = displacement_from(target)
        target_displacement_face_only = restrict_displacement(target_displacement_raw, mask)
        target_displacement_for_volume = (
            target_displacement_raw
            if case.target_volume_is_physical
            else target_displacement_face_only
        )
        inverse_displacement = displacement_from(inverse)
        tets = tetra_cells(base)
        rest_volume = tetra_volumes(np.asarray(base.points, dtype=np.float64), tets)
        rest_signed_volume = tetra_signed_volumes(
            np.asarray(base.points, dtype=np.float64), tets
        )
        if np.any(rest_volume <= 0.0):
            msg = f"{case.name} has non-positive rest tetra volume"
            raise ValueError(msg)
        surface = surface_triangles(base)
        rest_surface_area = triangle_areas(np.asarray(base.points, dtype=np.float64), surface)
        diagnostic_path = cfg.output_vtu_dir / f"{case.name}-volume-change.vtu"
        write_diagnostic_vtu(
            path=diagnostic_path,
            case=case,
            base=base,
            mask=mask,
            target_raw=target_displacement_raw,
            target_face_only=target_displacement_face_only,
            inverse=inverse_displacement,
            tets=tets,
            rest_signed_volume=rest_signed_volume,
            rest_volume=rest_volume,
        )

        target_row = measure_state(
            case_name=case.name,
            state="target"
            if case.target_volume_is_physical
            else "target-face-only-diagnostic",
            base=base,
            displacement=target_displacement_for_volume,
            mask=mask,
            tets=tets,
            rest_volume=rest_volume,
            surface=surface,
            rest_surface_area=rest_surface_area,
            rest_signed_volume=rest_signed_volume,
            volume_is_physical=case.target_volume_is_physical,
            volume_scope="full-field" if case.target_volume_is_physical else "IsFace-only",
        )
        target_row.update(
            displacement_stats(target_displacement_raw, mask, "target_raw_displacement")
        )
        inverse_row = measure_state(
            case_name=case.name,
            state="inverse",
            base=base,
            displacement=inverse_displacement,
            mask=mask,
            tets=tets,
            rest_volume=rest_volume,
            surface=surface,
            rest_surface_area=rest_surface_area,
            rest_signed_volume=rest_signed_volume,
            volume_is_physical=True,
            volume_scope="forward-solved-full-field",
        )
        diff_row = error_row(
            case_name=case.name,
            target_displacement=target_displacement_raw,
            inverse_displacement=inverse_displacement,
            mask=mask,
        )
        rows.extend([target_row, inverse_row, diff_row])
        log_metrics[f"{case.name}/target/volume_rel_change"] = target_row[
            "volume_rel_change"
        ]
        log_metrics[f"{case.name}/inverse/volume_rel_change"] = inverse_row[
            "volume_rel_change"
        ]
        log_metrics[f"{case.name}/target/mask_area_rel_change"] = target_row[
            "mask_area_all_rel_change"
        ]
        log_metrics[f"{case.name}/inverse/mask_area_rel_change"] = inverse_row[
            "mask_area_all_rel_change"
        ]
        log_metrics[f"{case.name}/inverse/error_rms_fraction"] = diff_row[
            "mask_error_rms_fraction_of_target"
        ]
        logger.info(
            "%s target volume change %.6g (%s), inverse %.6g, error fraction %.6g",
            case.name,
            target_row["volume_rel_change"],
            target_row["volume_scope"],
            inverse_row["volume_rel_change"],
            diff_row["mask_error_rms_fraction_of_target"],
        )

    cfg.output_summary.write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    cherries.log_metrics(log_metrics)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
