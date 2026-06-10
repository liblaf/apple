from __future__ import annotations

import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

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


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("42-toy-volume-field-backfill-summary.json")
    output_csv: Path = cherries.output("42-toy-volume-field-backfill-cases.csv")
    output_table: Path = cherries.output("42-toy-volume-field-backfill-table.md")


def case_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("20-toy-*.vtu"))
    return [
        path
        for path in paths
        if not path.name.endswith("-input.vtu") and not path.name.endswith("-target.vtu")
    ]


def read_grid(path: Path) -> pv.UnstructuredGrid:
    mesh = pv.read(path)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    return mesh


def displacement_from(mesh: pv.UnstructuredGrid, name: str) -> np.ndarray:
    if name not in mesh.point_data:
        msg = f"{name!r} missing from {mesh}"
        raise KeyError(msg)
    return np.asarray(mesh.point_data[name], dtype=np.float64)


def patch_pair(result_path: Path) -> dict[str, Any]:
    target_path = result_path.with_name(f"{result_path.stem}-target.vtu")
    if not target_path.exists():
        msg = f"missing target mesh for {result_path}: {target_path}"
        raise FileNotFoundError(msg)

    result = read_grid(result_path)
    target_mesh = read_grid(target_path)
    target_displacement = displacement_from(result, "TargetDisplacement")
    solution_displacement = displacement_from(result, "Displacement")
    BASE.add_tetra_volume_change_fields(
        result,
        target_displacement,
        solution_displacement,
    )
    BASE.add_tetra_volume_change_fields(
        target_mesh,
        target_displacement,
        target_displacement,
    )
    result.save(result_path)
    target_mesh.save(target_path)
    cherries.log_output(result_path)
    cherries.log_output(target_path)

    inverse_rel = np.asarray(result.cell_data["VolumeInverseRelChange"], dtype=np.float64)
    target_rel = np.asarray(result.cell_data["VolumeTargetRelChange"], dtype=np.float64)
    signed_inverse_rel = np.asarray(
        result.cell_data["SignedVolumeInverseRelChange"], dtype=np.float64
    )
    signed_target_rel = np.asarray(
        result.cell_data["SignedVolumeTargetRelChange"], dtype=np.float64
    )
    row = {
        "case": result_path.stem,
        "result_path": str(result_path),
        "target_path": str(target_path),
        "n_points": int(result.n_points),
        "n_tets": int(result.n_cells),
        "target_volume_rel_min": float(np.nanmin(target_rel)),
        "target_volume_rel_max": float(np.nanmax(target_rel)),
        "inverse_volume_rel_min": float(np.nanmin(inverse_rel)),
        "inverse_volume_rel_max": float(np.nanmax(inverse_rel)),
        "target_signed_volume_rel_min": float(np.nanmin(signed_target_rel)),
        "target_signed_volume_rel_max": float(np.nanmax(signed_target_rel)),
        "inverse_signed_volume_rel_min": float(np.nanmin(signed_inverse_rel)),
        "inverse_signed_volume_rel_max": float(np.nanmax(signed_inverse_rel)),
    }
    logger.info("Patched %s and %s", result_path, target_path)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_float(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | tets | target dV min | target dV max | inverse dV min | inverse dV max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["n_tets"]),
                    format_float(row["target_signed_volume_rel_min"]),
                    format_float(row["target_signed_volume_rel_max"]),
                    format_float(row["inverse_signed_volume_rel_min"]),
                    format_float(row["inverse_signed_volume_rel_max"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    data_dir = cfg.output_summary.parent
    rows = [patch_pair(path) for path in case_paths(data_dir)]
    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    cherries.log_metrics({"patched_cases": len(rows)})
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
