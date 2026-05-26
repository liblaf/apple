from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon

SOURCE = Path(
    "/home/liblaf/github/liblaf/melon/exp/2025/04/30/"
    "human-head-anatomy/data/41-expression-3152k.vtu"
)
OUTPUT_STEM = "10-inverse-face"
IN_FACE_CONVEX = "InFaceConvex"
IN_FACE_CONTEXT_TYPO = "InFaceContex"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    source: Path = SOURCE
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")

    expression: str = "Expression000"
    target_scale: float = 1.0
    target_point_mask: str = "IsFace"
    fixed_point_mask: str = "IsCranium"
    active_fraction_tol: float = 1.0e-3

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2


def require_array(obj: pv.DataSet, association: str, name: str) -> np.ndarray:
    data = obj.cell_data if association == "cell" else obj.point_data
    if name not in data:
        msg = f"{association}_data[{name!r}] is missing"
        raise KeyError(msg)
    return np.asarray(data[name])


def face_cell_mask(mesh: pv.UnstructuredGrid) -> tuple[np.ndarray, str]:
    if IN_FACE_CONVEX in mesh.cell_data:
        return np.asarray(mesh.cell_data[IN_FACE_CONVEX], dtype=bool), IN_FACE_CONVEX
    if IN_FACE_CONTEXT_TYPO in mesh.cell_data:
        return (
            np.asarray(mesh.cell_data[IN_FACE_CONTEXT_TYPO], dtype=bool),
            IN_FACE_CONTEXT_TYPO,
        )
    msg = f"source mesh has neither {IN_FACE_CONVEX!r} nor {IN_FACE_CONTEXT_TYPO!r}"
    raise KeyError(msg)


def extract_face_mesh(source: pv.UnstructuredGrid) -> tuple[pv.UnstructuredGrid, str]:
    mask, selected_name = face_cell_mask(source)
    if not np.any(mask):
        msg = f"no tetrahedra selected by {selected_name}"
        raise ValueError(msg)
    mesh = source.extract_cells(mask)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    cell_types = set(np.asarray(mesh.celltypes).tolist())
    if cell_types != {int(pv.CellType.TETRA)}:
        msg = f"expected tetra-only face mesh, got cell types {sorted(cell_types)}"
        raise ValueError(msg)
    return mesh, selected_name


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def active_mask(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    return muscle > cfg.active_fraction_tol


def zero_activation_fields(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    zero = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    mesh.cell_data[ACTIVATION.vtk] = zero.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero.copy()


def add_material_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    smas = require_array(mesh, "cell", "SmasFraction").astype(np.float64)
    background = np.clip(1.0 - muscle - smas, 0.0, 1.0)
    lambda_, mu = lame_parameters(cfg.E, cfg.nu)

    mesh.cell_data[BACKGROUND_FRACTION] = background
    mesh.cell_data[ACTIVE_FRACTION] = muscle
    mesh.cell_data[SMAS_STIFFNESS_FRACTION] = smas
    mesh.cell_data["ActivationMask"] = active_mask(mesh, cfg).astype(np.int8)
    mesh.cell_data["InverseActiveMask"] = mesh.cell_data["ActivationMask"]
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.ones(mesh.n_cells, dtype=np.float64)
    zero_activation_fields(mesh)


def add_boundary_conditions(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    fixed = require_array(mesh, "point", cfg.fixed_point_mask).astype(bool)
    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[fixed, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedCranium"] = fixed.astype(np.int8)
    return fixed


def target_point_mask(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    mask = require_array(mesh, "point", cfg.target_point_mask).astype(bool)
    if not np.any(mask):
        msg = f"point_data[{cfg.target_point_mask!r}] selected no target points"
        raise ValueError(msg)
    return mask


def make_target_mesh(
    mesh: pv.UnstructuredGrid, cfg: Config, fixed: np.ndarray
) -> pv.UnstructuredGrid:
    target = mesh.copy(deep=True)
    displacement = require_array(target, "point", cfg.expression).astype(np.float64)
    displacement = cfg.target_scale * displacement
    displacement[fixed] = 0.0
    mask = target_point_mask(target, cfg)

    target.point_data["Displacement"] = displacement
    target.point_data["TargetDisplacement"] = displacement
    target.point_data["TargetPoint"] = target.points + displacement
    target.point_data[TARGET_SURFACE_MASK] = mask.astype(np.int8)
    target.point_data["TargetSurfacePoint"] = mask.astype(np.int8)
    target.field_data["TargetExpression"] = np.asarray([cfg.expression])
    target.field_data["TargetScale"] = np.asarray([cfg.target_scale])
    target.field_data["TargetPointMask"] = np.asarray([cfg.target_point_mask])
    target.field_data["TargetPointCount"] = np.asarray([int(mask.sum())])
    zero_activation_fields(target)
    return target


def add_metadata(mesh: pv.UnstructuredGrid, cfg: Config, selected_cell_data: str) -> None:
    from liblaf.apple.common import FIXED_MASK

    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    fixed = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)[:, 0]
    mesh.field_data["Source"] = np.asarray([str(cfg.source)])
    mesh.field_data["SelectedCellData"] = np.asarray([selected_cell_data])
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["SmasStiffnessRatio"] = np.asarray([cfg.smas_stiffness_ratio])
    mesh.field_data["ActiveFractionTol"] = np.asarray([cfg.active_fraction_tol])
    mesh.field_data["ActiveTetCount"] = np.asarray([int(active.sum())])
    mesh.field_data["FixedPointCount"] = np.asarray([int(fixed.sum())])
    mesh.field_data["NoCollision"] = np.asarray([1])


def metric_summary(
    mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> dict[str, Any]:
    muscle = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    smas = np.asarray(mesh.cell_data["SmasFraction"], dtype=np.float64)
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    target_mask = np.asarray(target.point_data[TARGET_SURFACE_MASK], dtype=bool)
    target_disp = np.asarray(target.point_data["Displacement"], dtype=np.float64)
    target_norm = np.linalg.norm(target_disp[target_mask], axis=1)
    return {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "n_active_tets": int(np.asarray(mesh.cell_data["ActivationMask"]).sum()),
        "n_target_points": int(target_mask.sum()),
        "n_fixed_points": int(np.asarray(mesh.point_data["FixedCranium"]).sum()),
        "muscle_fraction_volume": float(np.sum(muscle * volume)),
        "smas_fraction_volume": float(np.sum(smas * volume)),
        "target_displacement_mean": float(target_norm.mean()),
        "target_displacement_rms": float(
            np.linalg.norm(target_disp[target_mask]) / np.sqrt(target_mask.sum())
        ),
        "target_displacement_max": float(target_norm.max()),
    }


def main(cfg: Config) -> None:
    source = pv.read(cfg.source)
    if not isinstance(source, pv.UnstructuredGrid):
        source = source.cast_to_unstructured_grid()

    mesh, selected_cell_data = extract_face_mesh(source)
    add_material_fields(mesh, cfg)
    fixed = add_boundary_conditions(mesh, cfg)
    add_metadata(mesh, cfg, selected_cell_data)
    target = make_target_mesh(mesh, cfg, fixed)
    add_metadata(target, cfg, selected_cell_data)
    zero_activation_fields(mesh)

    melon.save(cfg.output_input, mesh)
    melon.save(cfg.output_target, target)
    summary = metric_summary(mesh, target)
    cherries.log_metrics(summary)
    print(summary)
    print(f"saved: {cfg.output_input}")
    print(f"saved: {cfg.output_target}")


if __name__ == "__main__":
    cherries.main(main)
