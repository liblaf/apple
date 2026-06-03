from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon

SOURCE = Path(
    "/home/liblaf/github/liblaf/melon/exp/2025/04/30/"
    "human-head-anatomy/data/42-expression-muscle-orientation-3152k.vtu"
)
OUTPUT_STEM = "10-forward-face-3152k"
IN_FACE_CONVEX = "InFaceConvex"
IN_FACE_CONTEXT_TYPO = "InFaceContex"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    source: Path = cherries.input(SOURCE)
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output_stem: str = OUTPUT_STEM

    expression: str = "Expression000"
    target_scale: float = 1.0
    target_point_mask: str = "IsFace"
    fixed_point_masks: tuple[str, ...] = ("IsCranium", "IsMandible")
    activation_muscle_ids: tuple[int, ...] = (46, 47)
    active_fraction_tol: float = 1.0e-3

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0
    use_smas: bool = True


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


def zygomaticus_major_mask(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    muscle_id = require_array(mesh, "cell", "MuscleId").astype(np.int64)
    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    return np.isin(muscle_id, cfg.activation_muscle_ids) & (
        muscle > cfg.active_fraction_tol
    )


def zero_activation_fields(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    zero = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    mesh.cell_data[ACTIVATION.vtk] = zero.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero.copy()


def disjoint_fractions(
    muscle: np.ndarray, smas: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    muscle_fraction = np.clip(muscle, 0.0, 1.0)
    smas_fraction = np.clip(smas - muscle_fraction, 0.0, 1.0)
    background_fraction = np.clip(1.0 - np.maximum(smas, muscle_fraction), 0.0, 1.0)
    return background_fraction, muscle_fraction, smas_fraction


def add_material_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    muscle = require_array(mesh, "cell", "MuscleFraction").astype(np.float64)
    smas = require_array(mesh, "cell", "SmasFraction").astype(np.float64)
    background, muscle, smas_stiffness = disjoint_fractions(muscle, smas)
    if not cfg.use_smas:
        background = np.clip(1.0 - muscle, 0.0, 1.0)
        smas_stiffness = np.zeros_like(muscle)
    lambda_, mu = lame_parameters(cfg.E, cfg.nu)
    activation_mask = zygomaticus_major_mask(mesh, cfg)

    mesh.cell_data[BACKGROUND_FRACTION] = background
    mesh.cell_data[ACTIVE_FRACTION] = muscle
    mesh.cell_data[SMAS_STIFFNESS_FRACTION] = smas_stiffness
    mesh.cell_data["ActivationMask"] = activation_mask.astype(np.int8)
    mesh.cell_data["ZygomaticusMajorMask"] = activation_mask.astype(np.int8)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.ones(mesh.n_cells, dtype=np.float64)
    zero_activation_fields(mesh)


def fixed_point_mask(
    mesh: pv.UnstructuredGrid, cfg: Config
) -> tuple[np.ndarray, dict[str, int]]:
    fixed = np.zeros(mesh.n_points, dtype=bool)
    counts: dict[str, int] = {}
    for name in cfg.fixed_point_masks:
        mask = require_array(mesh, "point", name).astype(bool)
        fixed |= mask
        counts[name] = int(mask.sum())
        mesh.point_data[f"Fixed{name.removeprefix('Is')}"] = mask.astype(np.int8)
    if not np.any(fixed):
        msg = f"fixed point masks selected no points: {cfg.fixed_point_masks}"
        raise ValueError(msg)
    return fixed, counts


def add_boundary_conditions(
    mesh: pv.UnstructuredGrid, cfg: Config
) -> tuple[np.ndarray, dict[str, int]]:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    fixed, counts = fixed_point_mask(mesh, cfg)
    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[fixed, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedBoundary"] = fixed.astype(np.int8)
    return fixed, counts


def make_target_mesh(
    mesh: pv.UnstructuredGrid, cfg: Config, fixed: np.ndarray
) -> pv.UnstructuredGrid:
    target = mesh.copy(deep=True)
    displacement = require_array(target, "point", cfg.expression).astype(np.float64)
    displacement = cfg.target_scale * displacement
    displacement[fixed] = 0.0
    target_mask = require_array(target, "point", cfg.target_point_mask).astype(bool)
    if not np.any(target_mask):
        msg = f"point_data[{cfg.target_point_mask!r}] selected no target points"
        raise ValueError(msg)

    target.point_data["Displacement"] = displacement
    target.point_data["TargetDisplacement"] = displacement
    target.point_data["TargetPoint"] = target.points + displacement
    target.point_data[TARGET_SURFACE_MASK] = target_mask.astype(np.int8)
    target.point_data["TargetSurfacePoint"] = target_mask.astype(np.int8)
    target.field_data["TargetExpression"] = np.asarray([cfg.expression])
    target.field_data["TargetScale"] = np.asarray([cfg.target_scale])
    target.field_data["TargetPointMask"] = np.asarray([cfg.target_point_mask])
    target.field_data["TargetPointCount"] = np.asarray([int(target_mask.sum())])
    zero_activation_fields(target)
    return target


def add_metadata(
    mesh: pv.UnstructuredGrid,
    cfg: Config,
    selected_cell_data: str,
    fixed_counts: dict[str, int],
) -> None:
    from liblaf.apple.common import FIXED_MASK

    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    fixed = np.asarray(mesh.point_data[FIXED_MASK.vtk], dtype=bool)[:, 0]
    mesh.field_data["Source"] = np.asarray([str(cfg.source)])
    mesh.field_data["SelectedCellData"] = np.asarray([selected_cell_data])
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["SmasStiffnessRatio"] = np.asarray([cfg.smas_stiffness_ratio])
    mesh.field_data["UseSmas"] = np.asarray([int(cfg.use_smas)])
    mesh.field_data["ActivationMuscleIds"] = np.asarray(cfg.activation_muscle_ids)
    mesh.field_data["ActiveFractionTol"] = np.asarray([cfg.active_fraction_tol])
    mesh.field_data["ActivationTetCount"] = np.asarray([int(active.sum())])
    mesh.field_data["FixedPointCount"] = np.asarray([int(fixed.sum())])
    mesh.field_data["FixedPointMasks"] = np.asarray(list(cfg.fixed_point_masks))
    for name, count in fixed_counts.items():
        mesh.field_data[f"{name}FixedPointCount"] = np.asarray([count])
    mesh.field_data["NoCollision"] = np.asarray([1])


def metric_summary(
    mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> dict[str, Any]:
    muscle = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    smas = np.asarray(mesh.cell_data["SmasFraction"], dtype=np.float64)
    smas_stiffness = np.asarray(
        mesh.cell_data[SMAS_STIFFNESS_FRACTION], dtype=np.float64
    )
    volume = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    target_mask = np.asarray(target.point_data[TARGET_SURFACE_MASK], dtype=bool)
    target_disp = np.asarray(target.point_data["Displacement"], dtype=np.float64)
    target_norm = np.linalg.norm(target_disp[target_mask], axis=1)
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    fixed = np.asarray(mesh.point_data["FixedBoundary"], dtype=bool)
    return {
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "activation/n_tets": int(active.sum()),
        "target/n_points": int(target_mask.sum()),
        "fixed/n_points": int(fixed.sum()),
        "fixed/n_cranium_points": int(np.asarray(mesh.point_data["FixedCranium"]).sum()),
        "fixed/n_mandible_points": int(np.asarray(mesh.point_data["FixedMandible"]).sum()),
        "volume/muscle_fraction": float(np.sum(muscle * volume)),
        "volume/smas_fraction": float(np.sum(smas * volume)),
        "volume/smas_stiffness_fraction": float(np.sum(smas_stiffness * volume)),
        "volume/activation_fraction": float(np.sum(muscle[active] * volume[active])),
        "target/displacement_mean": float(target_norm.mean()),
        "target/displacement_rms": float(
            np.linalg.norm(target_disp[target_mask]) / np.sqrt(target_mask.sum())
        ),
        "target/displacement_max": float(target_norm.max()),
    }


def output_paths(cfg: Config) -> tuple[Path, Path]:
    if cfg.output_stem == OUTPUT_STEM:
        return cfg.output_input, cfg.output_target
    base = cfg.output_input.parent
    return (
        base / f"{cfg.output_stem}-input.vtu",
        base / f"{cfg.output_stem}-target.vtu",
    )


def log_dynamic_outputs(cfg: Config, paths: tuple[Path, Path]) -> None:
    if cfg.output_stem == OUTPUT_STEM:
        return
    for path in paths:
        if path.exists():
            cherries.log_output(path)


def main(cfg: Config) -> None:
    output_input, output_target = output_paths(cfg)
    source = pv.read(cfg.source)
    if not isinstance(source, pv.UnstructuredGrid):
        source = source.cast_to_unstructured_grid()

    mesh, selected_cell_data = extract_face_mesh(source)
    add_material_fields(mesh, cfg)
    fixed, fixed_counts = add_boundary_conditions(mesh, cfg)
    add_metadata(mesh, cfg, selected_cell_data, fixed_counts)
    target = make_target_mesh(mesh, cfg, fixed)
    add_metadata(target, cfg, selected_cell_data, fixed_counts)

    melon.save(output_input, mesh)
    melon.save(output_target, target)
    log_dynamic_outputs(cfg, (output_input, output_target))
    summary = metric_summary(mesh, target)
    cherries.log_metrics(summary)
    print(summary)
    print(f"saved: {output_input}")
    print(f"saved: {output_target}")


if __name__ == "__main__":
    cherries.main(main)
