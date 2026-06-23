from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from _human_face_config import (
    APONEUROSIS_E,
    APONEUROSIS_FRACTION,
    APONEUROSIS_NU,
    FAT_E,
    FAT_FRACTION,
    FAT_NU,
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    IS_FACE,
    IS_FIXED,
    MUSCLE_E,
    MUSCLE_FRACTION,
    MUSCLE_NU,
    SKIN_E,
    SKIN_NU,
    SKIN_THICKNESS,
    SOURCE_MESH,
    configure_runtime,
)
from _human_face_forward import set_volume_material
from _human_face_mesh import (
    add_required_fields,
    extract_simulation_mesh,
    orient_tetra_mesh,
    surface_original_ids,
    tetra_cells,
)
from _human_face_metrics import forward_solution_metrics, to_numpy
from _human_face_targets import make_target_mesh

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


class EstimateConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    output_summary: Path = cherries.output(
        "30-lips-corners-down-volume-prestrain-rest-recovery-summary.json", mkdir=True
    )
    output_result: Path = cherries.output(
        "30-lips-corners-down-volume-prestrain-rest-recovery.vtu", mkdir=True
    )
    output_skin: Path = cherries.output(
        "30-lips-corners-down-volume-prestrain-rest-recovery-skin.vtp", mkdir=True
    )
    output_skin_inspect: Path = cherries.output(
        "30-lips-corners-down-volume-prestrain-rest-recovery-skin-inspect.vtp",
        mkdir=True,
    )
    output_target: Path = cherries.output(
        "30-lips-corners-down-volume-prestrain-rest-recovery-target.vtu", mkdir=True
    )

    target_expression: str = "LipsCornersDown"
    area_ratio_floor: float = 1.0e-6
    stretch_floor: float = 1.0e-6
    max_steps: int = FORWARD_MAX_STEPS
    rtol: float = FORWARD_RTOL
    atol: float = FORWARD_ATOL


def triangle_faces(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = "expected a triangulated PolyData surface"
        raise ValueError(msg)
    return faces[:, 1:]


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}/min": math.nan,
            f"{prefix}/max": math.nan,
            f"{prefix}/mean": math.nan,
            f"{prefix}/rms": math.nan,
        }
    return {
        f"{prefix}/min": float(values.min()),
        f"{prefix}/max": float(values.max()),
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/rms": float(np.linalg.norm(values) / math.sqrt(values.size)),
    }


def vector_stats(
    prefix: str, values: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    norms = np.linalg.norm(np.asarray(values, dtype=np.float64), axis=1)
    return stats(prefix, norms)


def activation_inv_stats(prefix: str, activation_inv: np.ndarray) -> dict[str, float]:
    return {
        **stats(f"{prefix}/component", activation_inv.ravel()),
        **stats(f"{prefix}/norm", np.linalg.norm(activation_inv, axis=1)),
    }


def det_f_metrics(prefix: str, det_f: np.ndarray) -> dict[str, float | int]:
    return {
        f"{prefix}/inverted_cells": int((det_f <= 0.0).sum()),
        f"{prefix}/lt_0p5_cells": int((det_f < 0.5).sum()),
        f"{prefix}/gt_1p5_cells": int((det_f > 1.5).sum()),
        **stats(prefix, det_f),
    }


def surface_area_metrics(
    prefix: str, skin: pv.PolyData, displacement: np.ndarray
) -> dict[str, float]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    faces = triangle_faces(skin)
    surface_ids = np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    rest_area = np.asarray(skin.cell_data["RestArea"], dtype=np.float64)
    deformed_area = triangle_areas(skin.points + displacement[surface_ids], faces)
    area_ratio = np.ones_like(rest_area)
    valid = rest_area > 0.0
    area_ratio[valid] = deformed_area[valid] / rest_area[valid]
    is_face = np.asarray(skin.cell_data["IsFacePrestrainCell"], dtype=bool)
    active = np.asarray(skin.cell_data["IsStretchedPrestrainCell"], dtype=bool)

    result: dict[str, float] = {
        f"{prefix}/surface_area_ratio_all": float(
            deformed_area.sum() / rest_area.sum()
        ),
        f"{prefix}/surface_area_ratio_is_face": float(
            deformed_area[is_face].sum() / rest_area[is_face].sum()
        ),
    }
    if np.any(active):
        result[f"{prefix}/surface_area_ratio_active_prestrain"] = float(
            deformed_area[active].sum() / rest_area[active].sum()
        )
    else:
        result[f"{prefix}/surface_area_ratio_active_prestrain"] = math.nan
    result.update(stats(f"{prefix}/surface_area_ratio_cell", area_ratio))
    return result


def prepare_simulation_mesh(
    cfg: EstimateConfig,
) -> tuple[pv.UnstructuredGrid, dict[str, Any]]:
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh, subset_summary = extract_simulation_mesh(mesh)
    mesh, n_flipped = orient_tetra_mesh(mesh)
    field_summary = add_required_fields(mesh)
    summary: dict[str, Any] = {
        "source_mesh": str(cfg.input_mesh),
        "n_oriented_tets_flipped": int(n_flipped),
        **subset_summary,
        **field_summary,
    }
    return mesh, summary


def target_displacement(
    mesh: pv.UnstructuredGrid, expression: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if expression not in mesh.point_data:
        msg = f"missing target expression point data {expression!r}"
        raise KeyError(msg)
    raw = np.asarray(mesh.point_data[expression], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        msg = f"{expression!r} must be vec3 point data, got shape {raw.shape}"
        raise ValueError(msg)
    finite = np.isfinite(raw).all(axis=1)
    target = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    face = np.asarray(mesh.point_data[IS_FACE], dtype=bool)
    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    loss_mask = face & finite
    if not np.any(loss_mask):
        msg = f"{expression!r} selected no finite IsFace target points"
        raise ValueError(msg)
    target_norm = np.linalg.norm(target[loss_mask], axis=1)
    return (
        target,
        loss_mask,
        {
            "target/name": expression,
            "target/loss_points": int(loss_mask.sum()),
            "target/fixed_overlap_points": int((loss_mask & fixed).sum()),
            "target/displacement_rms": float(
                np.linalg.norm(target[loss_mask]) / math.sqrt(int(loss_mask.sum()))
            ),
            "target/displacement_mean": float(target_norm.mean()),
            "target/displacement_max": float(target_norm.max()),
        },
    )


def make_area_skin(
    mesh: pv.UnstructuredGrid, target: np.ndarray, cfg: EstimateConfig
) -> tuple[pv.PolyData, dict[str, Any]]:
    from _human_face_mesh import lame_parameters

    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    surface, original_ids = surface_original_ids(mesh)
    faces = triangle_faces(surface)
    rest_points = np.asarray(surface.points, dtype=np.float64)
    target_points = rest_points + target[original_ids]
    face_points = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[original_ids]
    face_cells = np.all(face_points[faces], axis=1)
    rest_area = triangle_areas(rest_points, faces)
    target_area = triangle_areas(target_points, faces)
    valid_rest_area = rest_area > 0.0
    area_ratio = np.ones_like(rest_area)
    area_ratio[valid_rest_area] = (
        target_area[valid_rest_area] / rest_area[valid_rest_area]
    )
    stretched_face_cells = face_cells & (area_ratio > 1.0)
    effective_area_ratio = np.ones_like(area_ratio)
    effective_area_ratio[stretched_face_cells] = area_ratio[stretched_face_cells]
    clipped_area_ratio = np.maximum(effective_area_ratio, cfg.area_ratio_floor)
    length_factor = np.sqrt(clipped_area_ratio)
    activation_diag = length_factor - 1.0

    activation_inv = np.zeros((surface.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = activation_diag
    activation_inv[:, 1] = activation_diag

    lambda_, mu = lame_parameters(SKIN_E, SKIN_NU)
    if GLOBAL_POINT_ID.vtk in mesh.point_data:
        mesh_point_ids = np.asarray(
            mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
        )
    else:
        mesh_point_ids = np.arange(mesh.n_points, dtype=np.int64)
    surface.point_data[GLOBAL_POINT_ID.vtk] = mesh_point_ids[original_ids]
    surface.cell_data[LAMBDA.vtk] = np.full(surface.n_cells, lambda_, dtype=np.float64)
    surface.cell_data[MU.vtk] = np.full(surface.n_cells, mu, dtype=np.float64)
    surface.cell_data[FRACTION.vtk] = np.ones(surface.n_cells, dtype=np.float64)
    surface.cell_data[ACTIVATION_INV.vtk] = activation_inv
    surface.cell_data["TargetRestAreaRatio"] = area_ratio
    surface.cell_data["EffectiveTargetRestAreaRatio"] = effective_area_ratio
    surface.cell_data["StressFreeAreaRatio"] = 1.0 / clipped_area_ratio
    surface.cell_data["TargetRestLengthFactor"] = length_factor
    surface.cell_data["RestArea"] = rest_area
    surface.cell_data["TargetArea"] = target_area
    surface.cell_data["IsFacePrestrainCell"] = face_cells.astype(np.int8)
    surface.cell_data["IsStretchedPrestrainCell"] = stretched_face_cells.astype(np.int8)

    metrics: dict[str, Any] = {
        "skin/surface_triangles": int(surface.n_cells),
        "skin/is_face_triangles": int(face_cells.sum()),
        "skin/stretched_is_face_triangles": int(stretched_face_cells.sum()),
        "skin/area_ratio_total": float(target_area.sum() / rest_area.sum()),
        "skin/is_face_area_ratio_total": float(
            target_area[face_cells].sum() / rest_area[face_cells].sum()
        ),
        "skin/effective_area_ratio_total": float(
            (effective_area_ratio * rest_area).sum() / rest_area.sum()
        ),
        "skin/degenerate_rest_area_cells": int((~valid_rest_area).sum()),
        "skin/raw_area_ratio_lt_one_cells": int((area_ratio < 1.0).sum()),
        "skin/raw_area_ratio_gt_one_cells": int((area_ratio > 1.0).sum()),
        "skin/masked_non_face_stretched_cells": int(
            ((~face_cells) & (area_ratio > 1.0)).sum()
        ),
        **stats("skin/area_ratio", area_ratio),
        **stats("skin/effective_area_ratio", effective_area_ratio),
        **stats("skin/activation_inv_diag", activation_diag),
    }
    return surface, metrics


def make_forward(
    mesh: pv.UnstructuredGrid, skin: pv.PolyData, cfg: EstimateConfig
) -> Any:
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_volume_material(
        mesh,
        E=APONEUROSIS_E,
        nu=APONEUROSIS_NU,
        fraction=np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="aponeurosis"))

    set_volume_material(
        mesh,
        E=FAT_E,
        nu=FAT_NU,
        fraction=np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="fat"))

    set_volume_material(
        mesh,
        E=MUSCLE_E,
        nu=MUSCLE_NU,
        fraction=np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))
    builder.add_potential(
        Koiter.from_pyvista(skin, name="skin", thickness=SKIN_THICKNESS)
    )

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.max_steps,
        atol=cfg.atol,
        rtol=cfg.rtol,
    )
    return forward


def run_forward(
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    activation_inv: np.ndarray,
    cfg: EstimateConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    forward = make_forward(mesh.copy(deep=True), skin.copy(deep=True), cfg)
    materials = forward.model.get_materials()
    activation_t = torch.as_tensor(
        activation_inv,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    for name in ("aponeurosis", "fat", "muscle"):
        materials[name]["activation_inv"] = activation_t
    forward.model.set_materials(materials)
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    return to_numpy(forward.state.u), forward_solution_metrics(solution)


def deformation_gradients(
    points: np.ndarray, displacement: np.ndarray, tets: np.ndarray
) -> np.ndarray:
    X = points[tets]
    x = X + displacement[tets]
    Dm = np.stack((X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2)
    Ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
    return Ds @ np.linalg.inv(Dm)


def estimate_volume_activation_inv(
    points: np.ndarray, displacement: np.ndarray, tets: np.ndarray, cfg: EstimateConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    F = deformation_gradients(points, displacement, tets)
    det_F = np.linalg.det(F)
    C = np.swapaxes(F, 1, 2) @ F
    eigvals, eigvecs = np.linalg.eigh(C)
    clipped = eigvals < cfg.stretch_floor**2
    stretches = np.sqrt(np.maximum(eigvals, cfg.stretch_floor**2))

    inv_stretches = 1.0 / stretches
    U = (eigvecs * stretches[:, None, :]) @ np.swapaxes(eigvecs, 1, 2)

    activation_inv = np.empty((tets.shape[0], 6), dtype=np.float64)
    activation_inv[:, 0] = U[:, 0, 0] - 1.0
    activation_inv[:, 1] = U[:, 1, 1] - 1.0
    activation_inv[:, 2] = U[:, 2, 2] - 1.0
    activation_inv[:, 3] = U[:, 0, 1]
    activation_inv[:, 4] = U[:, 1, 2]
    activation_inv[:, 5] = U[:, 0, 2]

    metrics: dict[str, Any] = {
        "volume/stretch_eigenvalues_clipped": int(clipped.sum()),
        **det_f_metrics("skin_only/detF", det_F),
        **stats("volume/stretch", stretches.ravel()),
        **stats("volume/implied_prestrain_stretch", inv_stretches.ravel()),
        **activation_inv_stats("volume/estimated_activation_inv", activation_inv),
    }
    return activation_inv, metrics


def write_outputs(
    mesh: pv.UnstructuredGrid,
    skin: pv.PolyData,
    target: np.ndarray,
    loss_mask: np.ndarray,
    skin_only: np.ndarray,
    compensated: np.ndarray,
    activation_inv: np.ndarray,
    cfg: EstimateConfig,
) -> None:
    from liblaf.apple.common import ACTIVATION_INV, GLOBAL_POINT_ID

    result = mesh.copy(deep=True)
    result.point_data["TargetDisplacement"] = target
    result.point_data["LossMask"] = loss_mask.astype(np.int8)
    result.point_data["SkinOnlyDisplacement"] = skin_only
    result.point_data["CompensatedDisplacement"] = compensated
    result.point_data["SkinOnlyPoint"] = result.points + skin_only
    result.point_data["CompensatedPoint"] = result.points + compensated
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["EstimatedActivationInvVol"] = activation_inv

    skin_inspect = skin.copy(deep=True)
    surface_ids = np.asarray(
        skin_inspect.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    for name, displacement in (
        ("SkinOnlyDisplacement", skin_only),
        ("CompensatedDisplacement", compensated),
        ("TargetDisplacement", target),
    ):
        skin_inspect.point_data[name] = displacement[surface_ids]
        skin_inspect.point_data[name.replace("Displacement", "Point")] = (
            skin_inspect.points + skin_inspect.point_data[name]
        )
    skin_activation = np.asarray(skin_inspect.cell_data[ACTIVATION_INV.vtk])
    raw_area_ratio = np.asarray(skin_inspect.cell_data["TargetRestAreaRatio"])
    effective_area_ratio = np.asarray(
        skin_inspect.cell_data["EffectiveTargetRestAreaRatio"]
    )
    skin_inspect.cell_data["SkinActivationInvDiag"] = skin_activation[:, 0]
    skin_inspect.cell_data["SkinActivationInvNorm"] = np.linalg.norm(
        skin_activation, axis=1
    )
    skin_inspect.cell_data["LogTargetRestAreaRatio"] = np.log(
        np.maximum(raw_area_ratio, np.finfo(np.float64).tiny)
    )
    skin_inspect.cell_data["LogEffectiveTargetRestAreaRatio"] = np.log(
        np.maximum(effective_area_ratio, np.finfo(np.float64).tiny)
    )
    skin_inspect.cell_data["SkinPrestrainOutlier"] = (
        np.abs(skin_activation[:, 0]) > 0.1
    ).astype(np.int8)

    melon.save(make_target_mesh(mesh, target, loss_mask), cfg.output_target)
    melon.save(skin, cfg.output_skin)
    melon.save(skin_inspect, cfg.output_skin_inspect)
    melon.save(result, cfg.output_result)
    cherries.log_output(cfg.output_target)
    cherries.log_output(cfg.output_skin)
    cherries.log_output(cfg.output_skin_inspect)
    cherries.log_output(cfg.output_result)


def run_estimate(cfg: EstimateConfig) -> dict[str, Any]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    configure_runtime()
    start = time.perf_counter()

    mesh, prep_summary = prepare_simulation_mesh(cfg)
    target, loss_mask, target_metrics = target_displacement(mesh, cfg.target_expression)
    skin, skin_metrics = make_area_skin(mesh, target, cfg)

    zero_activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    logger.info("Running skin-only forward solve")
    skin_only, skin_only_forward = run_forward(mesh, skin, zero_activation, cfg)

    tets = tetra_cells(mesh)
    logger.info("Estimating volume ActivationInv from skin-only deformation")
    estimated_activation, estimate_metrics = estimate_volume_activation_inv(
        np.asarray(mesh.points, dtype=np.float64), skin_only, tets, cfg
    )

    logger.info("Running compensated forward solve")
    compensated, compensated_forward = run_forward(
        mesh, skin, estimated_activation, cfg
    )
    compensated_det_f = np.linalg.det(
        deformation_gradients(
            np.asarray(mesh.points, dtype=np.float64), compensated, tets
        )
    )

    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    free = ~fixed
    surface_ids = np.unique(
        np.asarray(skin.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    )
    surface_mask = np.zeros(mesh.n_points, dtype=bool)
    surface_mask[surface_ids] = True

    skin_only_rms = float(np.linalg.norm(skin_only[free]) / math.sqrt(free.sum()))
    compensated_rms = float(np.linalg.norm(compensated[free]) / math.sqrt(free.sum()))
    summary: dict[str, Any] = {
        **prep_summary,
        "input_mesh": str(cfg.input_mesh),
        "output_result": str(cfg.output_result),
        "output_skin": str(cfg.output_skin),
        "output_skin_inspect": str(cfg.output_skin_inspect),
        "output_target": str(cfg.output_target),
        "elapsed_s": float(time.perf_counter() - start),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_free_points": int(free.sum()),
        "skin/thickness": float(SKIN_THICKNESS),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "volume/activation_inv_source": "polar_stretch_of_skin_only_deformation",
        "volume/implied_prestrain_source": (
            "inverse_polar_stretch_of_skin_only_deformation"
        ),
        "compensation/free_rms_ratio": float(compensated_rms / skin_only_rms)
        if skin_only_rms > 0.0
        else math.nan,
        **target_metrics,
        **skin_metrics,
        **estimate_metrics,
        **{f"skin_only/{key}": value for key, value in skin_only_forward.items()},
        **{f"compensated/{key}": value for key, value in compensated_forward.items()},
        **det_f_metrics("compensated/detF", compensated_det_f),
        **vector_stats("skin_only/displacement_all", skin_only),
        **vector_stats("skin_only/displacement_free", skin_only, free),
        **vector_stats("skin_only/displacement_surface", skin_only, surface_mask),
        **vector_stats("skin_only/displacement_target", skin_only, loss_mask),
        **surface_area_metrics("skin_only", skin, skin_only),
        **vector_stats("compensated/displacement_all", compensated),
        **vector_stats("compensated/displacement_free", compensated, free),
        **vector_stats("compensated/displacement_surface", compensated, surface_mask),
        **vector_stats("compensated/displacement_target", compensated, loss_mask),
        **surface_area_metrics("compensated", skin, compensated),
    }

    write_outputs(
        mesh,
        skin,
        target,
        loss_mask,
        skin_only,
        compensated,
        estimated_activation,
        cfg,
    )
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_summary)
    logger.info("Wrote %s", cfg.output_summary)
    return summary


if __name__ == "__main__":
    cherries.main(run_estimate)
