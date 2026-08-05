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
from _human_face_config import (
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING,
    SETUP_SKIN_ESTIMATED_PRESTRAIN,
    SMILE_LOSS_MASK,
    SMILE_TARGET,
    InverseCase,
    configure_runtime,
)
from _human_face_forward import build_forward
from _human_face_metrics import forward_solution_metrics, to_numpy
from _human_face_output import add_metric_fields
from _human_face_skin import filtered_isface_skin, triangle_area

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


class ForwardPrestrainConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input("10-human-face-prepared.vtu")
    output_mesh: Path = cherries.output(
        "15-estimated-skin-prestrain-forward.vtu", mkdir=True
    )
    output_target: Path = cherries.output(
        "15-estimated-skin-prestrain-target.vtu", mkdir=True
    )
    output_skin: Path = cherries.output(
        "15-estimated-skin-prestrain-skin.vtp", mkdir=True
    )
    output_skin_isface: Path = cherries.output(
        "15-estimated-skin-prestrain-isface-skin.vtp", mkdir=True
    )
    output_summary: Path = cherries.output(
        "15-estimated-skin-prestrain-forward-summary.json", mkdir=True
    )
    setup: str = SETUP_SKIN_ESTIMATED_PRESTRAIN
    area_ratio_floor: float = 0.1


def vector_stats(
    prefix: str, values: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, float]:
    selected = values if mask is None else values[mask]
    if selected.size == 0:
        return {
            f"{prefix}/rms": math.nan,
            f"{prefix}/max": math.nan,
            f"{prefix}/mean": math.nan,
        }
    norms = np.linalg.norm(selected, axis=1)
    return {
        f"{prefix}/rms": float(np.linalg.norm(selected) / math.sqrt(selected.shape[0])),
        f"{prefix}/max": float(norms.max()),
        f"{prefix}/mean": float(norms.mean()),
    }


def surface_area_metrics(
    skin: pv.PolyData, displacement: np.ndarray
) -> dict[str, float | int]:
    faces = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)
    triangles = faces[:, 1:]
    rest_points = np.asarray(skin.points, dtype=np.float64)
    deformed_points = rest_points + displacement
    rest_area = triangle_area(rest_points, triangles)
    deformed_area = triangle_area(deformed_points, triangles)
    is_face = np.asarray(skin.cell_data["IsFaceTriangle"], dtype=bool)
    active = np.asarray(skin.cell_data["ActivePrestrainMask"], dtype=bool)

    def total_ratio(mask: np.ndarray) -> float:
        if not np.any(mask):
            return math.nan
        denom = rest_area[mask].sum()
        if denom <= 0.0:
            return math.nan
        return float(deformed_area[mask].sum() / denom)

    return {
        "forward_area/all_deformed_rest_ratio": total_ratio(
            np.ones(skin.n_cells, dtype=bool)
        ),
        "forward_area/is_face_deformed_rest_ratio": total_ratio(is_face),
        "forward_area/active_deformed_rest_ratio": total_ratio(active),
        "forward_area/active_triangles": int(active.sum()),
    }


def make_target_mesh(
    mesh: pv.UnstructuredGrid,
) -> tuple[pv.UnstructuredGrid, np.ndarray, np.ndarray]:
    target = np.nan_to_num(
        np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    loss_mask = np.asarray(mesh.point_data[SMILE_LOSS_MASK], dtype=bool)
    target_mesh = mesh.copy(deep=True)
    target_mesh.point_data["TargetDisplacement"] = target
    target_mesh.point_data["LossMask"] = loss_mask.astype(np.int8)
    target_mesh.point_data["TargetPoint"] = target_mesh.points + target
    return target_mesh, target, loss_mask


def write_skin_outputs(
    *,
    skin: pv.PolyData,
    displacement: np.ndarray,
    target: np.ndarray,
    output_skin: Path,
    output_skin_isface: Path,
) -> None:
    from liblaf.apple.common import GLOBAL_POINT_ID

    skin_out = skin.copy(deep=True)
    point_ids = np.asarray(skin_out.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    skin_disp = displacement[point_ids]
    skin_target = target[point_ids]
    skin_out.point_data["SkinPrestrainDisplacement"] = skin_disp
    skin_out.point_data["SkinPrestrainPoint"] = skin_out.points + skin_disp
    skin_out.point_data["TargetDisplacement"] = skin_target
    skin_out.point_data["TargetPoint"] = skin_out.points + skin_target
    melon.save(skin_out, output_skin)
    melon.save(filtered_isface_skin(skin_out), output_skin_isface)


def run_forward(cfg: ForwardPrestrainConfig) -> dict[str, Any]:
    configure_runtime()
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    target_mesh, target, loss_mask = make_target_mesh(mesh)

    if cfg.setup not in {
        SETUP_SKIN_ESTIMATED_PRESTRAIN,
        SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING,
    }:
        msg = (
            f"setup must be {SETUP_SKIN_ESTIMATED_PRESTRAIN!r} or "
            f"{SETUP_SKIN_ESTIMATED_PLUS_TIGHTENING!r}, got {cfg.setup!r}"
        )
        raise ValueError(msg)

    case = InverseCase(
        target="smile",
        lr=0.03,
        setup=cfg.setup,  # pyright: ignore[reportArgumentType]
        label="forward-check",
    )
    forward, skin, skin_metrics = build_forward(
        mesh.copy(deep=True), case, area_ratio_floor=cfg.area_ratio_floor
    )
    if skin is None:
        msg = "estimated-prestrain case did not build a skin surface"
        raise RuntimeError(msg)

    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solution = forward.step()
    elapsed_s = time.perf_counter() - start
    displacement = to_numpy(forward.state.u)
    residual = displacement - target

    result = mesh.copy(deep=True)
    result.point_data["SkinPrestrainDisplacement"] = displacement
    result.point_data["Displacement"] = displacement
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["TargetPoint"] = result.points + target
    result.point_data["LossMask"] = loss_mask.astype(np.int8)
    result.point_data["DisplacementError"] = residual
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(residual, axis=1)

    metrics: dict[str, Any] = {
        "input_mesh": str(cfg.input_mesh),
        "output_mesh": str(cfg.output_mesh),
        "output_target": str(cfg.output_target),
        "output_skin": str(cfg.output_skin),
        "output_skin_isface": str(cfg.output_skin_isface),
        "case/setup": case.setup,
        "forward/elapsed_s": float(elapsed_s),
        "solver/forward": "PNCG",
        "solver/forward/rtol": float(FORWARD_RTOL),
        "solver/forward/atol": float(FORWARD_ATOL),
        "solver/forward/max_steps": int(FORWARD_MAX_STEPS),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "target/loss_points": int(loss_mask.sum()),
        **skin_metrics,
        **forward_solution_metrics(solution),
        **vector_stats("displacement/all", displacement),
        **vector_stats("displacement/loss_mask", displacement, loss_mask),
        **vector_stats("target/loss_mask", target, loss_mask),
        **vector_stats("residual/loss_mask", residual, loss_mask),
        **surface_area_metrics(
            skin,
            displacement[np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)],
        ),
    }
    numeric_metrics = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, int | float | bool)
    }
    add_metric_fields(result, numeric_metrics)

    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    melon.save(target_mesh, cfg.output_target)
    melon.save(result, cfg.output_mesh)
    write_skin_outputs(
        skin=skin,
        displacement=displacement,
        target=target,
        output_skin=cfg.output_skin,
        output_skin_isface=cfg.output_skin_isface,
    )
    cfg.output_summary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_target)
    cherries.log_output(cfg.output_mesh)
    cherries.log_output(cfg.output_skin)
    cherries.log_output(cfg.output_skin_isface)
    cherries.log_output(cfg.output_summary)
    logger.info("Wrote %s", cfg.output_mesh)
    logger.info("Wrote %s", cfg.output_skin)
    logger.info("Wrote %s", cfg.output_summary)
    return metrics


if __name__ == "__main__":
    cherries.main(run_forward)
