from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pyvista as pv
from _human_face_case import solve_case
from _human_face_config import (
    APONEUROSIS_E,
    APONEUROSIS_NU,
    FAT_E,
    FAT_NU,
    MUSCLE_E,
    MUSCLE_NU,
    SKIN_E,
    SKIN_NU,
    SKIN_THICKNESS,
    InverseConfig,
    PrepareConfig,
    configure_runtime,
    selected_cases,
)
from _human_face_mesh import (
    add_required_fields,
    extract_simulation_mesh,
    geometry_summary,
    orient_tetra_mesh,
)
from _human_face_output import write_table
from _human_face_skin import filtered_isface_skin, skin_prestrain_fields

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


def prepare_mesh(cfg: PrepareConfig) -> None:
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh, subset_summary = extract_simulation_mesh(mesh)
    mesh, n_flipped = orient_tetra_mesh(mesh)
    field_summary = add_required_fields(mesh)
    skin_surface, skin_metrics = skin_prestrain_fields(
        mesh, area_ratio_floor=cfg.area_ratio_floor
    )
    skin_prestrain = filtered_isface_skin(skin_surface)
    skin_plus_tightening_surface, skin_plus_tightening_metrics = skin_prestrain_fields(
        mesh,
        area_ratio_floor=cfg.area_ratio_floor,
        constant_tightening=cfg.extra_tightening,
    )
    skin_plus_tightening = filtered_isface_skin(skin_plus_tightening_surface)
    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, cfg.output_mesh)
    melon.save(skin_prestrain, cfg.output_skin_prestrain)
    melon.save(skin_plus_tightening, cfg.output_skin_plus_tightening)

    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    summary: dict[str, Any] = {
        "source_mesh": str(cfg.input_mesh),
        "mesh": str(cfg.output_mesh),
        "n_oriented_tets_flipped": int(n_flipped),
        **subset_summary,
        "n_active_tets": int(active_mask.sum()),
        "n_activation_parameter_dofs": int(active_mask.sum() * 6),
        "fat/E_MPa": float(FAT_E),
        "fat/nu": float(FAT_NU),
        "muscle/E_MPa": float(MUSCLE_E),
        "muscle/nu": float(MUSCLE_NU),
        "aponeurosis/E_MPa": float(APONEUROSIS_E),
        "aponeurosis/nu": float(APONEUROSIS_NU),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/thickness": float(SKIN_THICKNESS),
        "skin/prestrain_variants": [
            "skin-no-prestrain",
            "no-skin",
            "skin-estimated-prestrain",
            "skin-estimated-plus-tightening",
        ],
        "skin/prestrain_vtp": str(cfg.output_skin_prestrain),
        "skin/plus_tightening_vtp": str(cfg.output_skin_plus_tightening),
        **skin_metrics,
        **{
            f"plus_tightening/{key}": value
            for key, value in skin_plus_tightening_metrics.items()
        },
        **field_summary,
        **geometry_summary(mesh),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_mesh)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_skin_prestrain)
    cherries.log_output(cfg.output_skin_plus_tightening)
    logger.info("Wrote %s", cfg.output_mesh)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_skin_prestrain)
    logger.info("Wrote %s", cfg.output_skin_plus_tightening)


def run_inverse(cfg: InverseConfig) -> None:
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    rows = [solve_case(case, mesh, cfg) for case in selected_cases(cfg)]
    summary = {
        "complete": all(row["inverse/converged"] for row in rows),
        "cases": rows,
        "target/requested": cfg.target,
        "case_set/requested": cfg.case_set,
        "inverse/lr": float(cfg.inverse_lr),
        "loss/scale": float(cfg.loss_scale),
        "optimizer/adam_eps": float(cfg.adam_eps),
        "inverse/max_steps": int(cfg.inverse_max_steps),
        "baseline/mandatory_optimizer_steps": int(cfg.mandatory_baseline_steps),
        "segment_steps": int(cfg.segment_steps),
        "diagnostic_min_delta_rel": float(cfg.diagnostic_min_delta_rel),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_table(cfg.output_table, rows)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_table)

    failed = [row["case"] for row in rows if not row["inverse/converged"]]
    if failed and cfg.require_convergence:
        msg = (
            "inverse cases did not satisfy the adaptive convergence gate: "
            + ", ".join(failed)
        )
        raise RuntimeError(msg)
