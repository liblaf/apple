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

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


def prepare_mesh(cfg: PrepareConfig) -> None:
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh, subset_summary = extract_simulation_mesh(mesh)
    mesh, n_flipped = orient_tetra_mesh(mesh)
    field_summary = add_required_fields(mesh)
    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, cfg.output_mesh)

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
        "skin/prestrain_variants": [0.0, 0.05, 0.10],
        **field_summary,
        **geometry_summary(mesh),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_mesh)
    cherries.log_output(cfg.output_summary)
    logger.info("Wrote %s", cfg.output_mesh)
    logger.info("Wrote %s", cfg.output_summary)


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
        "inverse/max_steps": int(cfg.inverse_max_steps),
        "inverse/loss_min_delta": float(cfg.inverse_loss_min_delta),
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
        msg = "inverse cases did not hit the 20-step loss plateau: " + ", ".join(failed)
        raise RuntimeError(msg)
