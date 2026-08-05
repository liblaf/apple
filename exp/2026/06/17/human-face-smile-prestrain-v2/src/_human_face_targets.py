from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyvista as pv
from _human_face_config import (
    IS_FIXED,
    SMILE_LOSS_MASK,
    SMILE_TARGET,
    InverseCase,
    InverseConfig,
)


def target_displacement_and_mask(
    mesh: pv.UnstructuredGrid, case: InverseCase, _cfg: InverseConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if case.target == "smile":
        target = np.nan_to_num(
            np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mask = np.asarray(mesh.point_data[SMILE_LOSS_MASK], dtype=bool)
        extra_metrics: dict[str, Any] = {}
    else:
        msg = f"unknown target {case.target!r}; expected 'smile'"
        raise ValueError(msg)
    if not np.any(mask):
        msg = f"{case.target} selected no loss points"
        raise ValueError(msg)
    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    target_norm = np.linalg.norm(target[mask], axis=1)
    return (
        target,
        mask,
        {
            "target/name": case.target,
            "target/loss_points": int(mask.sum()),
            "target/fixed_overlap_points": int((mask & fixed).sum()),
            "target/displacement_rms": float(
                np.linalg.norm(target[mask]) / math.sqrt(mask.sum())
            ),
            "target/displacement_max": float(target_norm.max()),
            **extra_metrics,
        },
    )


def make_target_mesh(
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    mask: np.ndarray,
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    result.point_data["TargetDisplacement"] = target
    result.point_data["LossMask"] = mask.astype(np.int8)
    result.point_data["TargetPoint"] = result.points + target
    return result
