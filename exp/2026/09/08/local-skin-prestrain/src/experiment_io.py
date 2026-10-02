# ruff: noqa: FBT003, PLR0915, PT018
"""State and provenance utilities for the fixed local-skin ablation."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from local_physics import FacePhysics

ROOT = Path(__file__).resolve().parents[6]


GROUP = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT
FIXTURE = (
    ORIGINAL / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
LOG = logging.getLogger(__name__)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def activation(q: np.ndarray) -> np.ndarray:
    A = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    A[:, 0, 0] += q[:, 0]
    A[:, 1, 1] += q[:, 1]
    A[:, 2, 2] += q[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = q[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = q[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = q[:, 5]
    return A


def save_state(path: Path, physics: FacePhysics, state: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        q=state["q"],
        Ainv=activation(state["q"]),
        u=state["u"],
        rest_points=physics.points,
        active_ids=physics.ids,
        step=np.array(state["step"]),
        solver_valid=np.array(state["solver_valid"]),
        physical_volume_energy=np.array(True),
    )


def archive_runtime(out: Path) -> dict[str, Any]:
    records = {}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or not source.endswith(".py"):
            continue
        p = Path(source).resolve()
        local = p.parent == GROUP / "src"
        if not (local or name.startswith(("liblaf.apple", "liblaf.peach"))):
            continue
        relative = (
            Path("experiment") / p.name
            if local
            else Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        )
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        records[name] = {"path": str(p), "sha256": digest(p), "snapshot": str(target)}
    return records


def row_metrics(physics: FacePhysics, u: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    pred, target = u[physics.top], physics.target[physics.top]
    error = pred - target
    area = physics.weights
    J = physics.detf(u)
    actJ = J[physics.ids]
    volume_weights = physics.volumes
    eig = np.linalg.eigvalsh(activation(q))
    tet = physics.tets[27306]
    rest = physics.points[tet]
    current = rest + u[tet]
    F = (current[1:] - current[:1]).T @ np.linalg.inv((rest[1:] - rest[:1]).T)
    stretches = np.linalg.svd(F, compute_uv=False)
    return {
        "fit_rms_mm": float(1000 * np.linalg.norm(error) / math.sqrt(len(error))),
        "motion_rms_mm": float(1000 * np.linalg.norm(pred) / math.sqrt(len(pred))),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(area * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000 * np.sqrt(np.sum(area * np.sum(pred**2, axis=1)))
        ),
        "target_projection": float(np.sum(pred * target) / np.sum(target**2)),
        "detF_min": float(J.min()),
        "detF_max": float(J.max()),
        "inverted_all_cells": int((J <= 0).sum()),
        "inverted_active_cells": int((actJ <= 0).sum()),
        "active_volume_weighted_RMS_detF_minus_1": float(
            np.sqrt(np.average((actJ - 1) ** 2, weights=volume_weights))
        ),
        "A_eigen_min": float(eig.min()),
        "A_eigen_max": float(eig.max()),
        "non_spd_active_cells": int((eig[:, 0] <= 0).sum()),
        "cell27306_detF": float(np.linalg.det(F)),
        "cell27306_stretch_max": float(stretches[0]),
        "cell27306_stretch_middle": float(stretches[1]),
        "cell27306_stretch_min": float(stretches[2]),
    }
