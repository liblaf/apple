"""Bound full rigid-pose increments without evaluating forward physics."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.spatial.transform import Rotation


def pose_waypoints(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_rotation_deg: float = 1.0,
    max_translation_m: float = 0.001,
) -> tuple[np.ndarray, dict]:
    """Interpolate rotation on SO(3) and translation of the shared pivot.

    Six coordinates are a world rotation vector in radians followed by world
    translation in metres, matching ``joint_equilibrium.rigid_displacement``.
    Limits apply to each adjacent pair, not the total fitted pose. This is a
    geometry schedule only; each future solve must independently converge.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    assert source.shape == target.shape == (6,)
    assert np.isfinite(source).all()
    assert np.isfinite(target).all()
    assert max_rotation_deg > 0
    assert max_translation_m > 0
    old = Rotation.from_rotvec(source[:3])
    new = Rotation.from_rotvec(target[:3])
    delta = (new * old.inv()).as_rotvec()
    angle = float(np.degrees(np.linalg.norm(delta)))
    translation = float(np.linalg.norm(target[3:] - source[3:]))
    count = max(
        math.ceil(angle / max_rotation_deg),
        math.ceil(translation / max_translation_m),
    )
    poses = [source.copy()]
    for i in range(1, count + 1):
        fraction = i / count
        rotation = (Rotation.from_rotvec(fraction * delta) * old).as_rotvec()
        shift = source[3:] + fraction * (target[3:] - source[3:])
        poses.append(np.r_[rotation, shift])
    poses[-1] = target.copy()
    increments = []
    for index, (before, after) in enumerate(itertools.pairwise(poses), 1):
        rotation = float(
            np.degrees(
                (
                    Rotation.from_rotvec(after[:3])
                    * Rotation.from_rotvec(before[:3]).inv()
                ).magnitude()
            )
        )
        displacement = float(np.linalg.norm(after[3:] - before[3:]))
        assert rotation <= max_rotation_deg + 1e-10
        assert displacement <= max_translation_m + 1e-12
        increments.append(
            {"step": index, "rotation_deg": rotation, "translation_m": displacement}
        )
    return np.asarray(poses), {
        "scope": "Proposed geometric schedule; no forward solve performed",
        "max_rotation_deg": max_rotation_deg,
        "max_translation_m": max_translation_m,
        "total_relative_rotation_deg": angle,
        "total_translation_change_m": translation,
        "steps": increments,
    }
