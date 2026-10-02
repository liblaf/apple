# ruff: noqa: PT018
"""Area-weighted rigid pose fit for the saved connected chin patch."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _forward(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    return (points - pivot) @ rotation.T + pivot + pose[3:]


def estimate_rigid_chin_pose(
    neutral_points_m: np.ndarray,
    target_points_m: np.ndarray,
    skin_triangles: np.ndarray,
    patch_local_ids: np.ndarray,
    mandible_pivot_m: np.ndarray,
) -> dict[str, Any]:
    """Fit an unrestricted proper rigid transform in solver pose convention."""
    neutral = np.asarray(neutral_points_m, dtype=np.float64)
    target = np.asarray(target_points_m, dtype=np.float64)
    triangles = np.asarray(skin_triangles, dtype=np.int64)
    patch = np.asarray(patch_local_ids, dtype=np.int64)
    pivot = np.asarray(mandible_pivot_m, dtype=np.float64)
    assert neutral.shape == target.shape and neutral.ndim == 2 and neutral.shape[1] == 3
    assert triangles.ndim == 2 and triangles.shape[1] == 3
    assert patch.ndim == 1 and len(patch) >= 3
    assert patch.min() >= 0 and patch.max() < len(neutral)
    assert pivot.shape == (3,)
    assert np.isfinite(neutral).all() and np.isfinite(target).all()
    assert np.isfinite(pivot).all()

    tri = neutral[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    full_weights = np.zeros(len(neutral), dtype=np.float64)
    np.add.at(full_weights, triangles.ravel(), np.repeat(area / 3.0, 3))
    weights = full_weights[patch]
    assert np.all(weights > 0) and np.isfinite(weights).all()
    weights /= weights.sum()
    x, y = neutral[patch], target[patch]
    x_bar, y_bar = weights @ x, weights @ y
    x_centered, y_centered = x - x_bar, y - y_bar
    covariance = (x_centered * weights[:, None]).T @ y_centered
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(right_t.T @ left.T)
    rotation = right_t.T @ correction @ left.T
    assert np.linalg.det(rotation) > 0
    translation = y_bar - (x_bar - pivot) @ rotation.T - pivot
    pose = np.concatenate((Rotation.from_matrix(rotation).as_rotvec(), translation))
    fitted = _forward(x, pivot, pose)
    before = x - y
    residual = fitted - y
    condition = float(np.inf if singular[-1] == 0 else singular[0] / singular[-1])
    return {
        "schema": "chin-rigid-pose-estimate-v1",
        "pose_rad_m": pose.tolist(),
        "rotation_vector_rad": pose[:3].tolist(),
        "translation_m": pose[3:].tolist(),
        "mandible_pivot_m": pivot.tolist(),
        "fit_rotation_degrees": float(np.degrees(np.linalg.norm(pose[:3]))),
        "translation_norm_m": float(np.linalg.norm(pose[3:])),
        "patch_local_ids": patch.tolist(),
        "patch_vertices": len(patch),
        "area_weight_sum_m2": float(full_weights[patch].sum()),
        "weighted_rms_before_m": float(
            np.sqrt(np.sum(weights * np.sum(before**2, axis=1)))
        ),
        "weighted_rms_after_m": float(
            np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)))
        ),
        "residual_max_m": float(np.linalg.norm(residual, axis=1).max()),
        "det_rotation": float(np.linalg.det(rotation)),
        "svd_singular_values_m2": singular.tolist(),
        "svd_condition": condition,
    }


def validate_rigid_pose_fit() -> dict[str, float]:
    """Verify Kabsch and solver-pose conversion for a large-pivot SE(3) case."""
    rng = np.random.default_rng(20260923)
    points = rng.normal(size=(27, 3)) * np.array((0.03, 0.02, 0.01))
    pivot = np.array((17.0, -11.0, 5.0))
    r0 = Rotation.from_rotvec((0.31, -0.17, 0.23)).as_matrix()
    r1 = Rotation.from_rotvec((-0.09, 0.28, 0.13)).as_matrix()
    rotation = r1 @ r0
    expected = np.concatenate(
        (Rotation.from_matrix(rotation).as_rotvec(), (0.014, -0.008, 0.011))
    )
    target = _forward(points, pivot, expected)
    triangles = np.column_stack((np.zeros(25, int), np.arange(1, 26), np.arange(2, 27)))
    receipt = estimate_rigid_chin_pose(points, target, triangles, np.arange(27), pivot)
    recovered = np.asarray(receipt["pose_rad_m"])
    forward_error = float(np.abs(_forward(points, pivot, recovered) - target).max())
    return {
        "noncommuting_rotation_forward_max_error_m": forward_error,
        "rotation_vector_error_rad": float(
            np.linalg.norm(recovered[:3] - expected[:3])
        ),
        "translation_error_m": float(np.linalg.norm(recovered[3:] - expected[3:])),
        "det_rotation": float(receipt["det_rotation"]),
        "large_pivot_norm_m": float(np.linalg.norm(pivot)),
    }


__all__ = ["estimate_rigid_chin_pose", "validate_rigid_pose_fit"]
