"""Geometry-only chin-patch jaw-angle seed for saved blendshape targets."""
# ruff: noqa: PT018

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_array
from scipy.sparse.csgraph import dijkstra


def _anchor(points: np.ndarray, center_x: float) -> np.ndarray:
    """Lower, central, anterior landmark in the +z front-view convention."""
    x, y, z = points.T
    central = np.abs(x - center_x) <= np.quantile(np.abs(x - center_x), 0.35)
    lower = y <= np.quantile(y, 0.35)
    anterior = z >= np.quantile(z, 0.75)
    chosen = points[central & lower & anterior]
    assert len(chosen) >= 8
    return np.median(chosen, axis=0)


def _weights(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri = points[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    w = np.zeros(len(points))
    np.add.at(w, triangles.ravel(), np.repeat(area / 3, 3))
    return w


def _rotate(
    points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    q = points - pivot
    c, s = np.cos(angle), np.sin(angle)
    return pivot + c * q + s * np.cross(axis, q) + (1 - c) * np.outer(q @ axis, axis)


def estimate_chin_pose(
    neutral_points_m: np.ndarray,
    target_points_m: np.ndarray,
    skin_triangles: np.ndarray,
    hinge_pivot_m: np.ndarray,
    hinge_axis_m: np.ndarray,
    full_mandible_points_m: np.ndarray | None = None,
    patch_radius_m: float = 0.018,
    angle_bounds_rad: tuple[float, float] = (-0.8, 0.8),
) -> dict[str, Any]:
    """Fit one rigid hinge angle to an area-weighted observable chin patch.

    The returned angle is only a seed: facial expression targets are not rigid
    mandible observations and no contact, force, or geometry validity follows.
    """
    neutral = np.asarray(neutral_points_m, float)
    target = np.asarray(target_points_m, float)
    tri = np.asarray(skin_triangles, int)
    assert (
        neutral.shape == target.shape
        and neutral.ndim == 2
        and neutral.shape[1] == 3
        and np.isfinite(neutral).all()
        and np.isfinite(target).all()
    )
    assert (
        tri.ndim == 2
        and tri.shape[1] == 3
        and tri.min() >= 0
        and tri.max() < len(neutral)
    )
    pivot = np.asarray(hinge_pivot_m, float)
    axis = np.asarray(hinge_axis_m, float)
    assert pivot.shape == (3,) and axis.shape == (3,) and np.linalg.norm(axis) > 0
    landmark = _anchor(
        neutral
        if full_mandible_points_m is None
        else np.asarray(full_mandible_points_m, float),
        float(pivot[0]),
    )
    distance = np.linalg.norm(neutral - landmark, axis=1)
    nearby = np.flatnonzero(distance <= patch_radius_m)
    assert len(nearby) >= 12, (len(nearby), patch_radius_m, landmark)
    seed = int(nearby[np.argmax(neutral[nearby, 2])])
    edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    lengths = np.linalg.norm(neutral[edges[:, 0]] - neutral[edges[:, 1]], axis=1)
    graph = coo_array((lengths, (edges[:, 0], edges[:, 1])), shape=(len(neutral),) * 2)
    graph = graph + graph.T
    geodesic = dijkstra(graph.tocsr(), indices=seed)
    patch = np.flatnonzero(geodesic <= 0.012)
    assert len(patch) >= 12, (len(patch), patch_radius_m, landmark)
    weights = _weights(neutral, tri)[patch]
    assert np.all(weights > 0)

    def loss(angle: float) -> float:
        residual = _rotate(neutral[patch], pivot, axis, angle) - target[patch]
        return float(np.sum(weights * np.sum(residual**2, axis=1)) / np.sum(weights))

    result = minimize_scalar(
        loss, bounds=angle_bounds_rad, method="bounded", options={"xatol": 1e-10}
    )
    assert result.success, result.message
    residual = _rotate(neutral[patch], pivot, axis, float(result.x)) - target[patch]
    return {
        "schema": "chin-pose-seed-v1",
        "angle_rad": float(result.x),
        "angle_deg": float(np.degrees(result.x)),
        "patch_local_ids": patch.tolist(),
        "criteria": {
            "front_axis": "+z",
            "vertical_axis": "y",
            "anchor": "lower central anterior mandible when supplied; otherwise skin",
            "patch_radius_m": patch_radius_m,
            "patch_geodesic_radius_m": 0.012,
            "seed_local_id": seed,
            "angle_bounds_rad": list(angle_bounds_rad),
            "area_weighted": True,
        },
        "anchor_m": landmark.tolist(),
        "patch_vertices": len(patch),
        "residual_rms_m": float(
            np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)) / np.sum(weights))
        ),
        "residual_max_m": float(np.linalg.norm(residual, axis=1).max()),
    }
