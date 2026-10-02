"""Behavior checks for a joint-descent constraint in the jaw proposal QP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "exp/2026/09/23/new-neutral/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from mouthopen_pose_projection import (  # noqa: E402
    PoseProjectionError,
    project_pose_descent_direction,
    project_pose_direction,
)


def test_geometry_projection_ascent_is_replaced_by_feasible_joint_descent() -> None:
    """Geometry-only nearest projection can turn a descent proposal uphill."""
    requested = np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    lower = np.array([1.0])
    gradient = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    geometry_only, _ = project_pose_direction(requested, matrix, lower)
    assert gradient @ geometry_only - 0.5 > 0

    direction, receipt = project_pose_descent_direction(
        requested,
        matrix,
        lower,
        pose_gradient=gradient,
        strain_directional=-0.5,
        descent_target=-0.25,
    )

    np.testing.assert_allclose(direction[:2], [1.0, -0.75], atol=1e-10)
    assert np.all(matrix @ direction >= lower - 1e-10)
    assert receipt["descent"]["projected_joint_directional"] <= -0.25 + 1e-10


def test_already_feasible_descent_proposal_is_unchanged() -> None:
    requested = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0])
    direction, receipt = project_pose_descent_direction(
        requested,
        np.eye(6)[:1],
        np.array([0.0]),
        pose_gradient=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        strain_directional=-0.5,
        descent_target=-0.25,
    )
    np.testing.assert_array_equal(direction, requested)
    assert not receipt["projection_applied"]


def test_incompatible_geometry_and_descent_constraints_fail_visibly() -> None:
    with pytest.raises(PoseProjectionError):
        project_pose_descent_direction(
            np.zeros(6),
            np.eye(6)[:1],
            np.array([1.0]),
            pose_gradient=np.eye(6)[0],
            strain_directional=0.0,
            descent_target=-0.25,
        )
