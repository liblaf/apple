# Copyright (c) 2026 liblaf
"""CPU checks for the corresponding-skin and within-muscle face objective."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    str(Path(__file__).parents[2] / "exp/2026/09/30/collision-off-expressions/src"),
)

from face_shape_activation_objective import (
    ActivationSmoothness,
    FaceShapeActivationObjective,
    ObjectiveWeights,
    SkinShapeLoss,
    calibrate_smooth_weight,
    normal_anchor_coefficient,
    normal_pose_gradient_ratio,
    raw6_dual_volume_norm,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.array(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float64,
    )
    tets = np.array(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64)
    triangles = np.array(((0, 1, 2), (0, 3, 1)), dtype=np.int64)
    target = points.copy()
    target[0] += (0.03, 0.02, 0.01)
    target[3] += (0.02, -0.04, 0.01)
    target[4] += (0.0, 0.02, -0.03)
    return points, tets, triangles, target


def toy_objective() -> FaceShapeActivationObjective:
    points, tets, triangles, target = fixture()
    skin = SkinShapeLoss(
        points, points, target, np.arange(len(points)), triangles, device="cpu"
    )
    graph = ActivationSmoothness(
        points,
        tets,
        np.array((0, 1)),
        np.array((0.5, 0.75)),
        np.array((7, 7)),
        device="cpu",
    )
    assert graph.edge_count == 1
    assert graph.length_selection == "geometry-derived neighbor normalization"
    return FaceShapeActivationObjective(
        skin, graph, ObjectiveWeights(normal=3.0, smooth=0.2)
    )


def test_corresponding_skin_and_raw6_graph_have_cpu_autograd_derivatives() -> None:
    objective = toy_objective()
    u = torch.tensor(
        ((0.01, -0.01, 0), (0, 0.01, 0), (0, 0, -0.01), (0.01, 0, 0), (0, 0.02, 0)),
        dtype=torch.float64,
        requires_grad=True,
    )
    q = torch.tensor(
        ((0.1, 0, 0, 0.2, 0, 0), (0, 0.3, 0, 0, -0.1, 0.2)),
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(objective, (u, q), eps=1e-6, atol=2e-6, rtol=1e-4)
    assert torch.autograd.gradcheck(
        lambda value: objective.skin.components(value)[1],
        (u,),
        eps=1e-6,
        atol=2e-6,
        rtol=1e-4,
    )
    metrics = objective.metrics(u.detach(), q.detach())
    assert metrics["objective"] == pytest.approx(float(objective(u, q).detach()))
    assert metrics["normal_angle_rms_deg"] > 0
    assert metrics["activation_smoothness"] > 0


def test_graph_normalization_and_muscle_boundary() -> None:
    points, tets, _, _ = fixture()
    graph = ActivationSmoothness(
        points,
        tets,
        np.array((0, 1)),
        np.array((0.5, 0.75)),
        np.array((7, 7)),
        device="cpu",
    )
    q = torch.tensor(((0, 0, 0, 0, 0, 0), (1, 2, 3, 4, 5, 6)), dtype=torch.float64)
    assert float(graph(q)) == pytest.approx(168.0)
    assert float(graph(q[:1].expand(2, -1))) == 0
    assert graph.contract()["same_muscle_edge_count"] == 1
    with pytest.raises(AssertionError, match="no same-muscle"):
        ActivationSmoothness(
            points,
            tets,
            np.array((0, 1)),
            np.array((0.5, 0.75)),
            np.array((7, 8)),
            device="cpu",
        )


def test_collapse_is_rejected_and_normal_alignment_keeps_correspondence() -> None:
    points, _, triangles, target = fixture()
    collapsed = target.copy()
    collapsed[2] = collapsed[0] + 0.5 * (collapsed[1] - collapsed[0])
    with pytest.raises(AssertionError, match="collapsed skin triangle"):
        SkinShapeLoss(
            points, points, collapsed, np.arange(len(points)), triangles, device="cpu"
        )
    skin = toy_objective().skin
    u = torch.as_tensor(target - points)
    position, normal = skin.components(u)
    assert float(position) == pytest.approx(0, abs=1e-28)
    assert float(normal) == pytest.approx(0, abs=1e-28)


def test_calibration_uses_combined_data_gradient_and_physical_raw6_dual_norm() -> None:
    mass = torch.tensor((0.25, 0.75), dtype=torch.float64)
    position = torch.tensor(
        ((1, 0, 0, 2, 0, 0), (-1, 0, 0, 0, 0, 0)), dtype=torch.float64
    )
    normal = 2 * position
    smooth = 4 * position
    beta = normal_anchor_coefficient(4e-5)
    calibration = calibrate_smooth_weight(
        position, normal, smooth, mass, normal_coefficient=beta, smooth_target_ratio=0.1
    )
    assert raw6_dual_volume_norm(position, mass) == pytest.approx(
        np.sqrt(3 / 0.25 + 1 / 0.75)
    )
    assert calibration["smooth_coefficient"] == pytest.approx(0.1 * (1 + 2 * beta) / 4)
    assert calibration["weighted_normal_to_position_gradient_ratio"] == pytest.approx(
        2 * beta
    )
    assert normal_pose_gradient_ratio(
        torch.ones(6, dtype=torch.float64),
        2 * torch.ones(6, dtype=torch.float64),
        normal_coefficient=beta,
    ) == pytest.approx(2 * beta)
