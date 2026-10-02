"""Learned uniaxial contraction with a fixed same-muscle tensor prior."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from activation_model import pack, packed_smoothness

ROOT = Path(__file__).resolve().parents[6]
BASE = ROOT / "exp/2026/09/19/gradient-only-face"
sys.path.insert(0, str(BASE / "src"))
from face_study import FIXTURE as FIXTURE  # noqa: E402, PLC0414
from face_study import LEGACY as LEGACY  # noqa: E402, PLC0414
from face_study import FaceStudy, activation  # noqa: E402


def symmetric_gradient(g: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(g), 3, 3), dtype=g.dtype)
    matrix[:, 0, 0] = g[:, 0]
    matrix[:, 1, 1] = g[:, 1]
    matrix[:, 2, 2] = g[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = g[:, 3] / 2
    matrix[:, 1, 2] = matrix[:, 2, 1] = g[:, 4] / 2
    matrix[:, 0, 2] = matrix[:, 2, 0] = g[:, 5] / 2
    return matrix


class Study(FaceStudy):
    def __init__(self, data_loss: str, data_scale: float, length: float) -> None:
        super().__init__()
        assert data_loss in {"gradient", "l2"}
        self.data_loss = data_loss
        self.data_scale = data_scale
        i, j, weight = self.physics.graph
        assert np.all(self.physics.region[i] == self.physics.region[j])
        self.edges = torch.as_tensor(np.column_stack((i, j)), dtype=torch.long)
        self.edge_weights = torch.as_tensor(weight)
        self.smooth_factor = length**2 * weight.sum() / self.physics.volumes.sum()
        self.length = length

    def regularizer(self, q: torch.Tensor) -> torch.Tensor:
        return self.smooth_factor * packed_smoothness(q, self.edges, self.edge_weights)

    def evaluate_axis(
        self,
        strength: torch.Tensor,
        axis: torch.Tensor,
        seed: np.ndarray,
        coefficient: float,
        *,
        backward: bool,
    ) -> dict:
        if backward:
            strength.grad = None
            axis.grad = None
        q = pack(strength, axis)
        if backward:
            q.retain_grad()
        u = self.physics.solve(q, seed)
        forward = dict(self.physics.last_forward)
        assert forward["success"], forward
        l2, gradient = self.losses(u)
        data = self.data_scale * (gradient if self.data_loss == "gradient" else l2)
        regularizer = self.regularizer(q)
        objective = data + coefficient * regularizer
        assert torch.isfinite(objective)
        result = {
            "objective": float(objective.detach()),
            "data_objective": float(data.detach()),
            "activation_smoothness": float(regularizer.detach()),
            "regularizer_contribution": float(coefficient * regularizer.detach()),
            "position_loss_component_mm2": float(l2.detach()),
            "surface_gradient_loss": float(gradient.detach()),
            "u": u.detach().cpu().numpy().copy(),
            "q": q.detach().cpu().numpy().copy(),
            "forward": forward,
        }
        if backward:
            objective.backward()
            result["adjoint"] = self.physics.check_adjoint()
            for key, variable in (
                ("gradient", q),
                ("strength_gradient", strength),
                ("axis_gradient", axis),
            ):
                assert variable.grad is not None
                assert torch.isfinite(variable.grad).all()
                result[key] = variable.grad.detach().cpu().numpy().copy()
        return result

    def axis_metrics(
        self,
        strength: np.ndarray,
        axis: np.ndarray,
        result: dict,
        initial_axes: np.ndarray,
    ) -> dict:
        row = self.metrics(result["q"], result)
        C = activation(result["q"]) - np.eye(3)
        G = symmetric_gradient(result["gradient"])
        values, vectors = np.linalg.eigh(C - G)
        n = vectors[:, :, -1]
        projected = (
            np.maximum(values[:, -1], 0)[:, None, None] * n[:, :, None] * n[:, None, :]
        )
        mapping2 = np.sum((C - projected) ** 2, axis=(1, 2))
        physical_pg = float(np.sqrt(self.active_weights @ mapping2))
        angle = np.degrees(
            np.arccos(np.clip(np.abs(np.sum(axis * initial_axes, axis=1)), 0, 1))
        )
        unit_error = float(np.max(np.abs(np.linalg.norm(axis, axis=1) - 1)))
        rank_error = float(np.max(np.abs(np.linalg.eigvalsh(C)[:, :2])))
        assert strength.min() >= 0
        assert unit_error < 1e-12
        assert rank_error < 1e-10
        return {
            **row,
            "data_objective": result["data_objective"],
            "activation_smoothness": result["activation_smoothness"],
            "regularizer_contribution": result["regularizer_contribution"],
            "projected_gradient_rms": physical_pg,
            "projected_gradient_max": float(np.sqrt(mapping2.max())),
            "strength_max": float(strength.max()),
            "strength_mean_active_volume": float(self.active_weights @ strength),
            "active_axial_stretch_min": float(1 / (1 + strength.max())),
            "strength_positive_cells": int(np.count_nonzero(strength > 0)),
            "axis_rotation_rms_deg": float(np.sqrt(self.active_weights @ angle**2)),
            "axis_unit_error": unit_error,
            "rank_one_error": rank_error,
        }
