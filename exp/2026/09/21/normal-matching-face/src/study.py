"""Neutral Raw6 face fitting with positional, normal and tensor losses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from surface_normal import SurfaceNormalLoss

ROOT = Path(__file__).resolve().parents[6]
BASE = ROOT / "exp/2026/09/19/gradient-only-face/src"
sys.path.insert(0, str(BASE))
from face_study import FIXTURE as FIXTURE  # noqa: E402, PLC0414
from face_study import LEGACY as LEGACY  # noqa: E402, PLC0414
from face_study import FaceStudy  # noqa: E402


class Study(FaceStudy):
    def __init__(self, length: float = 0.005) -> None:
        super().__init__()
        p = self.physics
        self.normal_loss = SurfaceNormalLoss(
            p.points[self.skin_ids],
            self.triangles,
            p.target[self.skin_ids],
            device=self.target.device,
            dtype=self.target.dtype,
        )
        i, j, weights = p.graph
        assert np.all(p.region[i] == p.region[j])
        self.edge_i = torch.as_tensor(i, dtype=torch.long)
        self.edge_j = torch.as_tensor(j, dtype=torch.long)
        self.conductance = torch.as_tensor(weights)
        self.regularizer_factor = length**2 / p.volumes.sum()
        self.length = length
        with torch.no_grad():
            zero = torch.zeros_like(self.target)
            l20 = float(
                (self.weights_t[:, None] * self.target.square()).sum() * (1e6 / 3)
            )
            n0 = float(self.normal_loss(zero))
        assert l20 > 0
        assert n0 > 0
        self.normalization = {"L20": l20, "N0": n0}

    def regularizer(self, q: torch.Tensor) -> torch.Tensor:
        difference = q[self.edge_i] - q[self.edge_j]
        frobenius2 = difference[:, :3].square().sum(-1) + 2 * difference[
            :, 3:
        ].square().sum(-1)
        return self.regularizer_factor * (self.conductance * frobenius2).sum()

    def evaluate_normal(
        self,
        q: torch.Tensor,
        seed: np.ndarray,
        beta: float,
        alpha: float,
        *,
        backward: bool,
        normal_only: bool = False,
    ) -> dict:
        if backward:
            q.grad = None
        u = self.physics.solve(q, seed)
        forward = dict(self.physics.last_forward)
        assert forward["success"], forward
        l2, surface_gradient = self.losses(u)
        normal = self.normal_loss(u[self.skin_ids_t])
        regularizer = self.regularizer(q)
        normal_coefficient = beta * self.normalization["L20"] / self.normalization["N0"]
        smooth_coefficient = alpha
        position_contribution = torch.zeros_like(l2) if normal_only else l2
        loss = (
            position_contribution
            + normal_coefficient * normal
            + smooth_coefficient * regularizer
        )
        assert bool(torch.isfinite(loss))
        result = {
            "objective": float(loss.detach()),
            "position_loss_component_mm2": float(l2.detach()),
            "surface_gradient_loss": float(surface_gradient.detach()),
            "normal_loss": float(normal.detach()),
            "normal_contribution": float(normal_coefficient * normal.detach()),
            "activation_smoothness": float(regularizer.detach()),
            "regularizer_contribution": float(
                smooth_coefficient * regularizer.detach()
            ),
            "u": u.detach().cpu().numpy().copy(),
            "forward": forward,
        }
        if backward:
            loss.backward()
            result["adjoint"] = self.physics.check_adjoint()
            assert q.grad is not None
            assert bool(torch.isfinite(q.grad).all())
            result["gradient"] = q.grad.detach().cpu().numpy().copy()
        return result

    def normal_metrics(self, q: np.ndarray, result: dict) -> dict:
        row = super().metrics(q, result)
        skin_u = torch.as_tensor(result["u"][self.skin_ids])
        with torch.no_grad():
            normal_metrics = {
                key: float(value)
                for key, value in self.normal_loss.metrics(skin_u).items()
            }
        g = result["gradient"]
        # Raw6 off-diagonals contain both symmetric entries' derivatives.
        norm2 = np.sum(g[:, :3] ** 2, axis=1) + 0.5 * np.sum(g[:, 3:] ** 2, axis=1)
        return {
            **row,
            **normal_metrics,
            "normal_loss": result["normal_loss"],
            "normal_contribution": result["normal_contribution"],
            "activation_smoothness": result["activation_smoothness"],
            "regularizer_contribution": result["regularizer_contribution"],
            "physical_gradient_rms": float(np.sqrt(self.active_weights @ norm2)),
        }
