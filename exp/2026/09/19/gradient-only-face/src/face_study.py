"""Matched surface objectives using the corrected full-face equilibrium model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from surface_loss import SurfaceGradientLoss

ROOT = Path(__file__).resolve().parents[6]
LEGACY = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
sys.path.insert(0, str(LEGACY))
from study_metrics import StudyMetrics  # noqa: E402
from study_physics import FacePhysics, configure  # noqa: E402


def activation(q: np.ndarray) -> np.ndarray:
    B = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    B[:, 0, 0] += q[:, 0]
    B[:, 1, 1] += q[:, 1]
    B[:, 2, 2] += q[:, 2]
    B[:, 0, 1] = B[:, 1, 0] = q[:, 3]
    B[:, 1, 2] = B[:, 2, 1] = q[:, 4]
    B[:, 0, 2] = B[:, 2, 0] = q[:, 5]
    return B


class FaceStudy:
    def __init__(self) -> None:
        configure()
        self.physics = FacePhysics(FIXTURE, activation_model="raw6")
        self.diagnostics = StudyMetrics(FIXTURE)
        p = self.physics
        self.skin_ids = np.asarray(p.skin.point_data["GlobalPointId"], dtype=np.int64)
        self.triangles = np.asarray(p.skin.faces).reshape(-1, 4)[:, 1:].copy()
        assert np.array_equal(p.skin.points, p.points[self.skin_ids])
        assert np.isin(self.skin_ids, p.top).all()
        self.skin_ids_t = torch.as_tensor(self.skin_ids)
        self.target = torch.as_tensor(p.target[self.skin_ids])
        self.gradient_loss = SurfaceGradientLoss(
            p.points[self.skin_ids], self.triangles, device="cuda", dtype=torch.float64
        )
        area = self.gradient_loss.areas.detach().cpu().numpy()
        mass = np.zeros(len(self.skin_ids))
        np.add.at(mass, self.triangles.ravel(), np.repeat(area / 3, 3))
        assert np.all(mass > 0)
        self.weights = mass / mass.sum()
        self.weights_t = torch.as_tensor(self.weights)
        self.active_weights = p.volumes / p.volumes.sum()

    def losses(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skin_u = u[self.skin_ids_t]
        error = skin_u - self.target
        l2 = (self.weights_t[:, None] * error.square()).sum() * (1e6 / 3)
        gradient = self.gradient_loss(skin_u, self.target)
        return l2, gradient

    def evaluate(
        self,
        q: torch.Tensor,
        seed: np.ndarray,
        kind: str,
        scale: float,
        *,
        backward: bool,
    ):
        if backward:
            q.grad = None
        u = self.physics.solve(q, seed)
        forward = dict(self.physics.last_forward)
        assert forward["success"], forward
        l2, gradient = self.losses(u)
        assert kind in {"l2", "gradient"}
        loss = l2 if kind == "l2" else scale * gradient
        assert torch.isfinite(loss)
        result = {
            "objective": float(loss.detach()),
            "position_loss_component_mm2": float(l2.detach()),
            "surface_gradient_loss": float(gradient.detach()),
            "u": u.detach().cpu().numpy().copy(),
            "forward": forward,
        }
        if backward:
            loss.backward()
            result["adjoint"] = self.physics.check_adjoint()
            assert q.grad is not None
            assert torch.isfinite(q.grad).all()
            result["gradient"] = q.grad.detach().cpu().numpy().copy()
        return result

    def physical_step_rms(self, dq: np.ndarray) -> float:
        frobenius2 = np.sum(dq[:, :3] ** 2, axis=1) + 2 * np.sum(dq[:, 3:] ** 2, axis=1)
        return float(np.sqrt(self.active_weights @ frobenius2))

    def metrics(self, q: np.ndarray, result: dict) -> dict:
        p = self.physics
        u = result["u"]
        pred = u[self.skin_ids]
        target = p.target[self.skin_ids]
        error = pred - target
        mean = np.sum(self.weights[:, None] * error, axis=0)
        J = p.detf(u)
        fixed = np.asarray(p.mesh.point_data["FixedMask"], dtype=bool)
        prescribed = np.asarray(p.mesh.point_data["FixedValue"])
        fixed_error = float(np.max(np.abs(u[fixed] - prescribed[fixed])))
        assert fixed_error < 1e-14
        B = activation(q)
        eigenvalues = np.linalg.eigvalsh(B)
        return {
            "objective": result["objective"],
            "position_loss_component_mm2": result["position_loss_component_mm2"],
            "surface_gradient_loss": result["surface_gradient_loss"],
            "surface_gradient_rms": float(np.sqrt(result["surface_gradient_loss"])),
            "fit_rms_mm": float(
                1000 * np.sqrt(np.sum(self.weights[:, None] * error**2))
            ),
            "motion_rms_mm": float(
                1000 * np.sqrt(np.sum(self.weights[:, None] * pred**2))
            ),
            "centered_fit_rms_mm": float(
                1000 * np.sqrt(np.sum(self.weights[:, None] * (error - mean) ** 2))
            ),
            "mean_error_norm_mm": float(1000 * np.linalg.norm(mean)),
            "target_projection": float(
                np.sum(self.weights[:, None] * pred * target)
                / np.sum(self.weights[:, None] * target**2)
            ),
            "detF_min": float(J.min()),
            "detF_max": float(J.max()),
            "inverted_all_cells": int(np.count_nonzero(J <= 0)),
            "inverted_active_cells": int(np.count_nonzero(J[p.ids] <= 0)),
            "non_spd_active_cells": int(np.count_nonzero(eigenvalues[:, 0] <= 0)),
            "activation_eigen_min": float(eigenvalues.min()),
            "activation_eigen_max": float(eigenvalues.max()),
            "fixed_displacement_error": fixed_error,
            "gradient_rms": float(np.sqrt(np.mean(result["gradient"] ** 2))),
            "forward_steps": result["forward"]["steps"],
            "forward_grad_norm": result["forward"]["grad_norm"],
            **self.diagnostics.evaluate_surface(pred),
        }
