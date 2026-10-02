"""Raw6 face study with an explicitly dimensionless position/normal objective."""

from __future__ import annotations

import numpy as np
import torch
from study import Study


class ReferenceStudy(Study):
    def __init__(self, l_ref_mm: float, length: float = 0.005) -> None:
        assert np.isfinite(l_ref_mm)
        assert l_ref_mm > 0
        super().__init__(length)
        self.l_ref_mm = l_ref_mm

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
        """Use beta as the direct normal weight for the frozen runner interface."""
        if backward:
            q.grad = None
        u = self.physics.solve(q, seed)
        forward = dict(self.physics.last_forward)
        assert forward["success"], forward
        l2, surface_gradient = self.losses(u)
        normal = self.normal_loss(u[self.skin_ids_t])
        regularizer = self.regularizer(q)
        position = torch.zeros_like(l2) if normal_only else l2 / self.l_ref_mm**2
        loss = position + beta * normal + alpha * regularizer
        assert bool(torch.isfinite(loss))
        result = {
            "objective": float(loss.detach()),
            "position_loss_component_mm2": float(l2.detach()),
            "position_contribution": float(position.detach()),
            "surface_gradient_loss": float(surface_gradient.detach()),
            "normal_loss": float(normal.detach()),
            "normal_contribution": float(beta * normal.detach()),
            "activation_smoothness": float(regularizer.detach()),
            "regularizer_contribution": float(alpha * regularizer.detach()),
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
        return {
            **super().normal_metrics(q, result),
            "position_contribution": result["position_contribution"],
        }
