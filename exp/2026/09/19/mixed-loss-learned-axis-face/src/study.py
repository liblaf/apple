"""Learned-axis face fitting with normalized positional and gradient data losses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[6]
BASE = ROOT / "exp/2026/09/19/gradient-only-face"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
LEGACY = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
PREVIOUS = ROOT / "exp/2026/09/19/learned-axis-gradient-face/src"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep this experiment's activation implementation as the single source of truth.
_activation = _load_module(
    "mixed_loss_previous_activation", PREVIOUS / "activation_model.py"
)
pack = _activation.pack
packed_smoothness = _activation.packed_smoothness
project_ = _activation.project_

# face_study imports surface_loss by module name from its own source directory.
sys.path.insert(0, str(BASE / "src"))
from face_study import FaceStudy, activation  # noqa: E402


def symmetric_gradient(g: np.ndarray) -> np.ndarray:
    """Convert Raw6 derivatives to a symmetric Frobenius-gradient matrix."""
    matrix = np.zeros((len(g), 3, 3), dtype=g.dtype)
    matrix[:, 0, 0] = g[:, 0]
    matrix[:, 1, 1] = g[:, 1]
    matrix[:, 2, 2] = g[:, 2]
    matrix[:, 0, 1] = matrix[:, 1, 0] = g[:, 3] / 2
    matrix[:, 1, 2] = matrix[:, 2, 1] = g[:, 4] / 2
    matrix[:, 0, 2] = matrix[:, 2, 0] = g[:, 5] / 2
    return matrix


def normalized_data(
    l2: torch.Tensor,
    gradient: torch.Tensor,
    *,
    l2_0: torch.Tensor,
    gradient_0: torch.Tensor,
    scale: torch.Tensor,
    kind: str,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the configured data objective and its numeric components."""
    assert kind in {"l2", "mixed", "gradient"}
    assert beta >= 0.0
    assert bool(torch.all(l2_0 > 0))
    assert bool(torch.all(gradient_0 > 0))
    normalized_l2 = l2 / l2_0
    normalized_gradient = gradient / gradient_0
    if kind == "l2":
        l2_contribution = l2
        gradient_contribution = torch.zeros_like(l2)
    elif kind == "gradient":
        l2_contribution = torch.zeros_like(l2)
        gradient_contribution = scale * normalized_gradient
    else:
        denominator = 1.0 + beta
        l2_contribution = scale * normalized_l2 / denominator
        gradient_contribution = scale * beta * normalized_gradient / denominator
    data = l2_contribution + gradient_contribution
    assert bool(torch.isfinite(data))
    return data, {
        "normalized_position_loss": normalized_l2,
        "normalized_gradient_loss": normalized_gradient,
        "position_data_contribution": l2_contribution,
        "gradient_data_contribution": gradient_contribution,
    }


class Study(FaceStudy):
    """Corrected face physics with a selectable normalized data objective."""

    def __init__(self, length: float) -> None:
        super().__init__()
        self.data_loss = "mixed"
        self.data_scale = 1.0
        self.kind = "mixed"
        self.beta = 1.0
        with torch.no_grad():
            neutral_u = torch.zeros(
                (len(self.physics.points), 3),
                device=self.target.device,
                dtype=self.target.dtype,
            )
            l2_0, gradient_0 = self.losses(neutral_u)
        assert float(l2_0) > 0.0
        assert float(gradient_0) > 0.0
        self.normalization = {
            "L20": float(l2_0),
            "Lg0": float(gradient_0),
            "K": float(l2_0),
        }
        self._l2_0 = l2_0.detach()
        self._gradient_0 = gradient_0.detach()
        self._scale = l2_0.detach()
        i, j, weight = self.physics.graph
        assert np.all(self.physics.region[i] == self.physics.region[j])
        self.edges = torch.as_tensor(np.column_stack((i, j)), dtype=torch.long)
        self.edge_weights = torch.as_tensor(weight)
        self.smooth_factor = length**2 * weight.sum() / self.physics.volumes.sum()
        self.length = length

    def set_loss(self, kind: str, beta: float = 1.0) -> None:
        """Set L2, normalized gradient, or their normalized mixture."""
        assert kind in {"l2", "mixed", "gradient"}
        assert beta >= 0.0
        self.kind = kind
        self.beta = float(beta)

    def data_objective(
        self, l2: torch.Tensor, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate the configured data objective without a mechanics solve."""
        return normalized_data(
            l2,
            gradient,
            l2_0=self._l2_0,
            gradient_0=self._gradient_0,
            scale=self._scale,
            kind=self.kind,
            beta=self.beta,
        )

    def regularizer(self, q: torch.Tensor) -> torch.Tensor:
        return self.smooth_factor * packed_smoothness(q, self.edges, self.edge_weights)

    def evaluate_raw(
        self, q: torch.Tensor, seed: np.ndarray, *, backward: bool
    ) -> dict:
        """Evaluate configured data loss for a Raw6 activation without smoothness."""
        if backward:
            q.grad = None
        u = self.physics.solve(q, seed)
        forward = dict(self.physics.last_forward)
        assert forward["success"], forward
        l2, gradient = self.losses(u)
        data, components = self.data_objective(l2, gradient)
        result = {
            "objective": float(data.detach()),
            "data_objective": float(data.detach()),
            "position_loss_component_mm2": float(l2.detach()),
            "surface_gradient_loss": float(gradient.detach()),
            **{key: float(value.detach()) for key, value in components.items()},
            "u": u.detach().cpu().numpy().copy(),
            "forward": forward,
        }
        if backward:
            data.backward()
            result["adjoint"] = self.physics.check_adjoint()
            assert q.grad is not None
            assert torch.isfinite(q.grad).all()
            result["gradient"] = q.grad.detach().cpu().numpy().copy()
        return result

    def evaluate_axis(
        self,
        strength: torch.Tensor,
        axis: torch.Tensor,
        seed: np.ndarray,
        coefficient: float,
        *,
        backward: bool,
    ) -> dict:
        """Evaluate configured data loss plus tensor smoothness in axis variables."""
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
        data, components = self.data_objective(l2, gradient)
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
            **{key: float(value.detach()) for key, value in components.items()},
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
        """Add physical constrained-gradient and activation diagnostics."""
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
            "normalized_position_loss": result["normalized_position_loss"],
            "normalized_gradient_loss": result["normalized_gradient_loss"],
            "position_data_contribution": result["position_data_contribution"],
            "gradient_data_contribution": result["gradient_data_contribution"],
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
