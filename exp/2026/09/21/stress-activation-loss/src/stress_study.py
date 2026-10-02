# Copyright (c) 2026 liblaf
"""Dimensionless position/normal fitting and activation-field regularization."""
# ruff: noqa: EM102, TRY003

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from activation_models import STRESS_REF_MPA, matrices
from stress_physics import FacePhysics, configure
from stress_regularization import gradient_balance

from liblaf.apple.inverse import ImplicitNumericalError

ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
sys.path.append(str(ROOT / "exp/2026/09/21/normal-matching-face/src"))
sys.path.append(str(ROOT / "exp/2026/09/19/gradient-only-face/src"))
from surface_loss import SurfaceGradientLoss  # noqa: E402
from surface_normal import SurfaceNormalLoss  # noqa: E402

L_REF_MM = 13.236093032531715
SMOOTH_LENGTH_M = 0.005


class InvalidEquilibriumError(RuntimeError):
    pass


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        message = f"{name} is nonfinite"
        raise ImplicitNumericalError(message)


def deformation_diagnostics(J: np.ndarray) -> dict[str, float | int]:
    """Report finite deformation determinants without rejecting inversion.

    Stable Neo-Hookean continuation intentionally permits finite inverted cells;
    their count remains a diagnostic for the outer optimizer.
    """
    if not np.isfinite(J).all():
        message = "deformation determinant is nonfinite"
        raise ImplicitNumericalError(message)
    return {
        "detF_min": float(J.min()),
        "detF_max": float(J.max()),
        "inverted_all_cells": int(np.count_nonzero(J <= 0)),
    }


class StressStudy:
    def __init__(self, **physics_options) -> None:
        configure()
        self.physics = FacePhysics(FIXTURE, **physics_options)
        p = self.physics
        self.skin_ids = np.asarray(p.skin.point_data["GlobalPointId"], dtype=np.int64)
        self.triangles = np.asarray(p.skin.faces).reshape(-1, 4)[:, 1:].copy()
        assert np.array_equal(p.skin.points, p.points[self.skin_ids])
        self.skin_ids_t = torch.as_tensor(self.skin_ids, dtype=torch.long)
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
        self.normal_loss = SurfaceNormalLoss(
            p.points[self.skin_ids],
            self.triangles,
            p.target[self.skin_ids],
            device="cuda",
            dtype=torch.float64,
        )
        i, j, w = p.graph
        assert np.all(p.region[i] == p.region[j])
        self.edge_i, self.edge_j = torch.as_tensor(i), torch.as_tensor(j)
        self.conductance = torch.as_tensor(w)
        self.regularizer_factor = SMOOTH_LENGTH_M**2 / p.volumes.sum()
        self.active_weights = p.volumes / p.volumes.sum()

    def regularizer(self, Qhat: torch.Tensor) -> torch.Tensor:
        delta = Qhat[self.edge_i] - Qhat[self.edge_j]
        return (
            self.regularizer_factor
            * (self.conductance * delta.square().sum((-2, -1))).sum()
        )

    def evaluate(
        self,
        q: torch.Tensor,
        mode: str,
        axes: torch.Tensor | None,
        seed: np.ndarray,
        normal_weight: float,
        smooth_weight: float,
        *,
        backward: bool = True,
        component_gradients: bool = False,
    ) -> dict:
        if q.grad is not None:
            q.grad = None
        activation = matrices(q, mode, axes)
        solve_input = (
            activation * STRESS_REF_MPA
            if self.physics.activation_model == "stress"
            else activation
        )
        u = self.physics.solve(solve_input, seed)
        forward = dict(self.physics.last_forward)
        if not forward["success"] and self.physics.diff.require_convergence:
            raise InvalidEquilibriumError(f"Forward did not converge: {forward}")
        u_np = u.detach().cpu().numpy().copy()
        J = self.physics.detf(u_np)
        deformation = deformation_diagnostics(J)
        error = u[self.skin_ids_t] - self.target
        position = (
            (self.weights_t[:, None] * error.square()).sum() * (1e6 / 3) / L_REF_MM**2
        )
        normal = self.normal_loss(u[self.skin_ids_t])
        regularizer = self.regularizer(activation)
        loss = position + normal_weight * normal + smooth_weight * regularizer
        _require_finite(loss, name="inverse objective")
        result = {
            "objective": float(loss.detach()),
            "position_contribution": float(position.detach()),
            "normal_loss": float(normal.detach()),
            "normal_contribution": float(normal_weight * normal.detach()),
            "activation_smoothness": float(regularizer.detach()),
            "regularizer_contribution": float(smooth_weight * regularizer.detach()),
            "fit_rms_mm": float(
                torch.sqrt((self.weights_t[:, None] * error.square()).sum()).detach()
            )
            * 1000,
            **deformation,
            "u": u_np,
            "forward": forward,
            "solver_valid": bool(forward["solver_valid"]),
            "activation_model": self.physics.activation_model,
        }
        with torch.no_grad():
            result.update(
                {
                    k: float(v)
                    for k, v in self.normal_loss.metrics(u[self.skin_ids_t]).items()
                }
            )
            result["surface_gradient_loss"] = float(
                self.gradient_loss(u[self.skin_ids_t], self.target)
            )
            delta = activation[self.edge_i] - activation[self.edge_j]
            neighbor_rms = float(
                torch.sqrt(
                    (self.conductance * delta.square().sum((-2, -1))).sum()
                    / self.conductance.sum()
                )
            )
            eig = np.linalg.eigvalsh(activation.detach().cpu().numpy())
            field_rms = float(
                torch.sqrt(
                    (
                        torch.as_tensor(self.active_weights)
                        * activation.square().sum((-2, -1))
                    ).sum()
                )
            )
            if self.physics.activation_model == "stress":
                result["neighbor_stress_rms_kPa"] = neighbor_rms * STRESS_REF_MPA * 1000
                result["stress_eigen_min_kPa"] = (
                    float(eig.min()) * STRESS_REF_MPA * 1000
                )
                result["stress_eigen_max_kPa"] = (
                    float(eig.max()) * STRESS_REF_MPA * 1000
                )
                result["stress_frobenius_rms_kPa"] = field_rms * STRESS_REF_MPA * 1000
            else:
                B = np.eye(3) + activation.detach().cpu().numpy()
                B_eig = np.linalg.eigvalsh(B)
                result["neighbor_strain_rms_dimensionless"] = neighbor_rms
                result["strain_eigen_min_dimensionless"] = float(eig.min())
                result["strain_eigen_max_dimensionless"] = float(eig.max())
                result["strain_frobenius_rms_dimensionless"] = field_rms
                result["B_eigen_min_dimensionless"] = float(B_eig.min())
                result["B_eigen_max_dimensionless"] = float(B_eig.max())
        result.update(
            _loss=loss,
            _position=position,
            _Qhat=activation,
            _normal_weight=normal_weight,
            _smooth_weight=smooth_weight,
        )
        if backward:
            self.backward(q, result, component_gradients=component_gradients)
        return result

    def backward(
        self, q: torch.Tensor, result: dict, *, component_gradients: bool = False
    ) -> None:
        """Retain the symmetric activation gradient before pulling it into controls."""
        loss, position, Qhat = (result.pop(k) for k in ("_loss", "_position", "_Qhat"))
        beta, eta = result.pop("_normal_weight"), result.pop("_smooth_weight")
        g_l2 = None
        if component_gradients and beta != 0:
            g_l2 = torch.autograd.grad(position, Qhat, retain_graph=True)[0].detach()
            g_l2 = (g_l2 + g_l2.mT) / 2
            result["l2_adjoint"] = self.physics.check_adjoint()
        gradient = torch.autograd.grad(loss, Qhat)[0].detach()
        gradient = (gradient + gradient.mT) / 2
        result["adjoint"] = self.physics.check_adjoint()
        q.grad = torch.autograd.grad(Qhat, q, grad_outputs=gradient)[0].detach()
        _require_finite(gradient, name="ambient activation gradient")
        _require_finite(q.grad, name="stage-control gradient")
        result["gradient"] = q.grad.clone()
        result["solver_valid"] = bool(
            result["solver_valid"] and result["adjoint"]["success"]
        )
        result["tensor_gradient"] = gradient
        result["gradient_rms"] = float(torch.sqrt(q.grad.square().mean()))
        field = Qhat.detach().requires_grad_(requires_grad=True)
        g_regularizer = torch.autograd.grad(self.regularizer(field), field)[0].detach()
        if beta == 0:
            g_l2 = gradient - eta * g_regularizer
        if g_l2 is not None:
            result.update(
                gradient_balance(
                    g_l2, g_regularizer, torch.as_tensor(self.active_weights), eta
                )
            )
            if component_gradients:
                result["l2_tensor_gradient"] = g_l2
                result["regularizer_tensor_gradient"] = g_regularizer

    def save_geometry(self, path: Path) -> None:
        p = self.physics
        np.savez_compressed(
            path,
            rest_points=p.points,
            tets=p.tets,
            active_ids=p.ids,
            skin_ids=self.skin_ids,
            triangles=self.triangles,
            target_displacement_skin=p.target[self.skin_ids],
            skin_vertex_weights=self.weights,
            active_volume_weights=self.active_weights,
            edge_i=p.graph[0],
            edge_j=p.graph[1],
            edge_weight=p.graph[2],
            regularizer_factor=self.regularizer_factor,
        )
