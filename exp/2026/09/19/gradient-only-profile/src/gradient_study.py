"""Target-relative top-profile gradients with the saved corrected 2D physics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
PREVIOUS = ROOT / "exp/2026/09/15/activation-direction-smoothness/src"
sys.path.insert(0, str(PREVIOUS))
import activation_models as am  # noqa: E402
import study  # noqa: E402

ph = study.ph


def top_nodes(mesh: Any) -> np.ndarray:
    """Full reference top boundary, including its two fixed corners."""
    top = np.flatnonzero(np.isclose(mesh.p[:, 1], 0.1))
    return top[np.argsort(mesh.p[top, 0])]


def profile_loss(mesh: Any, u: np.ndarray, height: float):
    """Return pure gradient matching, its displacement derivative and metrics.

    Integrate squared piecewise-linear residual slopes over reference length.
    Both vector components are matched. Position/curvature errors are diagnostic
    only; neither contributes to the scalar objective or its derivative.
    """
    full_u = ph.unpack(mesh, u)
    top = top_nodes(mesh)
    x = mesh.p[top, 0]
    dx = np.diff(x)
    span = x[-1] - x[0]
    assert np.all(dx > 0)
    target = np.column_stack((np.zeros_like(x), 4 * height * x * (1 - x)))
    error = full_u[top] - target
    slope_error = np.diff(error, axis=0) / dx[:, None]
    value = float(np.sum(dx[:, None] * slope_error**2) / span)
    edge_gradient = 2 * slope_error / span
    node_gradient = np.zeros_like(error)
    node_gradient[:-1] -= edge_gradient
    node_gradient[1:] += edge_gradient
    lookup = mesh.lookup.reshape(-1, 2)[top]
    free = lookup >= 0
    gradient = np.zeros(mesh.nfree)
    gradient[lookup[free]] = node_gradient[free]
    position_loss = ph.loss(mesh, u, height, "l2")[0]
    center_dx = (dx[:-1] + dx[1:]) / 2
    curvature = np.diff(slope_error, axis=0) / center_dx[:, None]
    curvature_error = float(
        np.sqrt(np.sum(center_dx[:, None] * curvature**2) / np.sum(center_dx))
    )
    values = {
        "position_loss": position_loss,
        "gradient_loss": value,
        "slope_rms": float(np.sqrt(value)),
        "curvature_error": curvature_error,
        "slope_x_rms": float(np.sqrt(np.sum(dx * slope_error[:, 0] ** 2) / span)),
        "slope_y_rms": float(np.sqrt(np.sum(dx * slope_error[:, 1] ** 2) / span)),
    }
    return value, gradient, values


def evaluate(
    mesh: Any,
    q: np.ndarray,
    mode: str,
    height: float,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
):
    B = study.matrices(mesh, q, mode)
    state = ph.solve(mesh, B, seed, tolerance=tolerance, max_iterations=max_iterations)
    value, g_u, profile = profile_loss(mesh, state.u, height)
    packed_gradient, adjoint_residual = ph.control_gradient(
        mesh, state, B, g_u, "unconstrained"
    )
    pg = packed_gradient.reshape(-1, 3)
    g_B = np.zeros((len(pg), 2, 2))
    g_B[:, 0, 0], g_B[:, 1, 1] = pg[:, 0], pg[:, 1]
    g_B[:, 0, 1] = g_B[:, 1, 0] = pg[:, 2] / 2
    gradient = am.pullback(q, mode, g_B)
    roughness, _ = am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
    diagnostics = ph.diagnostics(
        mesh, state, study.packed(B[mesh.muscle]), "unconstrained", height
    )
    diagnostics.update(profile)
    diagnostics.update(
        {
            "objective": value,
            "normalized_loss": value / height**2,
            "normalized_position_loss": profile["position_loss"] / height**2,
            "roughness": roughness,
            "projected_gradient_inf": float(
                np.linalg.norm(
                    am.gradient_mapping(q, gradient / height**2, mode), np.inf
                )
            ),
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(len(q))),
            "adjoint_relative_residual": adjoint_residual,
            "last_forward_iterations": state.iterations,
        }
    )
    return state, B, gradient, diagnostics
