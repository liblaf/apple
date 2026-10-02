"""Adapters around the saved corrected 2D physics; material is never optimized."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np

ROOT = Path(__file__).resolve().parents[6]
PREVIOUS = ROOT / "exp/2026/09/14/fiber-contraction-parabola/src"
sys.path.insert(0, str(PREVIOUS))
import physics2d as ph  # noqa: E402


def matrices(mesh: Any, q: np.ndarray, mode: str):
    B = np.broadcast_to(np.eye(2), (len(mesh.tri), 2, 2)).copy()
    B[mesh.muscle] = am.matrices(q, mode)
    return B


def packed(B: np.ndarray):
    S = B - np.eye(2)
    return np.column_stack((S[:, 0, 0], S[:, 1, 1], S[:, 0, 1])).ravel()


def evaluate(
    mesh: Any,
    q: np.ndarray,
    mode: str,
    height: float,
    weight: float,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
):
    B = matrices(mesh, q, mode)
    state = ph.solve(mesh, B, seed, tolerance=tolerance, max_iterations=max_iterations)
    data_loss, g_u, _ = ph.loss(mesh, state.u, height, "l2")
    packed_grad, adjoint_residual = ph.control_gradient(
        mesh, state, B, g_u, "unconstrained"
    )
    pg = packed_grad.reshape(-1, 3)
    g_B = np.zeros((len(pg), 2, 2))
    g_B[:, 0, 0], g_B[:, 1, 1] = pg[:, 0], pg[:, 1]
    g_B[:, 0, 1] = g_B[:, 1, 0] = pg[:, 2] / 2
    roughness, g_smooth = am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
    gradient = am.pullback(q, mode, g_B + weight * height**2 * g_smooth)
    objective = data_loss + weight * height**2 * roughness
    diagnostics = ph.diagnostics(
        mesh, state, packed(B[mesh.muscle]), "unconstrained", height
    )
    eigenvalues = np.linalg.eigvalsh(B[mesh.muscle])
    S = B[mesh.muscle] - np.eye(2)
    magnitude = float(np.mean(np.sum(S * S, axis=(1, 2))))
    mapping = am.gradient_mapping(q, gradient / height**2, mode)
    diagnostics.update(
        {
            "raw_loss": data_loss,
            "normalized_loss": data_loss / height**2,
            "objective": objective,
            "objective_normalized": objective / height**2,
            "roughness": roughness,
            "max_neighbor_jump": float(
                np.linalg.norm(
                    B[mesh.muscle][np.asarray(mesh.edges)[:, 0]]
                    - B[mesh.muscle][np.asarray(mesh.edges)[:, 1]],
                    axis=(1, 2),
                ).max()
            ),
            "min_abs_det_B": float(np.abs(np.linalg.det(B[mesh.muscle])).min()),
            "tensor_neighbor_rms": float(np.sqrt(roughness)),
            "tensor_magnitude_rms": float(np.sqrt(magnitude)),
            "relative_roughness": roughness / magnitude if magnitude > 0 else 0.0,
            "regularizer_normalized": weight * roughness,
            "projected_gradient_inf": float(np.linalg.norm(mapping, np.inf)),
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(len(q))),
            "min_eigenvalue_B": float(eigenvalues.min()),
            "max_eigenvalue_B": float(eigenvalues.max()),
            "min_det_B": float(np.linalg.det(B[mesh.muscle]).min()),
            "nonpositive_det_B_fraction": float(
                np.mean(np.linalg.det(B[mesh.muscle]) <= 0)
            ),
            "active_extension_fraction": float(
                np.mean(
                    np.linalg.svd(B[mesh.muscle], compute_uv=False).min(axis=1)
                    < 1 - 1e-10
                )
            ),
            "rank1_secondary_abs_max": float(np.max(np.abs(eigenvalues[:, 0] - 1))),
            "adjoint_relative_residual": adjoint_residual,
            "last_forward_iterations": state.iterations,
        }
    )
    return state, B, gradient, diagnostics
