"""Target-normal and gradient refinements of the historical 2D L2 fit."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "exp/2026/09/19/gradient-only-profile/src"))
import gradient_study as gs  # noqa: E402

am, ph = gs.am, gs.ph
GROUP = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "exp/2026/09/15/activation-direction-smoothness/data/tune-w0"
MODES = ("unconstrained", "contraction_only", "learned_direction", "x_contraction")
HEIGHTS = (0.05, 0.20)
VARIANTS = (
    ("l2", "l2", 0.0),
    ("gradient-005", "gradient", 0.05),
    ("gradient-025", "gradient", 0.25),
    ("normal-005", "normal", 0.05),
    ("normal-025", "normal", 0.25),
)


def pack(mesh: Any, full_u: np.ndarray) -> np.ndarray:
    free = mesh.lookup >= 0
    result = np.empty(mesh.nfree)
    result[mesh.lookup[free]] = full_u.reshape(-1)[free]
    return result


def curve_normal_loss(
    points: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Fixed-reference-length-weighted cosine mismatch of oriented edge normals.

    In 2D, rotating both unit tangents by 90 degrees preserves their dot product.
    Half their squared difference is exactly 1 minus that dot product, with less
    cancellation near agreement. The gradient includes edge normalization.
    """
    edges, target_edges = np.diff(points, axis=0), np.diff(target, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    target_lengths = np.linalg.norm(target_edges, axis=1)
    assert np.all(lengths > 1e-12)
    assert np.all(target_lengths > 1e-12)
    tangent, target_tangent = (
        edges / lengths[:, None],
        target_edges / target_lengths[:, None],
    )
    delta = tangent - target_tangent
    value = float(0.5 * np.sum(weights[:, None] * delta**2))
    edge_gradient = (
        weights[:, None]
        * (delta - np.sum(delta * tangent, axis=1)[:, None] * tangent)
        / lengths[:, None]
    )
    gradient = np.zeros_like(points)
    gradient[:-1] -= edge_gradient
    gradient[1:] += edge_gradient
    dot = np.sum(tangent * target_tangent, axis=1)
    cross = tangent[:, 0] * target_tangent[:, 1] - tangent[:, 1] * target_tangent[:, 0]
    angles = np.arctan2(cross, dot)
    return (
        value,
        gradient,
        {
            "normal_angle_rms_deg": float(
                np.degrees(np.sqrt(np.sum(weights * angles**2)))
            ),
            "normal_chord_rms": float(np.sqrt(2 * value)),
            "minimum_top_edge_length": float(lengths.min()),
        },
    )


def normal_loss(mesh: Any, u: np.ndarray, height: float):
    top = gs.top_nodes(mesh)
    reference = mesh.p[top]
    x = reference[:, 0]
    weights = np.diff(x) / (x[-1] - x[0])
    target = reference + np.column_stack((np.zeros_like(x), 4 * height * x * (1 - x)))
    points = reference + ph.unpack(mesh, u)[top]
    value, gradient_top, metrics = curve_normal_loss(points, target, weights)
    lookup = mesh.lookup.reshape(-1, 2)[top]
    free = lookup >= 0
    gradient = np.zeros(mesh.nfree)
    gradient[lookup[free]] = gradient_top[free]
    return value, gradient, metrics


def normalization(mesh: Any, height: float) -> dict[str, float]:
    zero = np.zeros(mesh.nfree)
    l2 = ph.loss(mesh, zero, height, "l2")[0]
    gradient = gs.profile_loss(mesh, zero, height)[0]
    normal = normal_loss(mesh, zero, height)[0]
    assert min(l2, gradient, normal) > 0
    return {"l2_0": l2, "gradient_0": gradient, "normal_0": normal}


def losses(
    mesh: Any, u: np.ndarray, height: float, kind: str, beta: float, scales: dict
):
    l2, g_l2, target = ph.loss(mesh, u, height, "l2")
    lg, g_lg, profile = gs.profile_loss(mesh, u, height)
    ln, g_ln, normal = normal_loss(mesh, u, height)
    assert kind in {"l2", "gradient", "normal"}
    assert beta >= 0
    value, gradient = l2, g_l2.copy()
    coefficient = 0.0
    if kind != "l2":
        coefficient = beta * scales["l2_0"] / scales[f"{kind}_0"]
        value += coefficient * (lg if kind == "gradient" else ln)
        gradient += coefficient * (g_lg if kind == "gradient" else g_ln)
    prediction = ph.unpack(mesh, u)[mesh.top]
    projection = float(np.sum(prediction * target) / np.sum(target**2))
    metrics = {
        **profile,
        **normal,
        "normal_loss": ln,
        "objective": value,
        "objective_normalized": value / scales["l2_0"],
        "normal_loss_normalized": ln / scales["normal_0"],
        "gradient_loss_normalized": lg / scales["gradient_0"],
        "position_loss_normalized": l2 / scales["l2_0"],
        "shape_coefficient": coefficient,
        "target_projection": projection,
    }
    return value, gradient, metrics


def evaluate(
    mesh: Any,
    q: np.ndarray,
    mode: str,
    height: float,
    kind: str,
    beta: float,
    scales: dict,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
):
    B = gs.study.matrices(mesh, q, mode)
    state = ph.solve(mesh, B, seed, tolerance=tolerance, max_iterations=max_iterations)
    _, g_u, values = losses(mesh, state.u, height, kind, beta, scales)
    packed, adjoint_residual = ph.control_gradient(mesh, state, B, g_u, "unconstrained")
    pg = packed.reshape(-1, 3)
    g_B = np.zeros((len(pg), 2, 2))
    g_B[:, 0, 0], g_B[:, 1, 1] = pg[:, 0], pg[:, 1]
    g_B[:, 0, 1] = g_B[:, 1, 0] = pg[:, 2] / 2
    gradient = am.pullback(q, mode, g_B)
    roughness, _ = am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
    diagnostics = ph.diagnostics(
        mesh, state, gs.study.packed(B[mesh.muscle]), "unconstrained", height
    )
    eigen = np.linalg.eigvalsh(B[mesh.muscle])
    diagnostics.update(values)
    diagnostics.update(
        {
            "activation_roughness": roughness,
            "min_eigenvalue_B": float(eigen.min()),
            "max_eigenvalue_B": float(eigen.max()),
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(gradient.size)),
            "projected_gradient_inf": float(
                np.linalg.norm(
                    am.gradient_mapping(q, gradient / scales["l2_0"], mode), np.inf
                )
            ),
            "adjoint_relative_residual": adjoint_residual,
            "last_forward_iterations": state.iterations,
            "peak_target_fraction": diagnostics["peak_top_uy"] / height,
        }
    )
    return state, B, gradient, diagnostics


def numerical_sources() -> list[Path]:
    return [
        Path(__file__),
        Path(gs.__file__),
        Path(gs.study.__file__),
        Path(am.__file__),
        Path(ph.__file__),
        ph.MESH_SOURCE,
    ]
