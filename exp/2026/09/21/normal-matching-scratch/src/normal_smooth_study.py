"""Neutral-start factorial study of activation smoothing and target-normal loss."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/normal-matching-profile/src"))
import normal_study as ns  # noqa: E402

am, ph, gs = ns.am, ns.ph, ns.gs
GROUP = Path(__file__).resolve().parents[1]
MODES = ns.MODES
HEIGHT = 0.20
NORMAL_BETA = 0.05
VARIANTS = (
    ("smooth-off-l2", 0.0, "l2", 0.0),
    ("smooth-off-normal", 0.0, "normal", NORMAL_BETA),
    ("smooth-on-l2", 1.0, "l2", 0.0),
    ("smooth-on-normal", 1.0, "normal", NORMAL_BETA),
)


def evaluate(
    mesh: Any,
    q: np.ndarray,
    mode: str,
    height: float,
    kind: str,
    beta: float,
    smooth_weight: float,
    scales: dict,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
):
    state, B, gradient, values = ns.evaluate(
        mesh,
        q,
        mode,
        height,
        kind,
        beta,
        scales,
        seed,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    roughness, g_B = am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
    smooth_coefficient = smooth_weight * height**2
    gradient += smooth_coefficient * am.pullback(q, mode, g_B)
    data_objective = values["objective"]
    objective = data_objective + smooth_coefficient * roughness
    values.update(
        {
            "data_objective": data_objective,
            "objective": objective,
            "objective_normalized": objective / scales["l2_0"],
            "normal_coefficient": values["shape_coefficient"],
            "smooth_coefficient": smooth_coefficient,
            "weighted_normal_loss": values["shape_coefficient"] * values["normal_loss"],
            "weighted_smoothness_loss": smooth_coefficient * roughness,
            "roughness": roughness,
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(gradient.size)),
            "projected_gradient_inf": float(
                np.linalg.norm(
                    am.gradient_mapping(q, gradient / scales["l2_0"], mode), np.inf
                )
            ),
        }
    )
    return state, B, gradient, values


def numerical_sources() -> list[Path]:
    return [Path(__file__), *ns.numerical_sources()]
