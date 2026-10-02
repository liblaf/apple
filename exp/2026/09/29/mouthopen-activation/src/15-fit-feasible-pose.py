"""Test a chin pose fit constrained by the prescribed tetrahedra's orientation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    source: Path = Path("10-mandible")
    output: Path = Path("15-feasible-pose")
    minimum_jacobian: float = 0.1
    max_iterations: int = 500


def main(cfg: Config) -> None:
    assert 0 < cfg.minimum_jacobian < 1
    source = cherries.input(cfg.source)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source / "prepared.npz", allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    points = data["X"]
    pivot = data["pivot"]
    fixed_tets = data["tets"][data["allfixed_cell_ids"]]
    jaw_at_vertices = data["jaw_mask"][fixed_tets]
    mixed = jaw_at_vertices.any(axis=1) & ~jaw_at_vertices.all(axis=1)
    tets = fixed_tets[mixed]
    jaw_at_vertices = jaw_at_vertices[mixed]
    rest = points[tets]
    denominator = np.linalg.det(rest[:, 1:] - rest[:, :1])
    assert np.all(denominator > 0)
    skin = points[data["skin_ids"]]
    triangles = skin[data["triangles"]]
    area = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=1,
        )
        / 2
    )
    weights = np.zeros(len(skin))
    np.add.at(weights, data["triangles"].ravel(), np.repeat(area / 3, 3))
    patch = data["patch_ids"]
    weights = weights[patch] / weights[patch].sum()
    x, y = skin[patch], data["target_skin"][patch]
    scale = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    initial_loss = float(np.sum(weights[:, None] * (x - y) ** 2))

    def transform(xyz: np.ndarray, q: np.ndarray) -> np.ndarray:
        pose = q * scale
        return (
            (xyz - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
            + pivot
            + pose[3:]
        )

    def loss(q: np.ndarray) -> float:
        residual = transform(x, q) - y
        return float(np.sum(weights[:, None] * residual**2) / initial_loss)

    def jacobians(q: np.ndarray) -> np.ndarray:
        moved = np.where(jaw_at_vertices[..., None], transform(rest, q), rest)
        return np.linalg.det(moved[:, 1:] - moved[:, :1]) / denominator

    histories = []
    candidates = []
    for fraction in (0.0, 0.001):
        initial = fraction * data["pose"] / scale
        assert np.min(jacobians(initial)) >= cfg.minimum_jacobian
        history = []

        def callback(q: np.ndarray, history: list = history) -> None:
            history.append(
                {
                    "iteration": len(history) + 1,
                    "chin_rms_mm": float(1000 * np.sqrt(loss(q) * initial_loss)),
                    "min_prescribed_jacobian": float(jacobians(q).min()),
                }
            )

        result = minimize(
            loss,
            initial,
            method="SLSQP",
            constraints=[
                {"type": "ineq", "fun": lambda q: jacobians(q) - cfg.minimum_jacobian}
            ],
            options={"maxiter": cfg.max_iterations, "ftol": 1e-12, "eps": 1e-6},
            callback=callback,
        )
        j = jacobians(result.x)
        pose = result.x * scale
        candidates.append(
            {
                "initial_pose_fraction": fraction,
                "optimizer_success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "pose_rad_m": pose.tolist(),
                "rotation_degrees": float(np.rad2deg(np.linalg.norm(pose[:3]))),
                "translation_norm_mm": float(1000 * np.linalg.norm(pose[3:])),
                "chin_rms_mm": float(1000 * np.sqrt(loss(result.x) * initial_loss)),
                "min_prescribed_jacobian": float(j.min()),
                "inverted_prescribed_tetrahedra": int(np.count_nonzero(j <= 0)),
                "meets_requested_jacobian_margin": bool(
                    j.min() >= cfg.minimum_jacobian - 1e-7
                ),
            }
        )
        histories.append(history)
    result = {
        "schema": "mouthopen-constrained-chin-pose-v1",
        "scope": "CPU geometric pose fit only; no equilibrium, activation, contact, or anatomical pose validation",
        "constraint": "Mixed moving/stationary all-IsFixed tetrahedra retain det(F) >= minimum_jacobian; all original IsFixed constraints preserved",
        "minimum_jacobian": cfg.minimum_jacobian,
        "margin_note": "0.1 is an explicit numerical guard against near collapse, not a measured material limit",
        "mixed_prescribed_tetrahedra": len(tets),
        "neutral_chin_rms_mm": float(1000 * np.sqrt(initial_loss)),
        "unconstrained_chin_rms_mm": float(
            1000 * np.sqrt(loss(data["pose"] / scale) * initial_loss)
        ),
        "candidates": candidates,
        "iteration_histories": histories,
        "source": {
            "path": str((source / "prepared.npz").resolve()),
            "sha256": hashlib.sha256(
                (source / "prepared.npz").read_bytes()
            ).hexdigest(),
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    for i, row in enumerate(candidates):
        cherries.log_metrics(
            {
                f"candidate_{i}/{k}": row[k]
                for k in [
                    "chin_rms_mm",
                    "min_prescribed_jacobian",
                    "rotation_degrees",
                    "translation_norm_mm",
                ]
            }
        )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
