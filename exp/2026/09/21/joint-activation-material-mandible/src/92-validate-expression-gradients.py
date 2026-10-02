"""Finite-difference checks of real-mesh active-stress and jaw-pose gradients."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries


class Config(cherries.BaseConfig):
    neutral: Path = GROUP / "data/frozen-neutral-004"
    eyes: Path = GROUP / "data/rigid-eyes-001"
    eye_forward: Path = GROUP / "data/eye-neutral-forward-002"
    output_dir: Path = GROUP / "data/expression-gradient-validation-001"
    activation_mpa: float = 2e-6
    activation_fd_step_mpa: float = 1e-6
    pose_scale: float = 1e-7
    pose_fd_step: float = 1e-6
    relative_error_limit: float = 5e-2
    validation_force_atol: float = 1e-12


def seed(path: Path) -> torch.Tensor:
    summary = json.loads((path / "summary.json").read_text())
    assert summary["success"] is True
    checkpoint = Path(summary["checkpoint"]["path"])
    assert sha256(checkpoint) == summary["checkpoint"]["sha256"]
    with np.load(checkpoint, allow_pickle=False) as archive:
        return torch.as_tensor(np.asarray(archive["displacement_m"], dtype=np.float64))


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load(cfg.neutral)
    physics, _ = build_eye_collision_physics(neutral, cfg.eyes)
    runtime = install_expression_runtime(physics)
    # Validation-only solve accuracy: distinguish directional-response error
    # from the production 0.15 mN force stopping budget.
    runtime.tolerances["atol"] = cfg.validation_force_atol
    base_seed, zeros = seed(cfg.eye_forward), torch.zeros(6)
    count = len(physics.base.active_t)
    active = (
        torch.eye(3).expand(count, -1, -1).clone() * cfg.activation_mpa
    ).requires_grad_()
    pose = torch.tensor(
        [
            cfg.pose_scale,
            -cfg.pose_scale,
            cfg.pose_scale,
            cfg.pose_scale,
            0.0,
            -cfg.pose_scale,
        ]
    ).requires_grad_()
    # A physical per-tet stress direction: central differences perturb every
    # muscle tet by ±1 MPa microstress around a PSD 2 MPa microstress baseline.
    active_direction = torch.eye(3).expand(count, -1, -1).clone()
    pose_direction = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.5])
    pose_direction /= torch.linalg.vector_norm(pose_direction)
    ids = torch.as_tensor(neutral.arrays["observation_node_ids"], dtype=torch.long)
    view = torch.tensor([0.31, -0.27, 0.19])

    def objective(u: torch.Tensor) -> torch.Tensor:
        return 1e6 * torch.mean(u[ids] @ view)

    u = physics.solve(
        torch.ones(()), active, pose, base_seed, seed_pose=zeros, key="fd-base"
    )
    value = objective(u)
    value.backward()
    assert active.grad is not None
    assert pose.grad is not None
    predicted_active = float(torch.sum(active.grad * active_direction))
    predicted_pose = float(torch.sum(pose.grad * pose_direction))

    def evaluate(a: torch.Tensor, p: torch.Tensor, key: str) -> float:
        candidate = physics.solve(
            torch.ones(()), a, p, u.detach(), seed_pose=pose.detach(), key=key
        )
        return float(objective(candidate))

    ap = evaluate(
        active.detach() + cfg.activation_fd_step_mpa * active_direction,
        pose.detach(),
        "fd-active-plus",
    )
    am = evaluate(
        active.detach() - cfg.activation_fd_step_mpa * active_direction,
        pose.detach(),
        "fd-active-minus",
    )
    pp = evaluate(
        active.detach(),
        pose.detach() + cfg.pose_fd_step * pose_direction,
        "fd-pose-plus",
    )
    pm = evaluate(
        active.detach(),
        pose.detach() - cfg.pose_fd_step * pose_direction,
        "fd-pose-minus",
    )
    fd_active = (ap - am) / (2 * cfg.activation_fd_step_mpa)
    fd_pose = (pp - pm) / (2 * cfg.pose_fd_step)

    def error(predicted: float, finite_difference: float) -> float:
        return abs(predicted - finite_difference) / max(
            abs(predicted), abs(finite_difference), 1e-12
        )

    active_error, pose_error = (
        error(predicted_active, fd_active),
        error(predicted_pose, fd_pose),
    )
    success = (
        active_error <= cfg.relative_error_limit
        and pose_error <= cfg.relative_error_limit
    )
    result = {
        "schema": "joint-expression-gradient-validation-v1",
        "success": success,
        "relative_error_limit": cfg.relative_error_limit,
        "activation": {
            "step_mpa": cfg.activation_fd_step_mpa,
            "predicted": predicted_active,
            "finite_difference": fd_active,
            "relative_error": active_error,
        },
        "pose": {
            "step": cfg.pose_fd_step,
            "predicted": predicted_pose,
            "finite_difference": fd_pose,
            "relative_error": pose_error,
        },
        "base_forward": runtime.last_forward,
        "adjoint": runtime.last_adjoint,
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(
                Path(__file__).with_name("joint_expression_equilibrium.py").resolve()
            ): sha256(
                Path(__file__).with_name("joint_expression_equilibrium.py").resolve()
            ),
            str(
                Path(__file__).with_name("joint_rigid_eye_contact.py").resolve()
            ): sha256(Path(__file__).with_name("joint_rigid_eye_contact.py").resolve()),
        },
    }
    write_json(cfg.output_dir / "summary.json", result)
    cherries.log_output(cfg.output_dir)
    assert success, result


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
