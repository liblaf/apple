"""Validate the differentiable all-muscle expression equilibrium runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries


class Config(cherries.BaseConfig):
    neutral: Path = GROUP / "data/frozen-neutral-004"
    eyes: Path = GROUP / "data/rigid-eyes-001"
    eye_forward: Path = GROUP / "data/eye-neutral-forward-002"
    output_dir: Path = GROUP / "data/expression-equilibrium-validation-001"
    activation_mpa: float = 1e-8
    pose_scale: float = 1e-7


def terminal_seed(directory: Path) -> tuple[torch.Tensor, dict]:
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["success"] is True
    assert summary["final_free_force_norm"] <= summary["force_threshold"]
    checkpoint = Path(summary["checkpoint"]["path"])
    assert sha256(checkpoint) == summary["checkpoint"]["sha256"]
    with np.load(checkpoint, allow_pickle=False) as archive:
        value = np.asarray(archive["displacement_m"], dtype=np.float64)
    assert np.isfinite(value).all()
    return torch.as_tensor(value), summary


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    from joint_frozen_neutral import load_script

    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load(cfg.neutral)
    physics, _baseline = build_eye_collision_physics(neutral, cfg.eyes)
    runtime = install_expression_runtime(physics)
    seed, forward_summary = terminal_seed(cfg.eye_forward)
    zeros = torch.zeros(6)
    active_ids = physics.base.active_t
    inactive = torch.zeros((len(active_ids), 3, 3), requires_grad=True)
    u0 = physics.solve(
        torch.ones((), dtype=torch.float64),
        inactive,
        zeros,
        seed,
        seed_pose=zeros,
        key="expression-equilibrium-zero",
    )
    baseline_force = runtime.last_forward["grad_norm"]
    assert baseline_force <= runtime.tolerances["atol"]
    active = (
        torch.eye(3).expand(len(active_ids), -1, -1).clone() * cfg.activation_mpa
    ).requires_grad_()
    pose = torch.tensor(
        [
            cfg.pose_scale,
            -cfg.pose_scale,
            cfg.pose_scale,
            cfg.pose_scale,
            0.0,
            -cfg.pose_scale,
        ],
    ).requires_grad_()
    u = physics.solve(
        torch.ones((), dtype=torch.float64),
        active,
        pose,
        u0.detach(),
        seed_pose=zeros,
        key="expression-equilibrium-active-pose",
    )
    ids = torch.as_tensor(neutral.arrays["observation_node_ids"], dtype=torch.long)
    direction = torch.tensor([0.31, -0.27, 0.19])
    loss = torch.mean(u[ids] @ direction)
    loss.backward()
    assert active.grad is not None
    assert pose.grad is not None
    assert bool(torch.isfinite(active.grad).all() and torch.isfinite(pose.grad).all())
    assert torch.linalg.vector_norm(active.grad) > 0
    assert torch.linalg.vector_norm(pose.grad) > 0
    result = {
        "schema": "joint-expression-equilibrium-validation-v1",
        "success": True,
        "input": {
            "neutral_manifest": sha256(cfg.neutral / "manifest.json"),
            "eye_forward_summary": sha256(cfg.eye_forward / "summary.json"),
            "eye_forward_checkpoint": forward_summary["checkpoint"],
        },
        "baseline_zero": {
            "accepted_state_force": baseline_force,
            "force_threshold": runtime.tolerances["atol"],
            "contact": runtime.last_forward["contact"],
        },
        "active_pose": {
            "accepted_state_force": runtime.last_forward["grad_norm"],
            "force_threshold": runtime.tolerances["atol"],
            "contact": runtime.last_forward["contact"],
            "activation_mpa": cfg.activation_mpa,
            "activation_field": "constant symmetric isotropic stress across all muscle tetrahedra; spatially smooth representative probe",
            "pose_rad_m": pose.detach().cpu().tolist(),
            "forward": runtime.last_forward,
            "adjoint": runtime.last_adjoint,
            "activation_gradient_norm": float(torch.linalg.vector_norm(active.grad)),
            "pose_gradient_norm": float(torch.linalg.vector_norm(pose.grad)),
        },
        "runtime": {
            "line_search_armijo": runtime.line_search_armijo,
            "hessian_damping_initial": runtime.hessian_damping_initial,
            "pncg_restart_interval": runtime.pncg_restart_interval,
            "max_step_norm_m": runtime.max_step_norm_m,
            "ccd_tolerance_m": physics.contact_definition["config"]["ccd_tolerance_m"],
            "ccd_min_distance_m": float(
                physics.runtime.forward.model.collision.min_distance
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", result)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
