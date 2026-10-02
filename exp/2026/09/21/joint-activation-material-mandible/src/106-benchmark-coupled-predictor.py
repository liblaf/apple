"""Test coupled jaw/tissue prediction and strict contact PNCG on the full face."""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, write_json
from joint_coupled_predictor import (
    audit_coupled_motion,
    equilibrium_predictor,
    rotation_sagitta,
)
from joint_equilibrium import configure_cuda
from joint_expression_equilibrium import install_expression_runtime
from joint_expression_inputs import EyeExpressionInputs

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    inputs_dir: Path = GROUP / "data/expression-inputs-002"
    output_dir: Path = GROUP / "data/coupled-predictor-benchmark-001"
    maximum_angle_deg: float = 1.0
    minimum_angle_deg: float = 0.015625
    active_stress_mpa: float = 0.0
    forward_atol: float = 1.5192003475221146e-10
    linear_rtol: float = 1e-7


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    configure_cuda()
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    physics, _ = inputs.build_physics()
    physics.runtime.tolerances["atol"] = cfg.forward_atol
    physics.runtime.tolerances["adjoint_rtol"] = cfg.linear_rtol
    runtime = install_expression_runtime(physics)
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    axis = torch.as_tensor(inputs.arrays["mandible_frame_world"][:, 0])
    pivot = torch.as_tensor(inputs.arrays["mandible_pivot_m"])
    zero_pose = torch.zeros(6)
    neutral = torch.as_tensor(inputs.neutral_displacement_m)
    count = len(physics.base.active_t)
    old_stress = torch.zeros((count, 3, 3))
    one = torch.ones(())
    base = (
        physics.solve(
            one, old_stress, zero_pose, neutral, seed_pose=zero_pose, key="coupled-base"
        )
        .detach()
        .clone()
    )
    old_u = physics.full_skull.extend_seed(base, zero_pose)
    old_materials = physics.expression_materials(
        skin_multiplier=one, active_stress=old_stress
    )
    radial = collision.vertices - pivot
    radial = radial - (radial @ axis)[:, None] * axis
    # All collision points bound the radius of the rotating mandible subset.
    radius = float(torch.linalg.vector_norm(radial, dim=-1).max())
    report = {
        "schema": "coupled-predictor-full-face-benchmark-v1",
        "success": False,
        "running": True,
        "base_forward": copy.deepcopy(runtime.last_forward),
        "max_radius_bound_m": radius,
        "trials": [],
        "force_tolerance": cfg.forward_atol,
        "linear_rtol": cfg.linear_rtol,
        "active_stress_mpa_at_maximum_angle": cfg.active_stress_mpa,
        "source_geometry": physics.full_skull_receipt(),
    }
    write_json(cfg.output_dir / "summary.json", report)
    angle = cfg.maximum_angle_deg
    while angle >= cfg.minimum_angle_deg:
        radians = math.radians(angle)
        pose = torch.cat((radians * axis, torch.zeros(3)))
        stress = (
            cfg.active_stress_mpa
            * angle
            / cfg.maximum_angle_deg
            * torch.eye(3).expand(count, -1, -1).clone()
        )
        new_materials = physics.expression_materials(
            skin_multiplier=one, active_stress=stress
        )
        fixed = physics.boundary(pose)
        frozen = old_u.clone()
        frozen.flatten()[model.dof_map.fixed_indices] = fixed
        boundary_only = audit_coupled_motion(collision, old_u, frozen)
        LOG.info(
            "Testing %.6g degrees: frozen-soft CCD fraction %.6g",
            angle,
            boundary_only["ccd_fraction"],
        )
        predicted, predictor = equilibrium_predictor(
            model=model,
            solver=runtime.solver,
            displacement=old_u,
            old_materials=old_materials,
            new_materials=new_materials,
            fixed_target=fixed,
            linear_rtol=cfg.linear_rtol,
        )
        geometry = audit_coupled_motion(
            collision,
            old_u,
            predicted,
            rotation_margin_m=rotation_sagitta(radius, radians),
        )
        trial = {
            "angle_deg": angle,
            "boundary_only": boundary_only,
            "predictor": predictor,
            "coupled_geometry": geometry,
        }
        report["trials"].append(trial)
        write_json(cfg.output_dir / "summary.json", report)
        LOG.info(
            "Coupled %.6g degrees: admitted=%s fraction=%.6g prediction=%.3fs",
            angle,
            geometry["admitted"],
            geometry["ccd_fraction"],
            predictor["seconds"],
        )
        if geometry["admitted"]:
            torch.save(
                {
                    "displacement": predicted.cpu(),
                    "pose": pose.cpu(),
                    "active_stress": stress.cpu(),
                },
                cfg.output_dir / "predicted.pt",
            )
            started = time.perf_counter()
            result = (
                physics.solve(
                    one,
                    stress,
                    pose,
                    predicted[: len(base)],
                    seed_pose=pose,
                    key="coupled-corrector",
                )
                .detach()
                .clone()
            )
            trial["corrector_seconds"] = time.perf_counter() - started
            trial["corrector"] = copy.deepcopy(runtime.last_forward)
            obs = torch.as_tensor(
                inputs.arrays["observation_node_ids"], dtype=torch.long
            )
            weights = torch.as_tensor(inputs.arrays["observation_weight_normalized"])
            idx = inputs.expression_names.index("MouthOpen")
            target = torch.as_tensor(inputs.arrays["target_total_displacement_m"][idx])
            trial["mouth_open_rms_mm"] = float(
                (weights * (result[obs] - target).square().sum(-1)).sum().sqrt() * 1000
            )
            torch.save(
                {
                    "displacement": result.cpu(),
                    "pose": pose.cpu(),
                    "active_stress": stress.cpu(),
                },
                cfg.output_dir / "corrected.pt",
            )
            report.update(success=True, running=False, accepted_angle_deg=angle)
            write_json(cfg.output_dir / "summary.json", report)
            cherries.log_metrics(
                {
                    "accepted_angle_deg": angle,
                    "predictor_seconds": predictor["seconds"],
                    "corrector_seconds": trial["corrector_seconds"],
                    "mouth_open_rms_mm": trial["mouth_open_rms_mm"],
                }
            )
            break
        angle /= 2
    report["running"] = False
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir / "summary.json")
    assert report["success"]


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
