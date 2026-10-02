"""Reach a finite jaw/stress proposal by internal contact-safe seed continuation."""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path

import torch
from joint_common import GROUP, ProfileJoint, archive_sources, write_json
from joint_coupled_continuation import prepare_expression_seed
from joint_equilibrium import configure_cuda
from joint_expression_equilibrium import install_expression_runtime
from joint_expression_inputs import EyeExpressionInputs

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    inputs_dir: Path = GROUP / "data/expression-inputs-002"
    output_dir: Path = GROUP / "data/coupled-continuation-benchmark-001"
    angle_deg: float = 1.0
    active_stress_mpa: float = 0.000012328767123287673
    forward_atol: float = 1.5192003475221146e-10
    linear_rtol: float = 1e-7
    equilibrate_substeps: bool = True


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    configure_cuda()
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    physics, _ = inputs.build_physics()
    physics.runtime.tolerances["atol"] = cfg.forward_atol
    physics.runtime.tolerances["adjoint_rtol"] = cfg.linear_rtol
    runtime = install_expression_runtime(physics)
    axis = torch.as_tensor(inputs.arrays["mandible_frame_world"][:, 0])
    pivot = torch.as_tensor(inputs.arrays["mandible_pivot_m"])
    old_pose = torch.zeros(6)
    old_stress = torch.zeros((len(physics.base.active_t), 3, 3))
    one = torch.ones(())
    seed = (
        physics.solve(
            one,
            old_stress,
            old_pose,
            torch.as_tensor(inputs.neutral_displacement_m),
            seed_pose=old_pose,
            key="continuation-base",
        )
        .detach()
        .clone()
    )
    angle = torch.tensor(cfg.angle_deg, requires_grad=True)
    strength = torch.tensor(cfg.active_stress_mpa, requires_grad=True)
    pose = torch.cat((angle * (math.pi / 180) * axis, torch.zeros(3)))
    stress = strength * torch.eye(3).expand(len(old_stress), -1, -1)
    report = {
        "schema": "coupled-continuation-full-face-benchmark-v1",
        "success": False,
        "running": True,
        "angle_deg": cfg.angle_deg,
        "active_stress_mpa": cfg.active_stress_mpa,
        "base_forward": copy.deepcopy(runtime.last_forward),
        "phase": "seed_continuation",
    }
    write_json(cfg.output_dir / "summary.json", report)
    try:
        predicted, receipt = prepare_expression_seed(
            physics,
            old_stress=old_stress,
            new_stress=stress,
            old_pose=old_pose,
            new_pose=pose,
            seed=seed,
            axis=axis,
            pivot=pivot,
            linear_rtol=cfg.linear_rtol,
            equilibrate_substeps=cfg.equilibrate_substeps,
        )
        report.update(seed_continuation=receipt, phase="strict_corrector")
        write_json(cfg.output_dir / "summary.json", report)
        torch.save(
            {
                "displacement": predicted.cpu(),
                "pose": pose.detach().cpu(),
                "active_stress": stress.detach().cpu(),
            },
            cfg.output_dir / "predicted.pt",
        )
        started = time.perf_counter()
        result = physics.solve(
            one,
            stress,
            pose,
            predicted,
            seed_pose=pose.detach(),
            key="continuation-target",
        )
        report.update(
            corrector_seconds=time.perf_counter() - started,
            corrector=copy.deepcopy(runtime.last_forward),
            phase="final_adjoint",
        )
        write_json(cfg.output_dir / "summary.json", report)
        obs = torch.as_tensor(inputs.arrays["observation_node_ids"], dtype=torch.long)
        weights = torch.as_tensor(inputs.arrays["observation_weight_normalized"])
        idx = inputs.expression_names.index("MouthOpen")
        target = torch.as_tensor(inputs.arrays["target_total_displacement_m"][idx])
        mse = (weights * (result[obs] - target).square().sum(-1)).sum()
        grad_angle, grad_strength = torch.autograd.grad(mse, (angle, strength))
        assert bool(torch.isfinite(grad_angle))
        assert bool(torch.isfinite(grad_strength))
        report.update(
            success=True,
            running=False,
            phase="complete",
            mouth_open_rms_mm=float(mse.detach().sqrt() * 1000),
            mse_derivative_angle_deg=float(grad_angle),
            mse_derivative_stress_mpa=float(grad_strength),
            adjoint=copy.deepcopy(runtime.last_adjoint),
        )
        torch.save(
            {
                "displacement": result.detach().cpu(),
                "pose": pose.detach().cpu(),
                "active_stress": stress.detach().cpu(),
            },
            cfg.output_dir / "corrected.pt",
        )
        cherries.log_metrics(
            {
                "mouth_open_rms_mm": report["mouth_open_rms_mm"],
                "seed_seconds": receipt["seconds"],
                "corrector_seconds": report["corrector_seconds"],
                "accepted_substeps": receipt["accepted_substeps"],
            }
        )
    except Exception as error:
        report.update(running=False, phase="failed", error=str(error))
        if hasattr(error, "receipt"):
            report["failure_receipt"] = error.receipt
        write_json(cfg.output_dir / "summary.json", report)
        raise
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir / "summary.json")
    LOG.info("Coupled continuation and strict target equilibrium succeeded")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
