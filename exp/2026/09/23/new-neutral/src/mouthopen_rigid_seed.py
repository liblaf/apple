"""Smoothed rigid carry with bounded poses and independently converged correctors."""

# ruff: noqa: C901, EM101, PLR0915, TRY003
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mouthopen_harmonic_carry import carry_near_mandible
from mouthopen_pose_jump import push_out
from mouthopen_pose_path import pose_waypoints

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "_mouthopen_pose_jump_driver", HERE / "112-test-pose-jump.py"
)
assert spec is not None
assert spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
LOG = logging.getLogger(__name__)


@torch.no_grad()
def prepare_rigid_seed(
    physics: Any,
    materials: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_pose_rad_m: torch.Tensor,
    new_pose_rad_m: torch.Tensor,
    seed: torch.Tensor,
    output_dir: Path,
    *,
    forward_atol: float = 1e-8,
    max_newton_steps: int = 3000,
    off_wall_seconds: float = 600,
    no_contact_linear_max_steps: int = 1000,
    deadline: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Run each <=1 degree/1 mm step through carry, relaxation and contact.

    The returned displacement meets the force/contact gates. Tetrahedron
    validity is recorded separately, including inherited invalid cells.
    Initializer operations are detached from the final implicit derivative.
    """
    output = Path(output_dir)
    assert not output.exists(), output
    output.mkdir(parents=True)
    runtime = physics.runtime
    model = runtime.forward.model
    assert old_q.shape == new_q.shape
    assert old_pose_rad_m.shape == new_pose_rad_m.shape == (6,)
    assert no_contact_linear_max_steps > 0
    old_pose_rad_m = old_pose_rad_m.detach()
    new_pose_rad_m = new_pose_rad_m.detach()
    torch.testing.assert_close(
        seed.flatten()[model.dof_map.fixed_indices],
        physics.boundary(old_pose_rad_m),
        rtol=0,
        atol=1e-12,
    )
    poses, schedule = pose_waypoints(
        old_pose_rad_m.cpu().numpy(), new_pose_rad_m.cpu().numpy()
    )
    if len(poses) == 1 and not torch.equal(old_q, new_q):
        poses = np.repeat(poses, 2, axis=0)
        schedule["steps"] = [{"step": 1, "rotation_deg": 0.0, "translation_m": 0.0}]
    current_u = seed.detach().clone()
    current_pose = old_pose_rad_m
    started = time.perf_counter()
    receipt = {
        "method": "bounded-se3-harmonic-carry-contact-off-and-contact-on-correction",
        "schedule": schedule,
        "steps": [],
        "success": False,
        "status": "running",
    }

    def write():
        runner.write_json(output / "summary.json", receipt)

    def time_left() -> float | None:
        if deadline is None:
            return None
        left = deadline - time.perf_counter()
        if left <= 0:
            raise runner.ForwardConvergenceError(
                "declared forward wall budget exhausted"
            )
        return left

    def save_stage(
        directory: Path, name: str, u: torch.Tensor, q: torch.Tensor, pose: torch.Tensor
    ):
        path = directory / f"{name}.pt"
        runner.fit.save_torch(
            path,
            {
                "activation_inv": q.cpu(),
                "pose_rad_m": pose.cpu(),
                "displacement_m": u.cpu(),
            },
        )
        geometry = physics.metrics(u[: len(physics.points)])
        return {"checkpoint": runner.fit.record(path), "geometry": geometry}

    write()
    try:
        for index in range(1, len(poses)):
            time_left()
            step_start = time.perf_counter()
            step_dir = output / f"step-{index:03d}"
            step_dir.mkdir()
            target_pose = torch.as_tensor(
                poses[index], device=seed.device, dtype=seed.dtype
            )
            target_q = (
                new_q.detach()
                if index == len(poses) - 1
                else torch.lerp(old_q, new_q, index / (len(poses) - 1)).detach()
            )
            row = {
                "index": index,
                **schedule["steps"][index - 1],
                "old_pose_rad_m": current_pose.cpu().tolist(),
                "new_pose_rad_m": target_pose.cpu().tolist(),
                "status": "carry",
            }
            receipt["steps"].append(row)
            write()
            LOG.info(
                "Rigid step %d/%d: %.6f degrees, %.6f mm",
                index,
                len(poses) - 1,
                row["rotation_deg"],
                row["translation_m"] * 1000,
            )
            if torch.equal(current_pose, target_pose):
                carried, carry = (
                    current_u.detach().clone(),
                    {"method": "unchanged-pose-material-only", "seconds": 0.0},
                )
            else:
                carried, carry = carry_near_mandible(
                    physics, current_u, current_pose, target_pose, lambda value: value
                )
            row["carry"] = {
                **carry,
                **save_stage(step_dir, "carried", carried, target_q, target_pose),
            }
            row["status"] = "no_contact_relaxation"
            write()
            last_snapshot = -float("inf")

            def checkpoint(
                u: torch.Tensor,
                observation: dict,
                step_dir: Path = step_dir,
                target_q: torch.Tensor = target_q,
                target_pose: torch.Tensor = target_pose,
            ):
                nonlocal last_snapshot
                observation = dict(observation)
                if (
                    observation["seconds"] - last_snapshot >= 10
                    or observation["kind"] == "failure"
                ):
                    observation.update(
                        save_stage(
                            step_dir, "relaxed-partial", u, target_q, target_pose
                        )
                    )
                    last_snapshot = observation["seconds"]
                    runner.write_json(step_dir / "latest-no-contact.json", observation)
                with (step_dir / "force.jsonl").open("a") as stream:
                    stream.write(json.dumps(observation, allow_nan=False) + "\n")

            relaxed, relaxation = runner.relax_without_contact(
                physics,
                materials(target_q),
                physics.boundary(target_pose),
                carried,
                atol=forward_atol,
                step_cap=runtime.max_step_norm_m,
                max_steps=max_newton_steps,
                wall_seconds=min(off_wall_seconds, time_left())
                if deadline
                else off_wall_seconds,
                linear_max_steps=no_contact_linear_max_steps,
                checkpoint=checkpoint,
            )
            row["no_contact_relaxation"] = {
                **relaxation,
                **save_stage(step_dir, "relaxed", relaxed, target_q, target_pose),
            }
            row["status"] = "push_out"
            write()
            pushed, projection = push_out(
                physics, relaxed, target_pose, lambda value: value, step_dir
            )
            row["push_out"] = {
                **projection,
                **save_stage(step_dir, "pushed", pushed, target_q, target_pose),
            }
            row["status"] = "contact_equilibrium"
            write()
            final = runtime.primal(
                materials(target_q), physics.boundary(target_pose), pushed
            )
            row["proposal"] = {
                "forward": copy.deepcopy(runtime.last_forward),
                **save_stage(step_dir, "final", final, target_q, target_pose),
            }
            row["valid_forward"] = (
                row["proposal"]["geometry"]["inverted_tetrahedra"] == 0
            )
            row["seconds"] = time.perf_counter() - step_start
            row["status"] = "force_contact_converged"
            current_pose, current_u = target_pose, final.detach().clone()
            receipt["last_completed"] = save_stage(
                output, "endpoint", current_u, target_q, target_pose
            )
            write()
            LOG.info(
                "Rigid step %d complete: %.6g N, %d inverted tetrahedra",
                index,
                runtime.last_forward["grad_norm"] * 1e6,
                row["proposal"]["geometry"]["inverted_tetrahedra"],
            )
        receipt["success"] = True
        receipt["status"] = "force_contact_converged"
        receipt["seconds"] = time.perf_counter() - started
        if receipt["steps"]:
            receipt["proposal"] = receipt["steps"][-1]["proposal"]
            receipt["valid_forward"] = receipt["steps"][-1]["valid_forward"]
        else:
            current_u = runtime.primal(
                materials(new_q), physics.boundary(new_pose_rad_m), current_u
            )
            receipt["proposal"] = {
                "forward": copy.deepcopy(runtime.last_forward),
                **save_stage(output, "endpoint", current_u, new_q, new_pose_rad_m),
            }
            receipt["valid_forward"] = (
                receipt["proposal"]["geometry"]["inverted_tetrahedra"] == 0
            )
        write()
    except Exception as error:
        receipt["status"] = "failed"
        receipt["seconds"] = time.perf_counter() - started
        receipt["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": getattr(error, "receipt", None),
        }
        write()
        raise
    return current_u, receipt
