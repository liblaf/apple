"""Validate strict PNCG equilibria and implicit expression gradients."""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from joint_common import (
    GROUP,
    HISTORICAL,
    ROOT,
    ProfileJoint,
    archive_sources,
    sha256,
    write_json,
)
from joint_expression_precise_hvp import install_precise_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/precise-hvp-expression-validation-001"


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    source_hashes = {
        str(path.resolve()): sha256(path)
        for folder in (GROUP / "src", ROOT / "src/liblaf/apple", HISTORICAL)
        for path in sorted(folder.rglob("*.py"))
        if path.name != "93-fit-expressions.py"
    }
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    neutral = FrozenNeutral.load(GROUP / "data/frozen-neutral-004")
    physics, _ = build_eye_collision_physics(neutral, GROUP / "data/rigid-eyes-001")
    runtime = install_precise_expression_runtime(physics)
    runtime.tolerances["atol"] = 1e-12
    loaded_paths = {
        str(Path(module.__file__).resolve())
        for module in tuple(sys.modules.values())
        if getattr(module, "__file__", None)
    }
    loaded_paths.update(
        str((GROUP / "src" / name).resolve())
        for name in ("68-run-simple-skin-forward.py", Path(__file__).name)
    )
    implementation = {
        path: digest for path, digest in source_hashes.items() if path in loaded_paths
    }
    seed_summary_path = GROUP / "data/eye-neutral-forward-002/summary.json"
    seed_summary = json.loads(seed_summary_path.read_text())
    checkpoint = Path(seed_summary["checkpoint"]["path"])
    assert sha256(checkpoint) == seed_summary["checkpoint"]["sha256"]
    with np.load(checkpoint) as arrays:
        seed = torch.as_tensor(np.asarray(arrays["displacement_m"], dtype=np.float64))
    count = len(physics.base.active_t)
    assert count == 288235, count
    direction = torch.eye(3).expand(count, -1, -1).clone()
    activation = (2e-6 * direction).requires_grad_()
    pose = torch.tensor([1e-7, -1e-7, 1e-7, 1e-7, 0.0, -1e-7]).requires_grad_()
    pose_direction = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.5])
    pose_direction /= torch.linalg.vector_norm(pose_direction)
    observation_ids = torch.as_tensor(neutral.arrays["observation_node_ids"])
    observation_direction = torch.tensor([0.31, -0.27, 0.19])
    receipts = []
    report = {
        "schema": "precise-hvp-expression-gradient-validation-v1",
        "success": False,
        "implementation_sha256": implementation,
        "seed": {
            "summary": str(seed_summary_path),
            "summary_sha256": sha256(seed_summary_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "active_tet_count": count,
        "force_tolerance": 1e-12,
        "finite_difference_step": 1e-6,
        "relative_gradient_error_threshold": 0.05,
        "solves": receipts,
    }
    write_json(cfg.output_dir / "summary.json", report)

    def objective(displacement: torch.Tensor) -> torch.Tensor:
        return 1e6 * torch.mean(displacement[observation_ids] @ observation_direction)

    def solve(
        active: torch.Tensor,
        jaw: torch.Tensor,
        initial: torch.Tensor,
        seed_pose: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        started = time.perf_counter()
        runtime.diagnostic_directory = cfg.output_dir / key
        LOG.info("Starting validation solve %s", key)
        try:
            result = physics.solve(
                torch.ones(()), active, jaw, initial, seed_pose=seed_pose, key=key
            )
        except BaseException as error:
            failure = {
                "key": key,
                "success": False,
                "error": repr(error),
                "seconds": time.perf_counter() - started,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            write_json(cfg.output_dir / f"{key}-receipt.json", failure)
            report["failure"] = failure
            write_json(cfg.output_dir / "summary.json", report)
            raise
        receipt = {
            "key": key,
            "seconds": time.perf_counter() - started,
            "objective": float(objective(result).detach()),
            "forward": copy.deepcopy(runtime.last_forward),
        }
        target = cfg.output_dir / f"{key}-equilibrium.npz"
        np.savez_compressed(
            target,
            displacement_m=result.detach().cpu().numpy(),
            mandible_pose=jaw.detach().cpu().numpy(),
        )
        receipt["checkpoint"] = {"path": str(target), "sha256": sha256(target)}
        write_json(cfg.output_dir / f"{key}-receipt.json", receipt)
        receipts.append(receipt)
        write_json(cfg.output_dir / "summary.json", report)
        forward = receipt["forward"]
        assert forward["success"], forward
        assert forward["grad_norm"] <= 1e-12, forward
        assert forward["contact"]["contact_numerically_valid"], forward
        assert all(forward.get("terminal_gates", {}).values()), forward
        LOG.info(
            "Completed %s: force %.12g, objective %.12g, steps %s",
            key,
            forward["grad_norm"],
            receipt["objective"],
            forward["steps"],
        )
        return result

    base = solve(activation, pose, seed, torch.zeros(6), "base")
    objective(base).backward()
    assert activation.grad is not None
    assert pose.grad is not None
    predicted_activation = float((activation.grad * direction).sum())
    predicted_pose = float((pose.grad * pose_direction).sum())
    report["adjoint"] = copy.deepcopy(runtime.last_adjoint)
    report["predicted_gradients"] = {
        "activation": predicted_activation,
        "pose": predicted_pose,
    }
    write_json(cfg.output_dir / "summary.json", report)
    h = 1e-6
    a, p, u = activation.detach(), pose.detach(), base.detach()
    a_plus = float(objective(solve(a + h * direction, p, u, p, "activation-plus")))
    a_minus = float(objective(solve(a - h * direction, p, u, p, "activation-minus")))
    p_plus = float(objective(solve(a, p + h * pose_direction, u, p, "pose-plus")))
    p_minus = float(objective(solve(a, p - h * pose_direction, u, p, "pose-minus")))
    fd_activation, fd_pose = (a_plus - a_minus) / (2 * h), (p_plus - p_minus) / (2 * h)

    def comparison(predicted: float, finite_difference: float) -> dict[str, float]:
        return {
            "predicted": predicted,
            "fd": finite_difference,
            "relative_error": abs(predicted - finite_difference)
            / max(abs(predicted), abs(finite_difference), 1e-12),
        }

    report["activation"] = comparison(predicted_activation, fd_activation)
    report["pose"] = comparison(predicted_pose, fd_pose)
    report["success"] = all(
        report[key]["relative_error"] <= 0.05 for key in ("activation", "pose")
    )
    # All dependencies must remain exactly those actually archived before the run.
    for source, digest in implementation.items():
        assert sha256(Path(source)) == digest, source
    write_json(cfg.output_dir / "summary.json", report)
    cherries.log_output(cfg.output_dir)
    LOG.info(
        "Validation results: activation=%s pose=%s",
        report["activation"],
        report["pose"],
    )
    assert report["success"], report


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
