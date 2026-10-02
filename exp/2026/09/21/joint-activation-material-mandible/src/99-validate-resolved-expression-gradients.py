"""Validate implicit expression gradients at resolved physical step scales."""

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
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/expression-scale-gradient-validation-001"


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
    runtime = install_expression_runtime(physics)
    force_tolerance = 1.5192003475221146e-10
    runtime.tolerances["atol"] = force_tolerance
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
    refined_checkpoint = (
        GROUP / "data/precise-hvp-expression-validation-001/base/accepted-latest.npz"
    )
    refined_sha256 = sha256(refined_checkpoint)
    with np.load(refined_checkpoint) as arrays:
        full_seed = np.asarray(arrays["full_displacement_m"], dtype=np.float64)
        seed = torch.as_tensor(full_seed[: len(seed)].copy())
    fixed_indices = physics.runtime.forward.model.dof_map.fixed_indices
    assert torch.allclose(
        torch.as_tensor(full_seed).flatten()[fixed_indices],
        physics.boundary(pose).detach(),
        atol=1e-14,
        rtol=0.0,
    )
    pose_direction = torch.tensor([0.2, -0.3, 0.1, 0.4, -0.2, 0.5])
    pose_direction /= torch.linalg.vector_norm(pose_direction)
    observation_ids = torch.as_tensor(neutral.arrays["observation_node_ids"])
    observation_direction = torch.tensor([0.31, -0.27, 0.19])
    receipts = []
    report = {
        "schema": "scale-resolved-expression-gradient-validation-v1",
        "success": False,
        "implementation_sha256": implementation,
        "seed": {
            "summary": str(seed_summary_path),
            "summary_sha256": sha256(seed_summary_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "active_tet_count": count,
        "refined_seed": {
            "path": str(refined_checkpoint),
            "sha256": refined_sha256,
            "pose": pose.detach().cpu().tolist(),
        },
        "force_tolerance": force_tolerance,
        "activation_fd_steps_mpa": [1e-4, 3e-5],
        "pose_fd_steps": [1e-5, 3e-6],
        "probe_note": "Signed central activation probes extend beyond PSD constraints to test the physical derivative; no optimizer update uses these probes.",
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
        assert forward["grad_norm"] <= force_tolerance, forward
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

    base = solve(activation, pose, seed, pose.detach(), "base")
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
    a, p, u = activation.detach(), pose.detach(), base.detach()

    def comparison(predicted: float, finite_difference: float) -> dict[str, float]:
        return {
            "predicted": predicted,
            "fd": finite_difference,
            "relative_error": abs(predicted - finite_difference)
            / max(abs(predicted), abs(finite_difference), 1e-12),
        }

    for parameter, steps, predicted in (
        ("activation", (1e-4, 3e-5), predicted_activation),
        ("pose", (1e-5, 3e-6), predicted_pose),
    ):
        rows = []
        report[parameter] = {"rows": rows}
        for index, h in enumerate(steps):
            if parameter == "activation":
                plus = solve(a + h * direction, p, u, p, f"activation-{index}-plus")
                minus = solve(a - h * direction, p, u, p, f"activation-{index}-minus")
            else:
                plus = solve(a, p + h * pose_direction, u, p, f"pose-{index}-plus")
                minus = solve(a, p - h * pose_direction, u, p, f"pose-{index}-minus")
            fd = float((objective(plus) - objective(minus)) / (2 * h))
            rows.append({"step": h, **comparison(predicted, fd)})
            write_json(cfg.output_dir / "summary.json", report)
            LOG.info("%s step %.12g: %s", parameter, h, rows[-1])
        report[parameter]["plateau_relative_error"] = comparison(
            rows[0]["fd"], rows[1]["fd"]
        )["relative_error"]
    report["success"] = all(
        report[key]["plateau_relative_error"] <= 0.05
        and all(row["relative_error"] <= 0.05 for row in report[key]["rows"])
        for key in ("activation", "pose")
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
