# ruff: noqa: CPY001, E402, EM101, PT018, TRY003
"""Bounded direct-Newton diagnostic for Smile's first active-stress proposal."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(SOURCE_GROUP / "src")]

from accelerated_solvers import accelerate_runtime
from joint_equilibrium import configure_cuda
from joint_expression_equilibrium import install_expression_runtime
from joint_expression_inputs import EyeExpressionInputs
from joint_fields import activation_stresses_mpa, project_activation_
from remote_paths import install_loader_path_relocation


class Config(cherries.BaseConfig):
    source_root: Path = EXPERIMENT.parents[4]
    pilot_dir: Path = EXPERIMENT / "data/smile-hybrid-comp07-pilot-001"
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/smile-newton-proposal-probe-001"
    forward_atol: float = 1e-12
    forward_wall_seconds: float | None = None
    linear_rtol: float = 1e-3
    max_newton_steps: int = 100


def load_fitter() -> object:
    path = SOURCE_GROUP / "src/93-fit-expressions.py"
    spec = importlib.util.spec_from_file_location("smile_probe_fitter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def first_proposal(initial: dict, runner: object) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.nn.Parameter(initial["activation"].clone())
    jaw = torch.nn.Parameter(initial["jaw_normalized"].clone())
    optimizer = torch.optim.Adam((q, jaw), lr=0.003)
    q.grad, jaw.grad = initial["gradient_q"].clone(), initial["gradient_jaw"].clone()
    optimizer.step()
    project_activation_(q, runner.CAP)
    with torch.no_grad():
        jaw.clamp_(runner.HINGE_MIN, runner.HINGE_MAX)
    return q.detach(), jaw.detach()


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    runner = load_fitter()
    initial = torch.load(
        cfg.pilot_dir / "arms/hybrid_diag/expressions/Smile/initial.pt",
        map_location="cuda",
        weights_only=False,
    )
    q, jaw = first_proposal(initial, runner)
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    physics, _ = inputs.build_physics()
    physics.runtime.tolerances["atol"] = cfg.forward_atol
    runtime = install_expression_runtime(physics)
    runtime = accelerate_runtime(
        runtime,
        "newton_diag",
        rest_points=physics.points,
        wall_seconds=cfg.forward_wall_seconds,
        linear_rtol=cfg.linear_rtol,
        max_newton_steps=cfg.max_newton_steps,
    )
    physics.runtime = runtime
    seed = torch.load(
        cfg.pilot_dir / "shared-neutral-init.pt",
        map_location="cuda",
        weights_only=False,
    )["displacement_m"]
    axis = torch.as_tensor(
        physics.base.arrays["mandible_frame_world"][:, 0], device="cuda"
    )
    pose = runner.hinge_pose(jaw, axis)
    started = time.perf_counter()
    output = physics.solve(
        skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
        active_stress=activation_stresses_mpa(q, runner.REFERENCE_MPA),
        pose=pose,
        seed=seed,
        seed_pose=torch.zeros_like(pose),
        key="Smile-direct-newton-probe",
    )
    metrics = physics.metrics(output.detach(), target_index=12)
    forward = runtime.last_forward
    receipt = {
        "schema": "smile-first-proposal-direct-newton-v1",
        "success": bool(forward["success"] and metrics["inverted_tetrahedra"] == 0),
        "scope": "One bounded direct newton_diag primal solve from the exact first Adam proposal; no inverse update accepted.",
        "seconds": time.perf_counter() - started,
        "forward": forward,
        "shape": {
            "inverted_tetrahedra": metrics["inverted_tetrahedra"],
            "detF_min": metrics["detF_min"],
        },
        "proposal": {"q_max": float(q.abs().max()), "jaw": float(jaw[0])},
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            "newton_probe/success": float(receipt["success"]),
            "newton_probe/seconds": receipt["seconds"],
        }
    )
    if not receipt["success"]:
        raise RuntimeError("direct Newton proposal probe failed")


if __name__ == "__main__":
    cherries.main(main)
