"""Solve the approved chin pose using bounded smoothed rigid carry."""

# ruff: noqa: C901, E402, PLR0915
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Literal

import ipctk
import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path[:0] = [
    str(GROUP / "src"),
    str(ROOT / "exp/2026/09/22/solver-performance/src"),
    str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"),
]
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import configure_cuda
from mesh_step_scale import mean_rest_edge_length
from mouthopen_pose_jump import push_out
from mouthopen_rigid_seed import prepare_rigid_seed, runner
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics


class Config(cherries.BaseConfig):
    source_checkpoint: Path = GROUP / "data/inverse-mouthopen-003/initialization.pt"
    source_protocol: Path = GROUP / "data/inverse-mouthopen-003/protocol.json"
    estimate: Path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    output_dir: Path = GROUP / "data/pose-rigid-001"
    source_phase: Literal["contact", "no_contact"] = "contact"
    forward_atol: float = 1e-8
    max_newton_steps: int = 3000
    off_wall_seconds: float = 600
    no_contact_linear_max_steps: int = 1000
    source_start_from_newton: bool = False


def main(cfg: Config):
    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    configure_cuda()
    ipctk.set_num_threads(4)
    estimate = json.loads(cfg.estimate.read_text())
    source = json.loads(cfg.source_protocol.read_text())
    state = torch.load(cfg.source_checkpoint, map_location="cuda", weights_only=False)
    q = state["activation_inv"]
    if "pose_rad_m" in state:
        old_pose = state["pose_rad_m"]
    else:
        axis = torch.as_tensor(source["parameterization"]["jaw_axis"])
        old_pose = torch.cat((state["jaw"][0] * np.pi / 18 * axis, torch.zeros(3)))
    target_pose = torch.as_tensor(estimate["pose_rad_m"])
    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    baseline, strain_receipt, _ = install_active_strain(model)
    active = physics.base.active_t

    def materials(value: torch.Tensor):
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active, value)
        return result

    kappa = 0.3386
    model.set_materials(materials(q))
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=1e-7,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        newton_max_steps=cfg.max_newton_steps,
        fixed_stiffness_mpa=kappa,
    )
    archive = output / "sources"
    for folder, label in (
        (GROUP / "src", "new-neutral"),
        (ROOT / "exp/2026/09/22/solver-performance/src", "solver-performance"),
        (ROOT / "exp/2026/09/21/joint-activation-material-mandible/src", "joint"),
    ):
        shutil.copytree(
            folder,
            archive / label,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    protocol = {
        "schema": "mouthopen-approved-rigid-pose-forward-v1",
        "config": cfg.model_dump(mode="json"),
        "source_checkpoint": runner.fit.record(cfg.source_checkpoint),
        "estimate": runner.fit.record(cfg.estimate),
        "user_approved_chin_pose": True,
        "limits": {"max_rotation_deg": 1.0, "max_translation_m": 0.001},
        "old_pose_rad_m": old_pose.cpu().tolist(),
        "new_pose_rad_m": target_pose.cpu().tolist(),
        "materials": strain_receipt,
        "kappa_mpa": kappa,
        "source_hashes": {
            str(p.relative_to(archive)): sha256(p) for p in archive.rglob("*.py")
        },
        "rendering": source["rendering"],
        "blendshapes": source["sources"]["blendshapes"],
        "scope": "Forward initialization; no inverse parameter updates yet",
    }
    write_json(output / "protocol.json", protocol)
    source_relaxation = None
    source_pushout = None
    try:
        if cfg.source_phase == "contact":
            seed = runtime.primal(
                materials(q), physics.boundary(old_pose), state["displacement_m"]
            )
        else:
            relaxation_dir = output / "source-relaxation"
            relaxation_dir.mkdir()
            last_snapshot = -float("inf")

            def checkpoint(accepted: torch.Tensor, observation: dict) -> None:
                nonlocal last_snapshot
                row = dict(observation)
                if row["seconds"] - last_snapshot >= 10 or row["kind"] == "failure":
                    path = relaxation_dir / "accepted-partial.pt"
                    runner.fit.save_torch(
                        path,
                        {
                            "activation_inv": q.cpu(),
                            "pose_rad_m": old_pose.cpu(),
                            "displacement_m": accepted.cpu(),
                        },
                    )
                    row["checkpoint"] = runner.fit.record(path)
                    row["geometry"] = physics.metrics(accepted[: len(physics.points)])
                    write_json(relaxation_dir / "latest.json", row)
                    last_snapshot = row["seconds"]
                with (relaxation_dir / "force.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

            try:
                relaxed, source_relaxation = runner.relax_without_contact(
                    physics,
                    materials(q),
                    physics.boundary(old_pose),
                    state["displacement_m"],
                    atol=cfg.forward_atol,
                    step_cap=runtime.max_step_norm_m,
                    max_steps=cfg.max_newton_steps,
                    wall_seconds=cfg.off_wall_seconds,
                    linear_max_steps=cfg.no_contact_linear_max_steps,
                    start_from_newton=cfg.source_start_from_newton,
                    checkpoint=checkpoint,
                )
            except Exception as error:
                write_json(
                    relaxation_dir / "summary.json",
                    {
                        "success": False,
                        "failure": str(error),
                        "receipt": getattr(error, "receipt", None),
                    },
                )
                raise
            write_json(relaxation_dir / "summary.json", source_relaxation)
            pushed, source_pushout = push_out(
                physics, relaxed, old_pose, lambda value: value, relaxation_dir
            )
            seed = runtime.primal(materials(q), physics.boundary(old_pose), pushed)
        source_receipt = {
            "schema": "mouthopen-rigid-pose-source-corrector-v1",
            "source_phase": cfg.source_phase,
            "source_checkpoint": runner.fit.record(cfg.source_checkpoint),
            "no_contact_relaxation": source_relaxation,
            "push_out": source_pushout,
            "forward": copy.deepcopy(runtime.last_forward),
            "force_contact_converged": runtime.last_forward["success"],
            "terminal_gates": copy.deepcopy(runtime.last_forward["terminal_gates"]),
            "geometry": physics.metrics(seed[: len(physics.points)]),
        }
        write_json(output / "source-corrector.json", source_receipt)
    except Exception as error:
        write_json(
            output / "source-corrector-failure.json",
            {
                "source_phase": cfg.source_phase,
                "failure": str(error),
                "receipt": getattr(error, "receipt", None),
            },
        )
        if hasattr(runtime, "last_failed_displacement"):
            runner.fit.save_torch(
                output / "failed-source.pt",
                {
                    "activation_inv": q.cpu(),
                    "pose_rad_m": old_pose.cpu(),
                    "displacement_m": runtime.last_failed_displacement.cpu(),
                },
            )
        raise
    runner.fit.save_torch(
        output / "source.pt",
        {
            "activation_inv": q.cpu(),
            "pose_rad_m": old_pose.cpu(),
            "displacement_m": seed.cpu(),
        },
    )
    try:
        final, receipt = prepare_rigid_seed(
            physics,
            materials,
            q,
            q,
            old_pose,
            target_pose,
            seed,
            output / "continuation",
            forward_atol=cfg.forward_atol,
            max_newton_steps=cfg.max_newton_steps,
            off_wall_seconds=cfg.off_wall_seconds,
            no_contact_linear_max_steps=cfg.no_contact_linear_max_steps,
        )
        runner.fit.save_torch(
            output / "endpoint.pt",
            {
                "activation_inv": q.cpu(),
                "pose_rad_m": target_pose.cpu(),
                "displacement_m": final.cpu(),
            },
        )
        runner.fit.save_npz(
            output / "endpoint.npz",
            activation_inv=q.cpu().numpy(),
            pose_rad_m=target_pose.cpu().numpy(),
            displacement_m=final.cpu().numpy(),
            active_cell_ids=active.cpu().numpy(),
        )
        write_json(output / "summary.json", receipt)
    finally:
        cherries.log_output(output / "protocol.json")
        source_corrector = output / "source-corrector.json"
        if source_corrector.exists():
            cherries.log_output(source_corrector)
        source_relaxation_summary = output / "source-relaxation/summary.json"
        if source_relaxation_summary.exists():
            cherries.log_output(source_relaxation_summary)
        progress = output / "continuation/summary.json"
        if progress.exists():
            cherries.log_output(progress)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
