# Copyright (c) 2026 liblaf
# ruff: noqa: E402, PLR0915, PT018
"""One fixed-reference contact increment with a 3000-step Newton-PCG cap."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
RUN = GROUP / "src/52-run-fixed-activation-contact.py"
sys.path[:0] = [str(GROUP / "src"), str(PARENT / "src")]
spec = importlib.util.spec_from_file_location("fixed_contact_run52", RUN)
assert spec is not None
assert spec.loader is not None
RUN52 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN52
spec.loader.exec_module(RUN52)

from accelerated_solvers import CachedProblem, safeguarded_newton
from experiment import Profile
from fixed_reference_contact import (
    attach_fixed_reference_contact,
    audit_fixed_reference_contact,
    build_fixed_reference_contact,
)
from joint_expression_equilibrium import FeasibleExpressionProblem
from pncg_first import run_pncg_phase
from stress_physics import (
    FacePhysics,
    configure,
    strain_to_activation_inv,
)


class Config(cherries.BaseConfig):
    source: Path = GROUP / "data/52-fixed-activation-contact"
    fixture: Path = GROUP / "data/50-fixed-reference/fixture"
    output: Path = Path("72-linear-cap-pilot")
    parent_frame: int = 19
    target_frame: int = 20
    force_atol: float = 1e-8
    linear_rtol: float = 1e-3
    linear_max_steps: int = 3000
    max_newton_steps: int = 5000
    wall_seconds: float = 180.0
    memory_fraction: float = 0.04
    minimum_free_bytes: int = 2_000_000_000
    inversion_fraction_limit: float = 0.001


def record(path: Path) -> dict[str, str]:
    item = RUN52.BASE.record(path)
    return {"path": item["path"], "sha256": item["sha256"]}


def frame(summary: dict, index: int) -> dict:
    rows = [row for row in summary["frames"] if row["index"] == index]
    assert len(rows) == 1
    return rows[0]


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.target_frame == cfg.parent_frame + 1
    assert cfg.linear_max_steps == 3000
    assert cfg.wall_seconds <= 180 and 0 < cfg.memory_fraction <= 0.04
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    free_before, total_bytes = torch.cuda.mem_get_info()
    if free_before < cfg.minimum_free_bytes:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "linear-cap-paired-pilot-v1",
                    "status": "skipped_insufficient_free_vram",
                    "gpu_memory_before_bytes": {
                        "free": free_before,
                        "total": total_bytes,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return
    torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    source_summary_path = cfg.source / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    baseline_parent = frame(source_summary, cfg.parent_frame)
    baseline_target = frame(source_summary, cfg.target_frame)
    assert not any(
        item["phase"] == "transition"
        and float(baseline_parent["beta"])
        < float(item["fraction"])
        < float(baseline_target["beta"])
        for item in source_summary["accepted"]
    ), "baseline interval contains an accepted subdivision"
    parent_path = Path(baseline_parent["checkpoint"]["path"])
    target_path = Path(baseline_target["checkpoint"]["path"])
    assert record(parent_path)["sha256"] == baseline_parent["checkpoint"]["sha256"]
    assert record(target_path)["sha256"] == baseline_target["checkpoint"]["sha256"]
    with np.load(parent_path, allow_pickle=False) as archive:
        seed = archive["u_full"].copy()
        beta_parent = float(archive["beta"])
    with np.load(target_path, allow_pickle=False) as archive:
        baseline_u = archive["u_full"].copy()
        beta_target = float(archive["beta"])
    assert beta_parent == float(baseline_parent["beta"])
    assert beta_target == float(baseline_target["beta"])
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="strain", atol=cfg.force_atol)
    with np.load(cfg.source / "endpoints.npz", allow_pickle=False) as endpoints:
        mouth = endpoints["S_mouthopen"].copy()
        smile = endpoints["S_smile"].copy()
        full_pose = endpoints["pose_mouthopen"].copy()
        pivot = endpoints["pivot"].copy()
    with np.load(
        PARENT / "data/35-forward-pruned-002/harmonic-weight.npz", allow_pickle=False
    ) as archive:
        weight = archive["weight"].copy()
    reference = build_fixed_reference_contact(
        physics.mesh, stiffness_mpa=1.3544, dhat_m=1e-4, minimum_distance_m=1e-8
    )
    full_points = reference.full_reference_points(physics.points)
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], bool)
    names = list(physics.mesh.field_data["GroupName"])
    jaw = fixed & (
        np.asarray(physics.mesh.point_data["GroupId"]) == names.index("Mandible")
    )
    expected_fixed = np.r_[fixed, np.ones(len(full_points) - len(physics.points), bool)]
    full_jaw = np.r_[jaw, reference.full_mandible_mask[len(physics.points) :]]
    full_weight = np.r_[
        weight, reference.full_mandible_mask[len(physics.points) :].astype(float)
    ]

    def boundary(alpha: float) -> np.ndarray:
        pose = alpha * full_pose
        result = np.zeros_like(full_points)
        result[full_jaw] = (
            (full_points[full_jaw] - pivot)
            @ Rotation.from_rotvec(pose[:3]).as_matrix().T
            + pivot
            + pose[3:]
            - full_points[full_jaw]
        )
        return result

    old, new = (1 - beta_parent) * full_pose, (1 - beta_target) * full_pose
    old_r, new_r = (
        Rotation.from_rotvec(old[:3]).as_matrix(),
        Rotation.from_rotvec(new[:3]).as_matrix(),
    )
    x = full_points + seed
    carried = (x - pivot - old[3:]) @ old_r @ new_r.T + pivot + new[3:]
    trial = seed + full_weight[:, None] * (carried - x)
    trial[expected_fixed] = boundary(1 - beta_target)[expected_fixed]
    model = attach_fixed_reference_contact(
        physics.forward.model, reference, torch.as_tensor(boundary(1 - beta_target))
    )
    torch.testing.assert_close(
        torch.as_tensor(trial).flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
        rtol=0,
        atol=1e-12,
    )
    ccd = float(
        reference.contact.max_step_size(
            reference.contact.state_at(torch.as_tensor(seed)),
            torch.as_tensor(seed),
            torch.as_tensor(trial - seed),
        )
    )
    assert ccd == 1.0
    activation = (1 - beta_target) * mouth + beta_target * smile
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(torch.as_tensor(activation)))
    model.set_materials(physics.materials)
    state = model.State(
        u=torch.as_tensor(trial),
        collision=reference.contact.state_at(torch.as_tensor(trial)),
    )
    problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.95)
    started = time.perf_counter()
    pncg_state, pncg = run_pncg_phase(
        CachedProblem(
            problem,
            cache_gradient=True,
            exact_curvature=False,
            wall_seconds=cfg.wall_seconds,
        ),
        state,
        atol=cfg.force_atol,
        max_step_norm=physics.forward_tolerance["newton_max_step_norm_m"],
    )
    hessian = RUN52.ProjectedContactSearch(problem, "gpu_contact")
    state, newton = safeguarded_newton(
        CachedProblem(
            hessian,
            cache_gradient=True,
            exact_curvature=True,
            wall_seconds=max(1e-3, cfg.wall_seconds - (time.perf_counter() - started)),
        ),
        pncg_state,
        atol=cfg.force_atol,
        linear_rtol=cfg.linear_rtol,
        linear_max_steps=cfg.linear_max_steps,
        max_steps=cfg.max_newton_steps,
        max_step_norm=physics.forward_tolerance["newton_max_step_norm_m"],
        armijo=1e-4,
        max_shift_attempts=8,
        max_backtracking_trials=8,
        backtracking_factor=0.5,
        preconditioner="diag",
        initial_shift_scale=0,
        shift_policy="reuse",
        reuse_shift_force_ratio=0,
        shift_scale_policy="mean_abs",
    )
    force = float(torch.linalg.vector_norm(problem.grad(state)))
    contact = {
        **audit_fixed_reference_contact(reference, state.u),
        **reference.contact.diagnostics(state.collision, state.u),
    }
    contact["contact_valid"] = (
        contact["contact_numerically_valid"] and contact["scoped_no_intersections"]
    )
    j = physics.detf(state.u[: len(physics.points)].numpy(force=True))
    inverted = int(np.count_nonzero(j <= 0))
    torch.testing.assert_close(
        state.u.flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
        rtol=0,
        atol=1e-12,
    )
    assert force <= cfg.force_atol and contact["contact_valid"]
    assert inverted <= cfg.inversion_fraction_limit * len(j)
    baseline_state = model.State(
        u=torch.as_tensor(baseline_u),
        collision=reference.contact.state_at(torch.as_tensor(baseline_u)),
    )
    baseline_energy = float(problem.fun(baseline_state))
    final_energy = float(problem.fun(state))
    difference = state.u.numpy(force=True) - baseline_u
    previous_elapsed = float(baseline_parent["diagnostics"]["elapsed_seconds"])
    result = {
        "schema": "linear-cap-paired-pilot-v1",
        "status": "completed",
        "scope": "one target increment from immutable frame 19 to frame 20; frozen 52 policy except linear_max_steps 1000 to 3000",
        "inputs": {
            "source_summary": record(source_summary_path),
            "parent_frame": record(parent_path),
            "target_frame": record(target_path),
            "endpoints": record(cfg.source / "endpoints.npz"),
        },
        "baseline": {
            "linear_max_steps": 1000,
            "elapsed_delta_seconds": float(
                baseline_target["diagnostics"]["elapsed_seconds"]
            )
            - previous_elapsed,
            "diagnostics": baseline_target["diagnostics"],
            "energy_recomputed": baseline_energy,
        },
        "pilot": {
            "linear_max_steps": cfg.linear_max_steps,
            "elapsed_seconds": time.perf_counter() - started,
            "force_l2": force,
            "energy": final_energy,
            "contact": contact,
            "minimum_J": float(j.min()),
            "inverted_cells": inverted,
            "pncg": pncg,
            "newton": newton,
            "hessian": hessian.report(),
            "ccd_carry_fraction": ccd,
        },
        "comparison": {
            "energy_difference": final_energy - baseline_energy,
            "full_displacement_vector_rms_m": float(
                np.sqrt(np.mean(np.sum(difference * difference, axis=1)))
            ),
            "full_displacement_vector_max_m": float(
                np.linalg.norm(difference, axis=1).max()
            ),
        },
        "gpu_memory_bytes": {
            "free_before": free_before,
            "total": total_bytes,
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "pilot/elapsed_seconds": result["pilot"]["elapsed_seconds"],
            "pilot/force_l2": force,
            "pilot/peak_allocated_bytes": result["gpu_memory_bytes"]["peak_allocated"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
