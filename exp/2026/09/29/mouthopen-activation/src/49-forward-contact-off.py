# Copyright (c) 2026 liblaf
"""Continue the historical MouthOpen bulk model without contact constraints."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS = ROOT / "exp/2026/09/21/stress-activation-loss/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.extend(map(str, (STRESS, SOLVERS, JOINT)))

from accelerated_solvers import CachedProblem, safeguarded_newton  # noqa: E402
from experiment import Profile  # noqa: E402
from joint_equilibrium import ForwardConvergenceError  # noqa: E402
from mouthopen_geometry import MouthOpenGeometry  # noqa: E402
from stress_physics import FacePhysics, configure  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pruned_forward_35", Path(__file__).with_name("35-forward-pruned-mouthopen.py")
)
assert spec is not None
assert spec.loader is not None
PARENT = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PARENT
spec.loader.exec_module(PARENT)
record = PARENT.record
write = PARENT.write
freeze = PARENT.freeze

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("49-forward-contact-off")
    fixture: Path = GROUP / "data/30-pruned-fixture"
    parent: Path = GROUP / "data/48-forward-local-folds"
    harmonic_weight: Path = GROUP / "data/35-forward-pruned-002/harmonic-weight.npz"
    force_atol: float = 1e-10
    linear_rtol: float = 1e-3
    linear_max_steps: int = 3000
    max_newton_steps: int = 100
    wall_seconds: float = 1200
    initial_pose_step: float = 0.025
    minimum_pose_step: float = 1e-5
    maximum_pose_step: float = 0.1
    maximum_inverted_cell_fraction: float = 0.001


class ContactOffProblem(ForwardProblem):
    """Give Newton its full proposed step; line search still checks energy."""

    def __init__(self, physics: FacePhysics) -> None:
        super().__init__(model=physics.forward.model)

    def max_step_size(self, _state: Any, direction: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(1.0, dtype=direction.dtype, device=direction.device)


@torch.no_grad()
def main(cfg: Config) -> None:  # noqa: PLR0915
    assert 0 < cfg.maximum_inverted_cell_fraction <= 0.001
    assert 0 < cfg.minimum_pose_step <= cfg.initial_pose_step <= cfg.maximum_pose_step
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    parent_path = cherries.input(cfg.parent / "final.npz")
    parent_summary_path = cherries.input(cfg.parent / "summary.json")
    weight_path = cherries.input(cfg.harmonic_weight)
    volume_path = cherries.input(cfg.fixture / "volume.vtu")
    skin_path = cherries.input(cfg.fixture / "skin.vtp")
    fixture_summary_path = cherries.input(cfg.fixture / "summary.json")
    mapping_path = cherries.input(cfg.fixture / "mapping.npz")
    pose_path = cherries.input(GROUP / "data/10-mandible/prepared.npz")
    parent_summary = json.loads(parent_summary_path.read_text())
    assert parent_summary["status"] == "blocked_at_minimum_pose_step"
    assert parent_summary["final_checkpoint"]["sha256"] == record(parent_path)["sha256"]
    with np.load(parent_path, allow_pickle=False) as source:
        saved_u = np.asarray(source["displacement"], dtype=np.float64).copy()
        saved_pose = np.asarray(source["pose"], dtype=np.float64).copy()
        fraction = float(source["fraction"])
    with np.load(weight_path, allow_pickle=False) as source:
        weight = np.asarray(source["weight"], dtype=np.float64).copy()
    with np.load(pose_path, allow_pickle=False) as source:
        full_pose = np.asarray(source["pose"], dtype=np.float64).copy()
        pivot = np.asarray(source["pivot"], dtype=np.float64).copy()
    assert fraction == parent_summary["completed_pose_fraction"]
    np.testing.assert_allclose(saved_pose, fraction * full_pose, rtol=0, atol=1e-14)

    summary = {
        "schema": "pruned-mouthopen-contact-off-forward-v1",
        "status": "initializing",
        "numerically_converged_full_pose": False,
        "orientation_valid": False,
        "physical_validity_claim": False,
        "completed_pose_fraction": fraction,
        "attempts": [],
        "inputs": {
            "parent_checkpoint": record(parent_path),
            "parent_summary": record(parent_summary_path),
            "parent_harmonic_weight": record(weight_path),
            "volume": record(volume_path),
            "skin": record(skin_path),
            "fixture_summary": record(fixture_summary_path),
            "fixture_mapping": record(mapping_path),
            "full_pose": record(pose_path),
        },
        "config": cfg.model_dump(mode="json"),
        "model": "historical no-skin bulk active strain at S=0; prescribed jaw; no contact forces or geometric collision feasibility constraint",
        "inversion_policy": "finite inversions allowed; more than 0.1% of remaining cells stops the trial; no physical validity claim for inverted states",
        "collision_policy": "No CCD or intersection gate in carry or Newton. Report complete FEM boundary self-intersections at trial and accepted endpoints only; intersecting states are physically unvalidated.",
    }
    write(out / "summary.json", summary)
    configure()
    summary["runtime"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "warp": version("warp-lang"),
        "cupy": version("cupy-cuda13x"),
        "gpu": torch.cuda.get_device_name(),
    }
    physics = FacePhysics(cfg.fixture, activation_model="strain", atol=cfg.force_atol)
    physics.target = np.asarray(physics.mesh.point_data["MouthOpen"]).copy()
    model, state = physics.forward.model, physics.forward.state
    assert model.collision is None
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    group_names = [str(name) for name in physics.mesh.field_data["GroupName"]]
    jaw = fixed & (
        np.asarray(physics.mesh.point_data["GroupId"]) == group_names.index("Mandible")
    )
    assert not np.any(fixed[physics.tets].all(axis=1))
    assert weight.shape == (len(physics.points),)
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.numpy(force=True),
        np.flatnonzero(np.repeat(fixed, 3)),
    )
    assert np.allclose(weight[jaw], 1, rtol=0, atol=1e-8)
    assert np.allclose(weight[fixed & ~jaw], 0, rtol=0, atol=1e-8)
    materials = model.get_materials()
    materials["muscle"]["activation_inv"] = torch.zeros((physics.mesh.n_cells, 6))
    model.set_materials(materials)
    max_inverted = int(np.floor(cfg.maximum_inverted_cell_fraction * len(physics.tets)))
    geometry = MouthOpenGeometry(physics.mesh)
    problem = ContactOffProblem(physics)
    total_volume = float(physics.volumes_all.sum())
    skin_ids = np.asarray(physics.skin.point_data["GlobalPointId"], dtype=np.int64)
    skin_triangles = np.asarray(physics.skin.faces).reshape(-1, 4)[:, 1:]
    skin_rest = physics.points[skin_ids]
    tri = skin_rest[skin_triangles]
    areas = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        / 2
    )
    skin_mass = np.zeros(len(skin_ids))
    np.add.at(skin_mass, skin_triangles.ravel(), np.repeat(areas / 3, 3))
    skin_mass /= skin_mass.sum()
    summary["maximum_allowed_inverted_cells"] = max_inverted

    def metrics(u: torch.Tensor) -> dict:
        u_np = u.numpy(force=True)
        j = physics.detf(u_np)
        assert np.isfinite(j).all()
        inverted = j <= 0
        error = u_np[skin_ids] - physics.target[skin_ids]
        return {
            "minimum_J": float(j.min()),
            "inverted_cells": int(inverted.sum()),
            "inverted_cell_fraction": float(inverted.mean()),
            "inverted_rest_volume_fraction": float(
                physics.volumes_all[inverted].sum() / total_volume
            ),
            "fit_rms_mm": float(1000 * np.sqrt(np.sum(skin_mass[:, None] * error**2))),
            "force_norm": float(torch.linalg.vector_norm(problem.grad(state))),
            **geometry.audit(u_np),
        }

    initial_boundary = np.zeros_like(physics.points)
    old_r = Rotation.from_rotvec(saved_pose[:3]).as_matrix()
    initial_boundary[jaw] = (
        (physics.points[jaw] - pivot) @ old_r.T
        + pivot
        + saved_pose[3:]
        - physics.points[jaw]
    )
    np.testing.assert_allclose(
        saved_u[fixed], initial_boundary[fixed], rtol=0, atol=1e-12
    )
    model.dof_map.fixed_values = (
        torch.as_tensor(initial_boundary).flatten()[model.dof_map.fixed_indices].clone()
    )
    model.update(state, torch.as_tensor(saved_u))
    summary["initial"] = metrics(state.u)
    assert summary["initial"]["inverted_cells"] <= max_inverted
    assert summary["initial"]["force_norm"] <= cfg.force_atol
    np.testing.assert_allclose(
        summary["initial"]["minimum_J"], parent_summary["final"]["minimum_J"], rtol=1e-8
    )
    summary["boundary_audit"] = {
        "fixed_vertices": int(fixed.sum()),
        "jaw_fixed_vertices": int(jaw.sum()),
        "allfixed_cells": 0,
        "runtime_DofMap_verified": True,
    }
    summary["source_material_spec"] = physics.material_spec
    freeze(out)
    write(out / "summary.json", summary)

    started = time.perf_counter()
    step = cfg.initial_pose_step
    current = state.u.detach().clone()
    current_pose = saved_pose.copy()
    max_step_norm = physics.forward_tolerance["newton_max_step_norm_m"]
    summary["status"] = "running"
    while fraction < 1 and time.perf_counter() - started < cfg.wall_seconds:
        target_fraction = min(1.0, fraction + step)
        target_pose = target_fraction * full_pose
        old_r = Rotation.from_rotvec(current_pose[:3]).as_matrix()
        new_r = Rotation.from_rotvec(target_pose[:3]).as_matrix()
        x = physics.points + current.numpy(force=True)
        carried = (
            (x - pivot - current_pose[3:]) @ old_r @ new_r.T + pivot + target_pose[3:]
        )
        candidate = current + torch.as_tensor(weight[:, None] * (carried - x))
        boundary = np.zeros_like(physics.points)
        boundary[jaw] = (
            (physics.points[jaw] - pivot) @ new_r.T
            + pivot
            + target_pose[3:]
            - physics.points[jaw]
        )
        candidate[torch.as_tensor(fixed)] = torch.as_tensor(boundary[fixed])
        row = {
            "index": len(summary["attempts"]),
            "from_fraction": fraction,
            "target_fraction": target_fraction,
            "step": step,
            "status": "solving",
        }
        summary["attempts"].append(row)
        write(out / "summary.json", summary)
        row["carry_full_boundary_trial"] = geometry.audit(candidate.numpy(force=True))
        model.dof_map.fixed_values = (
            torch.as_tensor(boundary).flatten()[model.dof_map.fixed_indices].clone()
        )
        model.update(state, candidate)
        cached = CachedProblem(
            problem,
            cache_gradient=True,
            exact_curvature=True,
            wall_seconds=max(0.01, cfg.wall_seconds - (time.perf_counter() - started)),
        )
        try:
            _, solve = safeguarded_newton(
                cached,
                state,
                atol=cfg.force_atol,
                linear_rtol=cfg.linear_rtol,
                linear_max_steps=cfg.linear_max_steps,
                max_steps=cfg.max_newton_steps,
                max_step_norm=max_step_norm,
                max_backtracking_trials=16,
                initial_shift_scale=0,
                shift_policy="reuse",
                shift_scale_policy="mean_abs",
            )
            row["newton"] = solve
            row["metrics"] = metrics(state.u)
            assert row["metrics"]["force_norm"] <= cfg.force_atol
            if row["metrics"]["inverted_cells"] > max_inverted:
                row["status"] = "rejected_inversion_diagnostic_limit"
                summary["status"] = "blocked_by_inversion_diagnostic_limit"
            else:
                current = state.u.detach().clone()
                fraction, current_pose = target_fraction, target_pose
                row["status"] = "accepted"
                checkpoint = out / f"accepted-{len(summary['attempts']) - 1:03d}.npz"
                np.savez_compressed(
                    checkpoint,
                    displacement=current.numpy(force=True),
                    pose=current_pose,
                    fraction=fraction,
                )
                row["checkpoint"] = record(checkpoint)
                step = min(cfg.maximum_pose_step, step * 1.5)
        except ForwardConvergenceError as error:
            row["status"] = "rejected_solver"
            row["failure"] = str(error)
            row["solver_receipt"] = getattr(error, "receipt", None)
        row["solver_work"] = dict(cached.counts)
        if row["status"] != "accepted":
            step *= 0.5
            model.dof_map.fixed_values = current.flatten()[
                model.dof_map.fixed_indices
            ].clone()
            model.update(state, current)
        summary["completed_pose_fraction"] = fraction
        summary["elapsed_seconds"] = time.perf_counter() - started
        write(out / "summary.json", summary)
        LOG.info(
            "Pose attempt %d target %.6f: %s; completed %.6f",
            row["index"],
            target_fraction,
            row["status"],
            fraction,
        )
        if summary["status"] == "blocked_by_inversion_diagnostic_limit":
            break
        if step < cfg.minimum_pose_step:
            summary["status"] = "blocked_at_minimum_pose_step"
            break
    else:
        summary["status"] = "completed" if fraction == 1 else "wall_budget_exhausted"
    summary["numerically_converged_full_pose"] = fraction == 1
    summary["final"] = metrics(current)
    summary["orientation_valid"] = summary["final"]["inverted_cells"] == 0
    summary["physical_validity_claim"] = False
    final_path = out / "final.npz"
    np.savez_compressed(
        final_path,
        displacement=current.numpy(force=True),
        pose=current_pose,
        fraction=fraction,
    )
    summary["final_checkpoint"] = record(final_path)
    write(out / "summary.json", summary)
    freeze(out)
    cherries.log_metrics(
        {
            "completed_pose_fraction": fraction,
            "numerically_converged_full_pose": int(
                summary["numerically_converged_full_pose"]
            ),
            "inverted_cells": summary["final"]["inverted_cells"],
            "inverted_rest_volume_fraction": summary["final"][
                "inverted_rest_volume_fraction"
            ],
            "minimum_J": summary["final"]["minimum_J"],
            "force_norm": summary["final"]["force_norm"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
