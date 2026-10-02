"""Test a prescribed MouthOpen pose after removing every fully fixed cell."""

# ruff: noqa: E402, PLR0915
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial.transform import Rotation

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS = ROOT / "exp/2026/09/21/stress-activation-loss/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.extend(map(str, (STRESS, SOLVERS, JOINT)))
from accelerated_solvers import CachedProblem, safeguarded_newton
from experiment import Profile
from joint_equilibrium import ForwardConvergenceError
from mouthopen_geometry import MouthOpenGeometry
from stress_physics import FacePhysics, configure

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("35-forward-pruned")
    fixture: Path = GROUP / "data/30-pruned-fixture"
    force_atol: float = 1e-10
    linear_rtol: float = 1e-3
    linear_max_steps: int = 3000
    max_newton_steps: int = 100
    wall_seconds: float = 1200
    initial_pose_step: float = 0.025
    minimum_pose_step: float = 1e-5
    maximum_pose_step: float = 0.1
    volume_safety: float = 0.8


def write(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest()}


def freeze(out: Path) -> None:
    """Archive imported numerical sources, preserving their module identities."""
    records = {}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        local_source = path.is_relative_to(ROOT / "exp") or path.is_relative_to(
            ROOT / "src"
        )
        if path.suffix != ".py" or not local_source or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
        records[name] = {**record(target), "source": str(path)}
    write(out / "source-manifest.json", records)


def harmonic_weight(
    p: FacePhysics, fixed: np.ndarray, jaw: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Graph harmonic jaw carry; initialization only, never an equilibrium."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from cupyx.scipy.sparse.linalg import cg

    edges = np.concatenate(
        [p.tets[:, pair] for pair in itertools.combinations(range(4), 2)]
    )
    weights = np.tile(p.volumes_all, 6) / np.sum(
        (p.points[edges[:, 0]] - p.points[edges[:, 1]]) ** 2, axis=1
    )
    adjacency = sp.coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(p.points), len(p.points)),
    ).tocsr()
    lap = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    free_ids, fixed_ids = np.flatnonzero(~fixed), np.flatnonzero(fixed)
    matrix = csp.csr_matrix(lap[free_ids][:, free_ids])
    rhs = cp.asarray(-(lap[free_ids][:, fixed_ids] @ jaw[fixed_ids].astype(float)))
    value, info = cg(
        matrix, rhs, M=csp.diags(1 / matrix.diagonal()), rtol=1e-9, atol=0, maxiter=8000
    )
    relative = float(cp.linalg.norm(matrix @ value - rhs) / cp.linalg.norm(rhs))
    assert info == 0, (info, relative)
    assert relative <= 1.1e-9, relative
    w = jaw.astype(float)
    w[free_ids] = cp.asnumpy(value)
    assert w.min() >= -1e-8
    assert w.max() <= 1 + 1e-8
    return w, {
        "method": "rest-volume-over-edge-length-squared graph harmonic carry",
        "cg_info": int(info),
        "relative_residual": relative,
        "minimum": float(w.min()),
        "maximum": float(w.max()),
    }


class GuardedProblem(ForwardProblem):
    """Keep every linear trial path orientation-preserving and intersection-free."""

    def __init__(
        self, physics: FacePhysics, geometry: MouthOpenGeometry, safety: float
    ) -> None:
        super().__init__(model=physics.forward.model)
        self.points = torch.as_tensor(physics.points)
        self.tets = torch.as_tensor(physics.tets, dtype=torch.long)
        self.dm_inv = torch.as_tensor(physics.dm_inv)
        self.geometry = geometry
        self.safety = safety

    def deformation(self, u: torch.Tensor) -> torch.Tensor:
        x = (self.points + u)[self.tets]
        return (x[:, 1:] - x[:, :1]).transpose(1, 2) @ self.dm_inv

    def volume_fraction(self, u: torch.Tensor, du: torch.Tensor) -> float:
        f = self.deformation(u)
        minimum = float(torch.linalg.det(f).min())
        assert minimum > 0, minimum
        dx = du[self.tets]
        df = (dx[:, 1:] - dx[:, :1]).transpose(1, 2) @ self.dm_inv
        bound = float(
            torch.linalg.matrix_norm(torch.linalg.solve(f, df), ord="fro").max()
        )
        assert np.isfinite(bound)
        return 1.0 if bound == 0 else min(1.0, self.safety / bound)

    def max_step_size(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        full = self.model.dof_map.to_full_grad(direction)
        fraction = self.volume_fraction(state.u, full)
        collision_fraction = self.geometry.max_step_size(
            state.u.numpy(force=True), (fraction * full).numpy(force=True)
        )
        return torch.as_tensor(fraction * collision_fraction)

    def fun(self, state: Any) -> torch.Tensor:
        if float(torch.linalg.det(self.deformation(state.u)).min()) <= 0:
            return torch.tensor(float("inf"))
        return super().fun(state)


@torch.no_grad()
def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    pose_path = cherries.input(GROUP / "data/10-mandible/prepared.npz")
    fixture_path = cherries.input(cfg.fixture / "volume.vtu")
    with np.load(pose_path) as z:
        pose, pivot = z["pose"].copy(), z["pivot"].copy()
    summary = {
        "schema": "pruned-mouthopen-forward-v1",
        "status": "initializing",
        "valid_forward": False,
        "completed_pose_fraction": 0.0,
        "attempts": [],
        "inputs": {
            "volume": record(fixture_path),
            "pose": record(pose_path),
            **{
                name: record(cherries.input(cfg.fixture / name))
                for name in ("skin.vtp", "summary.json", "mapping.npz")
            },
        },
        "config": cfg.model_dump(mode="json"),
        "model": "historical no-skin bulk active strain at S=0; prescribed jaw; contact forces absent; complete FEM boundary CCD and endpoint intersection checks",
    }
    write(out / "summary.json", summary)
    configure()
    summary["runtime"] = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }
    p = FacePhysics(cfg.fixture, activation_model="strain", atol=cfg.force_atol)
    p.target = np.asarray(p.mesh.point_data["MouthOpen"]).copy()
    model, state = p.forward.model, p.forward.state
    fixed = np.asarray(p.mesh.point_data["IsFixed"], bool)
    group_id = list(p.mesh.field_data["GroupName"]).index("Mandible")
    jaw = fixed & (np.asarray(p.mesh.point_data["GroupId"]) == group_id)
    assert not np.any(fixed[p.tets].all(axis=1))
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.numpy(force=True),
        np.flatnonzero(np.repeat(fixed, 3)),
    )
    assert not np.any(fixed & np.asarray(p.mesh.point_data["IsLip"], bool))
    materials = model.get_materials()
    materials["muscle"]["activation_inv"] = torch.zeros((p.mesh.n_cells, 6))
    model.set_materials(materials)
    geometry = MouthOpenGeometry(p.mesh)
    problem = GuardedProblem(p, geometry, cfg.volume_safety)
    skin_ids = np.asarray(p.skin.point_data["GlobalPointId"], int)
    tri = p.points[skin_ids][np.asarray(p.skin.faces).reshape(-1, 4)[:, 1:]]
    areas = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        / 2
    )
    skin_mass = np.zeros(len(skin_ids))
    np.add.at(
        skin_mass,
        np.asarray(p.skin.faces).reshape(-1, 4)[:, 1:].ravel(),
        np.repeat(areas / 3, 3),
    )
    skin_mass /= skin_mass.sum()

    def metrics(u: torch.Tensor) -> dict:
        u_np = u.numpy(force=True)
        j = p.detf(u_np)
        error = u_np[skin_ids] - p.target[skin_ids]
        return {
            "minimum_J": float(j.min()),
            "inverted_cells": int(np.count_nonzero(j <= 0)),
            "fit_rms_mm": float(np.sqrt(np.sum(skin_mass[:, None] * error**2)) * 1000),
            "force_norm": float(torch.linalg.vector_norm(problem.grad(state))),
            **geometry.audit(u_np),
        }

    summary["boundary_audit"] = {
        "fixed_vertices": int(fixed.sum()),
        "jaw_fixed_vertices": int(jaw.sum()),
        "fixed_lip_vertices": 0,
        "fully_fixed_tets": 0,
        "runtime_DofMap_verified": True,
    }
    summary["neutral"] = metrics(state.u)
    summary["material_spec"] = dict(p.material_spec)
    del summary["material_spec"]["jaw_enabled"]
    summary["material_spec"]["jaw_motion"] = (
        "prescribed rigid kinematics on IsFixed intersect Mandible; pose is not a solved variable"
    )
    summary["material_spec"]["collision_policy"] = (
        "no contact force; experiment-local complete FEM boundary CCD and intersection checks"
    )
    freeze(out)
    write(out / "summary.json", summary)
    if not summary["neutral"]["no_intersections"]:
        summary["status"] = "blocked_by_neutral_surface_intersections"
        write(out / "summary.json", summary)
        return
    assert summary["neutral"]["minimum_J"] > 0
    assert summary["neutral"]["force_norm"] <= cfg.force_atol, summary["neutral"]
    w, harmonic = harmonic_weight(p, fixed, jaw)
    summary["harmonic_initializer"] = harmonic
    np.savez_compressed(out / "harmonic-weight.npz", weight=w)
    started = time.perf_counter()
    fraction, step = 0.0, cfg.initial_pose_step
    current = state.u.detach().clone()
    current_pose = np.zeros(6)
    max_step_norm = p.forward_tolerance["newton_max_step_norm_m"]
    summary["status"] = "running"

    while fraction < 1 and time.perf_counter() - started < cfg.wall_seconds:
        target_fraction = min(1.0, fraction + step)
        target_pose = target_fraction * pose
        old_r = Rotation.from_rotvec(current_pose[:3]).as_matrix()
        new_r = Rotation.from_rotvec(target_pose[:3]).as_matrix()
        x = p.points + current.numpy(force=True)
        carried = (
            (x - pivot - current_pose[3:]) @ old_r @ new_r.T + pivot + target_pose[3:]
        )
        candidate = current + torch.as_tensor(w[:, None] * (carried - x))
        boundary = np.zeros_like(p.points)
        boundary[jaw] = (
            (p.points[jaw] - pivot) @ new_r.T + pivot + target_pose[3:] - p.points[jaw]
        )
        candidate[torch.as_tensor(fixed)] = torch.as_tensor(boundary[fixed])
        row = {
            "index": len(summary["attempts"]),
            "from_fraction": fraction,
            "target_fraction": target_fraction,
            "step": step,
            "status": "checking_carry",
        }
        summary["attempts"].append(row)
        write(out / "summary.json", summary)
        safe = problem.volume_fraction(current, candidate - current)
        row["carry_volume_fraction"] = safe
        if safe < 1:
            row["status"] = "rejected_carry_volume_bound"
        else:
            ccd = geometry.max_step_size(
                current.numpy(force=True), (candidate - current).numpy(force=True)
            )
            row["carry_collision_fraction"] = ccd
            if ccd < 1:
                row["status"] = "rejected_carry_collision"
            else:
                model.dof_map.fixed_values = (
                    torch.as_tensor(boundary)
                    .flatten()[model.dof_map.fixed_indices]
                    .clone()
                )
                model.update(state, candidate)
                cached = CachedProblem(
                    problem,
                    cache_gradient=True,
                    exact_curvature=True,
                    wall_seconds=max(
                        0.01, cfg.wall_seconds - (time.perf_counter() - started)
                    ),
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
                    assert row["metrics"]["inverted_cells"] == 0
                    assert row["metrics"]["no_intersections"]
                    assert row["metrics"]["force_norm"] <= cfg.force_atol
                    current = state.u.detach().clone()
                    fraction, current_pose = target_fraction, target_pose
                    row["status"] = "accepted"
                    checkpoint = (
                        out / f"accepted-{len(summary['attempts']) - 1:03d}.npz"
                    )
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
                    row["metrics"] = metrics(state.u)
                    failed_path = out / f"rejected-{row['index']:03d}.npz"
                    np.savez_compressed(
                        failed_path,
                        displacement=state.u.numpy(force=True),
                        pose=target_pose,
                        fraction=target_fraction,
                    )
                    row["rejected_checkpoint"] = record(failed_path)
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
        if step < cfg.minimum_pose_step:
            summary["status"] = "blocked_at_minimum_pose_step"
            break
    else:
        summary["status"] = "completed" if fraction == 1 else "wall_budget_exhausted"
    summary["valid_forward"] = fraction == 1
    summary["final"] = metrics(current)
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
            "valid_forward": int(summary["valid_forward"]),
            **{
                k: summary["final"][k]
                for k in ("minimum_J", "inverted_cells", "fit_rms_mm", "force_norm")
            },
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
