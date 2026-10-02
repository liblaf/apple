# ruff: noqa: C901, E402, PLR0912, PLR0915, EM101, TRY003
"""Keep fitted tensors fixed; re-equilibrate with deformable contact."""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(PARENT / "src"))
spec = importlib.util.spec_from_file_location(
    "contact_transition_parent", PARENT / "src/91-solve-activation-transition.py"
)
assert spec is not None
assert spec.loader is not None
BASELINE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASELINE
spec.loader.exec_module(BASELINE)
BASE = BASELINE.BASE

from accelerated_solvers import CachedProblem, safeguarded_newton
from experiment import Profile
from joint_equilibrium import ForwardConvergenceError
from mouthopen_geometry import MouthOpenGeometry
from stress_physics import FacePhysics, configure, strain_to_activation_inv
from transition_contact import audit_contact, build_self_contact

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("23-contact-existing-activation")
    source: Path = PARENT / "data/91-smile-mouthopen-transition-003"
    fixture: Path = PARENT / "data/30-pruned-fixture"
    initialization: Literal["saved-mouthopen", "neutral-continuation"] = (
        "neutral-continuation"
    )
    initial_seed: Path = (
        PARENT / "data/70-mouthopen-four-stage/rankone_learned/last.npz"
    )
    hessian_backend: Literal["gpu_contact", "gpu_sparse"] = "gpu_contact"
    frames: int = 121
    stiffness_mpa: float = 0.0012
    minimum_distance_m: float = 1e-8
    force_atol: float = 1e-10
    linear_rtol: float = 1e-3
    linear_max_steps: int = 3000
    max_newton_steps: int = 200
    initial_step: float = 0.025
    minimum_step: float = 1e-5
    maximum_step: float = 0.05
    max_subdivisions: int = 12
    inversion_fraction_limit: float = 0.001
    wall_seconds: float = 1200


class ProjectedContactSearch(HessianProblem):
    """Use PSD contact search curvature with either exact bulk backend."""

    def prepare_contact(self, state: Any) -> None:
        contact = self.model.collision
        assert contact is not None
        assert state.collision is not None
        if state.collision.hess is None:
            positions = (contact.vertices + state.u[contact.indices]).numpy(force=True)
            state.collision.hess = contact.potential.hessian(
                collisions=state.collision.collisions,
                mesh=contact.collision_mesh,
                X=positions,
                project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
            )

    def _prepare(self, state: Any) -> None:
        self.prepare_contact(state)
        super()._prepare(state)

    def hess_prod(self, state: Any, direction: Any) -> Any:
        self.prepare_contact(state)
        return super().hess_prod(state, direction)

    def report(self) -> dict:
        return {
            **super().report(),
            "contact_search_hessian": "IPC stencilwise PSD CLAMP, search only",
            "bulk_hessian": "exact, unprojected",
            "energy_gradient_and_force_gate_changed": False,
        }


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.frames >= 3
    assert 0 < cfg.minimum_step < cfg.initial_step <= cfg.maximum_step
    assert 0 < cfg.inversion_fraction_limit <= 0.001
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    (out / "checkpoints").mkdir()
    (out / "frames").mkdir()
    inputs = {
        name: BASE.record(path)
        for name, path in {
            "source_summary": cfg.source / "summary.json",
            "endpoints": cfg.source / "endpoints.npz",
            "mesh": cfg.source / "mesh.npz",
            "volume": cfg.fixture / "volume.vtu",
            "skin": cfg.fixture / "skin.vtp",
            "weight": PARENT / "data/35-forward-pruned-002/harmonic-weight.npz",
            "initial_seed": cfg.initial_seed,
        }.items()
    }
    parent = json.loads((cfg.source / "summary.json").read_text())
    assert parent["status"] == "completed"
    for name in ("mesh.npz", "endpoints.npz"):
        assert (
            BASE.record(cfg.source / name)["sha256"]
            == parent["outputs"][name]["sha256"]
        )
        shutil.copy2(cfg.source / name, out / name)
    with np.load(cfg.source / "endpoints.npz") as z:
        smile = z["S_smile"].copy()
        mouth = z["S_mouthopen"].copy()
        full_pose = z["pose_mouthopen"].copy()
        pivot = z["pivot"].copy()
    with np.load(Path(inputs["weight"]["path"])) as z:
        weight = z["weight"].copy()
    summary = {
        "schema": "collision-activation-transition-v1",
        "status": "initializing",
        "config": cfg.model_dump(mode="json"),
        "inputs": inputs,
        "collision_scope": "all FEM boundary faces retained; free-free and free-prescribed primitive contacts; prescribed-only pairs exempt and full-boundary intersections reported separately; no separate rigid obstacles",
        "activation_policy": "reuse the exact solved endpoint S arrays; no activation optimization, no inverse refit; only displacement is solved",
        "initialization": cfg.initialization,
        "transition": "S(beta)=(1-beta)S_MouthOpen+beta S_Smile; jaw=(1-beta)*MouthOpen pose",
        "physical_validity_claim": False,
        "accepted": [],
        "failures": [],
        "frames": [],
    }
    BASE.write(out / "summary.json", summary)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="strain", atol=cfg.force_atol)
    with np.load(cfg.source / "mesh.npz") as z:
        np.testing.assert_array_equal(physics.points, z["rest_points"])
        np.testing.assert_array_equal(physics.tets, z["tets"])
        np.testing.assert_array_equal(physics.ids, z["active_ids"])
    assert smile.shape == mouth.shape == (len(physics.ids), 3, 3)
    model = physics.forward.model
    contact, surface_receipt = build_self_contact(
        physics.mesh,
        stiffness_mpa=cfg.stiffness_mpa,
        minimum_distance_m=cfg.minimum_distance_m,
    )
    model.collision = contact
    summary["contact_surface"] = surface_receipt
    summary["material_spec"] = {
        **physics.material_spec,
        "contact_enabled": True,
        "collision_policy": summary["collision_scope"],
        "jaw_enabled": True,
    }
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], bool)
    contact.collision_set_type = ipctk.NormalCollisions.CollisionSetType.IPC
    patches = np.where(fixed, 0, np.arange(len(fixed)) + 1).astype(np.int32)
    local_patches = patches[contact.indices.numpy(force=True)]
    contact.collision_mesh.can_collide = ipctk.make_vertex_patches_filter(local_patches)
    surface_receipt.update(
        {
            "collision_set_type": "IPC",
            "area_weighting": True,
            "prescribed_only_pairs_exempt": True,
            "surface_exclusions": "no faces removed; fixed-fixed primitive pairs exempt",
        }
    )
    full_geometry = MouthOpenGeometry(physics.mesh)
    names = list(physics.mesh.field_data["GroupName"])
    jaw = fixed & (
        np.asarray(physics.mesh.point_data["GroupId"]) == names.index("Mandible")
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.numpy(force=True),
        np.flatnonzero(np.repeat(fixed, 3)),
    )
    np.testing.assert_allclose(weight[jaw], 1, atol=1e-8, rtol=0)
    np.testing.assert_allclose(weight[fixed & ~jaw], 0, atol=1e-8, rtol=0)
    started = time.perf_counter()
    deadline = started + cfg.wall_seconds
    BASE.freeze(out)
    current = np.zeros_like(physics.points)
    last_state = None
    phase = "neutral"
    fraction = 0.0
    BASE.write(out / "summary.json", summary)

    def contact_audit(u: torch.Tensor) -> dict:
        receipt = audit_contact(contact, u)
        receipt["scoped_boundary_no_intersections"] = receipt.pop(
            "complete_boundary_no_intersections"
        )
        receipt["scoped_boundary_has_intersections"] = receipt.pop(
            "complete_boundary_has_intersections"
        )
        receipt["intersection_scope"] = summary["collision_scope"]
        receipt["unfiltered_boundary"] = full_geometry.audit(u)
        return receipt

    def boundary(alpha: float) -> np.ndarray:
        result = np.zeros_like(physics.points)
        pose = alpha * full_pose
        result[jaw] = (
            (physics.points[jaw] - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
            + pivot
            + pose[3:]
            - physics.points[jaw]
        )
        return result

    def carry(seed: np.ndarray, a: float, b: float) -> np.ndarray:
        old, new = a * full_pose, b * full_pose
        old_r = Rotation.from_rotvec(old[:3]).as_matrix()
        new_r = Rotation.from_rotvec(new[:3]).as_matrix()
        x = physics.points + seed
        carried = (x - pivot - old[3:]) @ old_r @ new_r.T + pivot + new[3:]
        candidate = seed + weight[:, None] * (carried - x)
        candidate[fixed] = boundary(b)[fixed]
        state = contact.state_at(torch.as_tensor(seed))
        fraction = float(
            contact.max_step_size(
                state, torch.as_tensor(seed), torch.as_tensor(candidate - seed)
            )
        )
        if fraction < 1:
            raise ForwardConvergenceError(
                "prescribed jaw carry restricted by CCD",
                receipt={"ccd_fraction": fraction},
            )
        return candidate

    def solve(
        activation: np.ndarray, alpha: float, seed: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        nonlocal last_state
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("declared continuation wall budget exhausted")
        bc = boundary(alpha)
        np.testing.assert_allclose(seed[fixed], bc[fixed], rtol=0, atol=1e-12)
        before = contact_audit(torch.as_tensor(seed))
        if not before["contact_valid"]:
            raise ForwardConvergenceError(
                "contact-infeasible initialization", receipt=before
            )
        model.dof_map.fixed_values = (
            torch.as_tensor(bc).flatten()[model.dof_map.fixed_indices].clone()
        )
        physics.materials["muscle"]["activation_inv"] = torch.zeros(
            (physics.mesh.n_cells, 6)
        ).index_copy(
            0, physics.id_t, strain_to_activation_inv(torch.as_tensor(activation))
        )
        model.set_materials(physics.materials)
        state = model.State(
            u=torch.as_tensor(seed.copy()),
            collision=contact.state_at(torch.as_tensor(seed)),
        )
        last_state = state
        summary["last_attempt"] = {"phase": phase, "jaw_fraction": alpha}
        BASE.write(out / "summary.json", summary)
        problem = ForwardProblem(model=model)
        hessian = ProjectedContactSearch(problem, cfg.hessian_backend)
        cached = CachedProblem(
            hessian, cache_gradient=True, exact_curvature=True, wall_seconds=remaining
        )
        initial_force = float(torch.linalg.vector_norm(problem.grad(state)))
        state, receipt = safeguarded_newton(
            cached,
            state,
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
        contact_receipt = contact_audit(state.u)
        inner_contact = contact.diagnostics(state.collision, state.u)
        contact_receipt.update(inner_contact)
        contact_receipt["contact_valid"] = (
            inner_contact["contact_numerically_valid"]
            and contact_receipt["scoped_boundary_no_intersections"]
        )
        j = physics.detf(state.u.numpy(force=True))
        inverted = int(np.count_nonzero(j <= 0))
        result = {
            "accepted_force_norm": force,
            "initial_force_norm": initial_force,
            "solver_valid": bool(np.isfinite(force) and force <= cfg.force_atol),
            "contact": contact_receipt,
            "inverted_cells": inverted,
            "minimum_J": float(j.min()),
            "orientation_valid": bool(np.all(j > 0)),
            "newton": receipt,
            "hessian": hessian.report(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        torch.testing.assert_close(
            state.u.flatten()[model.dof_map.fixed_indices],
            model.dof_map.fixed_values,
            rtol=0,
            atol=1e-12,
        )
        if not result["solver_valid"] or not contact_receipt["contact_valid"]:
            raise ForwardConvergenceError(
                "accepted force/contact gate failed", receipt=result
            )
        if not np.isfinite(j).all() or inverted > cfg.inversion_fraction_limit * len(j):
            raise ForwardConvergenceError(
                "inverted-cell limit exceeded", receipt=result
            )
        return state.u.numpy(force=True).copy(), result

    def save(u: np.ndarray, value: float, receipt: dict, stage: str) -> dict:
        alpha = value if stage == "initialization" else 1 - value
        if stage == "mouthopen":
            alpha = 1.0
        elif stage == "neutral":
            alpha = 0.0
        path = out / "checkpoints" / f"state-{len(summary['accepted']):04d}.npz"
        np.savez_compressed(
            path, u=u, fraction=value, alpha=alpha, pose=alpha * full_pose
        )
        row = {
            "phase": stage,
            "fraction": value,
            "checkpoint": BASE.record(path),
            "diagnostics": receipt,
        }
        summary["accepted"].append(row)
        summary["latest"] = row
        BASE.write(out / "summary.json", summary)
        LOG.info(
            "Accepted %s %.8f force %.3g contacts %s inverted %d",
            stage,
            value,
            receipt["accepted_force_norm"],
            receipt["contact"].get("active_contact_count"),
            receipt["inverted_cells"],
        )
        return row

    def advance(
        seed: np.ndarray, a: float, b: float, stage: str, depth: int = 0
    ) -> tuple[np.ndarray, dict]:
        try:
            if stage == "initialization":
                trial = carry(seed, a, b)
                u, receipt = solve(b * mouth, b, trial)
            else:
                trial = carry(seed, 1 - a, 1 - b)
                u, receipt = solve((1 - b) * mouth + b * smile, 1 - b, trial)
        except ForwardConvergenceError as error:
            summary["failures"].append(
                {
                    "phase": stage,
                    "from": a,
                    "to": b,
                    "depth": depth,
                    "message": str(error),
                    "receipt": getattr(error, "receipt", None),
                }
            )
            BASE.write(out / "summary.json", summary)
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    "declared continuation wall budget exhausted"
                ) from error
            if depth >= cfg.max_subdivisions or b - a < 2 * cfg.minimum_step:
                raise
            mid = (a + b) / 2
            LOG.warning("Subdividing %s %.8f -> %.8f: %s", stage, a, b, error)
            middle, _ = advance(seed, a, mid, stage, depth + 1)
            return advance(middle, mid, b, stage, depth + 1)
        save(u, b, receipt, stage)
        return u, receipt

    try:
        if cfg.initialization == "saved-mouthopen":
            phase = "mouthopen"
            summary["status"] = "re_equilibrating_fixed_mouthopen_activation"
            with np.load(cfg.initial_seed) as z:
                current = z["u"].copy()
                if "S" in z:
                    np.testing.assert_array_equal(z["S"], mouth)
            current, receipt = solve(mouth, 1.0, current)
            save(current, 0, receipt, "mouthopen")
        else:
            summary["status"] = "relaxing_neutral"
            current, receipt = solve(np.zeros_like(mouth), 0, current)
            save(current, 0, receipt, "neutral")
            phase = "initialization"
            summary["status"] = "initializing_mouthopen"
            step = cfg.initial_step
            while fraction < 1:
                target = min(1.0, fraction + step)
                current, receipt = advance(current, fraction, target, phase)
                fraction = target
                step = min(cfg.maximum_step, step * 1.5)
        phase = "transition"
        fraction = 0.0
        summary["status"] = "running_transition"
        for index in range(cfg.frames):
            beta = float((1 - np.cos(np.pi * index / (cfg.frames - 1))) / 2)
            if index:
                current, receipt = advance(current, fraction, beta, phase)
            path = out / "frames" / f"frame-{index:03d}.npz"
            np.savez_compressed(
                path, u=current, beta=beta, alpha=1 - beta, pose=(1 - beta) * full_pose
            )
            summary["frames"].append(
                {
                    "index": index,
                    "beta": beta,
                    "alpha": 1 - beta,
                    "checkpoint": BASE.record(path),
                    "diagnostics": receipt,
                }
            )
            BASE.write(out / "summary.json", summary)
            cherries.set_step(index)
            cherries.log_metrics(
                {
                    "smile_blend": beta,
                    "force_norm": receipt["accepted_force_norm"],
                    "inverted_cells": receipt["inverted_cells"],
                }
            )
            fraction = beta
        summary["status"] = "completed"
    except (ForwardConvergenceError, TimeoutError) as error:
        summary["status"] = "blocked"
        summary["failure"] = {
            "phase": phase,
            "completed_fraction": summary.get("latest", {}).get("fraction", 0.0),
            "type": type(error).__name__,
            "message": str(error),
            "receipt": getattr(error, "receipt", None),
        }
        if last_state is not None:
            np.savez_compressed(
                out / "failed-solver-state.npz", u=last_state.u.numpy(force=True)
            )
            terminal_force = float(
                torch.linalg.vector_norm(ForwardProblem(model=model).grad(last_state))
            )
            terminal_j = physics.detf(last_state.u.numpy(force=True))
            summary["failure_terminal_diagnostics"] = {
                "force_norm": terminal_force,
                "contact": contact_audit(last_state.u),
                "inverted_cells": int(np.count_nonzero(terminal_j <= 0)),
                "minimum_J": float(terminal_j.min()),
                "accepted_equilibrium": False,
            }
        LOG.warning("Contact continuation stopped: %s", summary["failure"])
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        if last_state is not None:
            np.savez_compressed(
                out / "interrupted-solver-state.npz", u=last_state.u.numpy(force=True)
            )
        raise
    except Exception as error:
        summary["status"] = "failed"
        summary["failure"] = {
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        BASE.write(out / "summary.json", summary)
    for item in inputs.values():
        assert BASE.record(Path(item["path"]))["sha256"] == item["sha256"]
    for item in json.loads((out / "source-manifest.json").read_text()).values():
        assert BASE.record(Path(item["source"]))["sha256"] == item["sha256"]
    summary["provenance_verified"] = True
    BASE.write(out / "summary.json", summary)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
