"""Compare two jaw initializers from one audited corrected-IsFixed checkpoint."""

# ruff: noqa: C901, E402, EM101, PLR0912, PLR0915, SLF001, TRY003, TRY301
from __future__ import annotations

import copy
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import ipctk
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from chin_rigid_pose import estimate_rigid_chin_pose
from joint_common import ProfileJoint, sha256, write_json
from joint_coupled_predictor import audit_coupled_motion, rotation_sagitta
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from mouthopen_coupled_seed import _mandible_arc_radius, prepare_coupled_seed
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-005"
    output_dir: Path = GROUP / "data/collision-off-seed-comparison-001"
    max_rotation_deg: float = 1.0
    max_translation_m: float = 0.001
    fractions: tuple[float, ...] = (1.0, 0.25)
    off_wall_seconds: float = 600.0
    corrector_wall_seconds: float = 600.0
    ipc_threads: int = 4
    saved_estimate_run: Path | None = None


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict) -> Path:
    path = Path(item["path"])
    assert record(path) == item
    return path


def main(cfg: Config) -> None:
    from mouthopen_collision_off_seed import prepare_collision_off_seed

    output = cfg.output_dir.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    source = cfg.source_run.resolve()
    source_protocol = json.loads((source / "protocol.json").read_text())
    source_summary = json.loads((source / "summary.json").read_text())
    audit = json.loads((source / "independent-audit.json").read_text())
    for name in ("summary", "protocol", "endpoint"):
        assert bound(audit["inputs"][name]) == source / (
            name + (".npz" if name == "endpoint" else ".json")
        )
    assert source_summary["status"] != "running"
    assert audit["force"]["converged"]
    assert audit["collision"]["feasible"]
    assert audit["valid_forward"]
    checkpoint = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["iteration"] == source_summary["final"]["iteration"]
    with np.load(source / "endpoint.npz") as saved:
        for key in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        active_source_ids = saved["active_cell_ids"].copy()

    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    reference = bound(source_protocol["sources"]["reference_repair"])
    target_path = bound(source_protocol["sources"]["blendshapes"])
    neutral_path = bound(source_protocol["sources"]["neutral_endpoint"])
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected = np.r_[
        np.repeat(isfixed, 3),
        np.ones(model.dof_map.n_full - 3 * len(isfixed), dtype=bool),
    ]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected)
    )
    assert not isfixed[np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)].any()
    install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    baseline = model.get_materials()
    with np.load(neutral_path.parent / "active-strain-fields.npz") as saved:
        for key, saved_key in (
            ("activation_inv", "skin_activation_inverse"),
            ("mu", "skin_mu_mpa"),
            ("thickness", "skin_thickness_m"),
        ):
            np.testing.assert_array_equal(
                baseline["skin"][key].cpu().numpy(), saved[saved_key]
            )
    np.testing.assert_array_equal(
        active_source_ids, physics.base.retained_active_cell_ids
    )
    q = checkpoint["activation_inv"].to(device="cuda")
    old_pose = checkpoint["pose_rad_m"].to(device="cuda")
    old_u = checkpoint["displacement_m"].to(device="cuda")
    active_ids = physics.base.active_t

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return result

    kappa = float(source_protocol["ipc_stiffness_mpa"])
    collision = model.collision
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=1e-8,
        adjoint_rtol=1e-7,
        adjoint_relative_shift=0.001,
        newton_max_steps=3000,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    policy = source_protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] == 100
    assert policy["maximum_inverted_rest_volume_fraction"] == 0.0001

    def geometry_allowed(geometry: dict) -> bool:
        return (
            geometry["inverted_tetrahedra"] <= 100
            and geometry["inverted_rest_volume_fraction"] <= 0.0001
        )

    with np.load(target_path) as data:
        index = list(data["expression_names"]).index("MouthOpen")
        ids = data["skin_global_ids"].copy()
        triangles = data["skin_triangles"].copy()
        neutral_points = data["new_neutral_points_m"].copy()
        target_points = data["target_points_m"][index].copy()
    patch_path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    patch = np.asarray(json.loads(patch_path.read_text())["patch_local_ids"])
    chin = estimate_rigid_chin_pose(
        neutral_points,
        target_points,
        triangles,
        patch,
        np.asarray(physics.pivot_t.cpu()),
    )
    write_json(output / "chin-estimate.json", chin)
    goal = np.asarray(chin["pose_rad_m"])
    old_pose_np = old_pose.cpu().numpy()
    old_rotation = Rotation.from_rotvec(old_pose_np[:3])
    delta_rot = (Rotation.from_rotvec(goal[:3]) * old_rotation.inv()).as_rotvec()
    delta_translation = goal[3:] - old_pose_np[3:]
    fraction = min(
        1.0,
        math.radians(cfg.max_rotation_deg) / np.linalg.norm(delta_rot),
        cfg.max_translation_m / np.linalg.norm(delta_translation),
    )
    area_points = neutral_points[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(
            area_points[:, 1] - area_points[:, 0], area_points[:, 2] - area_points[:, 0]
        ),
        axis=1,
    )
    weights = np.zeros(len(ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()

    def reset() -> None:
        runtime.deadline = None
        model.set_materials(materials(q))
        model.dof_map.fixed_values = physics.boundary(old_pose).detach().clone()
        runtime.forward.state.u = old_u.detach().clone()
        runtime.forward.state.collision = collision.state_at(old_u)
        runtime.last_problem = None
        runtime.last_sparse_problem = None

    def evaluate(u: torch.Tensor, pose: torch.Tensor) -> dict:
        model.set_materials(materials(q))
        model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
        state = model.State(u=u.detach().clone())
        state.collision = collision.state_at(u)
        problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.9)
        force = float(torch.linalg.vector_norm(problem.grad(state)))
        contact = runtime._contact_gate(state)
        geometry = geometry_metrics(physics, u)
        residual = physics.points[ids] + u.cpu().numpy()[ids] - target_points
        return {
            "force_norm_n": force * 1e6,
            "force_converged": force <= 1e-8,
            "contact": contact,
            "geometry": geometry,
            "geometry_allowed": geometry_allowed(geometry),
            "fixed_max_error_m": float(
                (u.flatten()[model.dof_map.fixed_indices] - physics.boundary(pose))
                .abs()
                .max()
            ),
            "fit_rms_mm": 1000
            * float(np.sqrt(np.sum(weights * np.sum(residual**2, axis=1)))),
            "valid_endpoint": force <= 1e-8
            and geometry_allowed(geometry)
            and contact["receipt"]["contact_numerically_valid"]
            and contact["no_intersections"]
            and contact["minimum_active_gap_at_least_buffer"],
        }

    reset()
    initial = evaluate(old_u, old_pose)
    assert initial["valid_endpoint"]
    protocol = {
        "schema": "mouthopen-collision-off-initializer-comparison-v1",
        "source_checkpoint": record(source / "checkpoint.pt"),
        "source_audit": record(source / "independent-audit.json"),
        "source_protocol": record(source / "protocol.json"),
        "target": record(target_path),
        "chin_patch": record(patch_path),
        "initial": initial,
        "old_pose_rad_m": old_pose_np.tolist(),
        "chin_goal_pose_rad_m": goal.tolist(),
        "bounded_fraction_toward_chin": float(fraction),
        "fractions": cfg.fractions,
        "active_strain_fixed": True,
        "exact_skin_prestrain_preserved": True,
        "isfixed_only": True,
        "tetrahedron_exclusion": exclusion,
        "inversion_policy": policy,
        "ipc_stiffness_mpa": kappa,
        "force_atol": 1e-8,
        "acceptance": "Seed and endpoint geometry caps plus final collision/force gates. Old-to-new CCD checked independently; an endpoint-only result never replaces the live inverse fit.",
        "timing_scope": "Single ordered comparison on shared GPU, includes seed and correction; JIT and unrelated process contention can affect timing.",
        "saved_estimate_run": None
        if cfg.saved_estimate_run is None
        else str(cfg.saved_estimate_run.resolve()),
    }
    write_json(output / "protocol.json", protocol)
    sources = output / "sources"
    sources.mkdir()
    for name, root in (
        ("new-neutral", GROUP / "src"),
        ("joint", JOINT),
        ("solvers", SOLVERS),
    ):
        dest = sources / name
        dest.mkdir()
        for path in root.glob("*.py"):
            shutil.copy2(path, dest / path.name)
    results = []
    write_json(
        output / "summary.json",
        {"status": "running", "initial": initial, "results": results},
    )
    for alpha in cfg.fractions:
        target_pose_np = np.r_[
            (
                Rotation.from_rotvec(alpha * fraction * delta_rot) * old_rotation
            ).as_rotvec(),
            old_pose_np[3:] + alpha * fraction * delta_translation,
        ]
        target_pose = torch.as_tensor(
            target_pose_np, device=old_pose.device, dtype=old_pose.dtype
        )
        angle = float(np.linalg.norm(alpha * fraction * delta_rot))
        for name, initializer in (
            ("coupled_tangent", prepare_coupled_seed),
            ("collision_off_pushout", prepare_collision_off_seed),
        ):
            reset()
            trial = output / f"alpha-{alpha:g}" / name
            trial.mkdir(parents=True)
            row = {
                "method": name,
                "alpha": alpha,
                "rotation_increment_deg": math.degrees(angle),
                "translation_increment_mm": float(
                    np.linalg.norm(alpha * fraction * delta_translation)
                )
                * 1000,
                "target_pose_rad_m": target_pose_np.tolist(),
                "admitted_continuation": False,
                "stage": "initializer",
            }
            results.append(row)
            callback = initializer
            if cfg.saved_estimate_run is not None and name == "collision_off_pushout":
                from mouthopen_saved_estimate_seed import prepare_saved_estimate_seed

                callback = prepare_saved_estimate_seed
                prior = json.loads(
                    (cfg.saved_estimate_run / "summary.json").read_text()
                )
                prior_row = next(
                    r
                    for r in prior["results"]
                    if r["alpha"] == alpha and r["method"] == name
                )
                row["prior_estimate_seconds"] = prior_row["total_seconds"]
                row["saved_estimate_approximation_explicit"] = True
            write_json(
                output / "summary.json",
                {"status": "running", "initial": initial, "results": results},
            )
            started = time.perf_counter()
            LOG.info(
                "Trial %s alpha %.3g, rotation %.6g deg, translation %.6g mm",
                name,
                alpha,
                row["rotation_increment_deg"],
                row["translation_increment_mm"],
            )
            try:
                kwargs = {"deadline": time.perf_counter() + cfg.off_wall_seconds}
                if name == "collision_off_pushout":
                    if cfg.saved_estimate_run is None:
                        kwargs["off_wall_seconds"] = cfg.off_wall_seconds
                    else:
                        kwargs["estimate_path"] = (
                            cfg.saved_estimate_run
                            / f"alpha-{alpha:g}"
                            / name
                            / "seed/collision-off.pt"
                        )
                candidate, receipt = callback(
                    physics,
                    materials,
                    q,
                    q,
                    old_pose,
                    target_pose,
                    old_u,
                    trial / "seed",
                    **kwargs,
                )
                row["seed_seconds"] = time.perf_counter() - started
                row["seed_receipt"] = receipt
                row["seed_geometry"] = geometry_metrics(physics, candidate)
                np.savez_compressed(
                    trial / "seed.npz",
                    displacement_m=candidate.cpu().numpy(),
                    pose_rad_m=target_pose_np,
                )
                radius, _ = _mandible_arc_radius(
                    physics,
                    collision,
                    torch.as_tensor(
                        physics.pivot_t, device=old_u.device, dtype=old_u.dtype
                    ),
                    physics.boundary(old_pose),
                    physics.boundary(target_pose),
                )
                row["old_to_seed_motion"] = audit_coupled_motion(
                    collision,
                    old_u,
                    candidate,
                    rotation_margin_m=rotation_sagitta(radius, angle),
                )
                if not geometry_allowed(row["seed_geometry"]):
                    raise ForwardConvergenceError(
                        "repaired seed exceeds unchanged retained-tet allowance"
                    )
                row["stage"] = "contact_correction"
                write_json(
                    output / "summary.json",
                    {"status": "running", "initial": initial, "results": results},
                )
                runtime.deadline = time.perf_counter() + cfg.corrector_wall_seconds
                final = runtime.primal(
                    materials(q), physics.boundary(target_pose), candidate
                )
                row["forward"] = copy.deepcopy(runtime.last_forward)
                row["endpoint"] = evaluate(final, target_pose)
                row["admitted_continuation"] = (
                    row["endpoint"]["valid_endpoint"]
                    and row["old_to_seed_motion"]["admitted"]
                )
                row["stage"] = "finished"
                np.savez_compressed(
                    trial / "endpoint.npz",
                    displacement_m=final.cpu().numpy(),
                    activation_inv=q.cpu().numpy(),
                    active_cell_ids=active_source_ids,
                    pose_rad_m=target_pose_np,
                )
            except ForwardConvergenceError as error:
                row["failure"] = {"message": str(error), "receipt": error.receipt}
                row["failed_stage"] = row["stage"]
                row["stage"] = "rejected"
            finally:
                row["total_seconds"] = time.perf_counter() - started
                write_json(trial / "result.json", row)
                write_json(
                    output / "summary.json",
                    {"status": "running", "initial": initial, "results": results},
                )
                reset()
            LOG.info(
                "Trial ended: %s",
                {
                    key: row[key]
                    for key in (
                        "method",
                        "alpha",
                        "stage",
                        "total_seconds",
                        "admitted_continuation",
                    )
                },
            )
    write_json(
        output / "summary.json",
        {
            "status": "finished_comparison",
            "initial": initial,
            "results": results,
            "inverse_converged": False,
        },
    )
    cherries.log_asset(output / "protocol.json")
    cherries.log_asset(output / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
