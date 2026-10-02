# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, PLR0912, PLR0915, TRY003
"""Audit a completed collision-off endpoint before testing nearby descent.

This is a separate Cherries run. It never changes the inverse checkpoint. A
stricter baseline that moves outside the declared inversion policy ends the
diagnostic before any adjoint or neighboring-parameter probe is attempted.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from collision_off_runtime import install_collision_off_runtime  # noqa: E402
from collision_off_seed import prepare_coupled_seed  # noqa: E402
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import ForwardConvergenceError, configure_cuda  # noqa: E402
from mesh_step_scale import mean_rest_edge_length  # noqa: E402
from mouthopen_constrained_direction import project_coupled_pose  # noqa: E402
from mouthopen_pose_projection import PoseProjectionError  # noqa: E402
from mouthopen_tet_policy import (  # noqa: E402
    exclude_fully_fixed_tetrahedra,
    geometry_metrics,
)
from neutral_active_strain import install_active_strain  # noqa: E402
from reference_rebase import build_rebased_physics  # noqa: E402

POSE_SCALE = torch.tensor([math.pi / 18.0] * 3 + [0.01] * 3, dtype=torch.float64)


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/mouthopen-001"
    output_dir: Path = GROUP / "data/mouthopen-probe-001"
    audit_name: str = "independent-audit.json"
    strict_forward_atol: float = 1e-9
    refined_forward_atol: float = 1e-10
    adjoint_rtol: float = 1e-7
    probe_alpha: float = 1e-6
    loss_resolution_abs: float = 1e-12
    max_newton_steps: int = 3000
    wall_seconds: float = 3600
    unshifted_wall_seconds: float = 300


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict[str, str]) -> Path:
    """Resolve a protocol input by hash after an optional bundle relocation."""
    original = Path(item["path"])
    candidates = [original]
    if "data" in original.parts:
        suffix = Path(*original.parts[original.parts.index("data") + 1 :])
        candidates.append(GROUP / "data" / suffix)
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == item["sha256"]:
            return candidate
    raise AssertionError(item)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def determinant_ratios(physics: Any, u: torch.Tensor) -> np.ndarray:
    base = physics.base
    tets = np.asarray(base.tets, dtype=np.int64)[
        base._mouthopen_retained_tetrahedron_ids  # noqa: SLF001
    ]
    reference = np.asarray(base.points, dtype=np.float64)
    deformed = reference + u[: len(reference)].detach().cpu().numpy()
    rest_edges = np.transpose(
        reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1)
    )
    deformed_edges = np.transpose(
        deformed[tets[:, 1:]] - deformed[tets[:, :1]], (0, 2, 1)
    )
    rest_det = np.linalg.det(rest_edges)
    assert np.all(rest_det > 0)
    result = np.linalg.det(deformed_edges) / rest_det
    assert np.isfinite(result).all()
    return result


def geometry_allowed(metrics: dict, policy: dict) -> bool:
    return (
        metrics["inverted_tetrahedra"] <= policy["maximum_inverted_tetrahedra"]
        and metrics["inverted_rest_volume_fraction"]
        <= policy["maximum_inverted_rest_volume_fraction"]
    )


def gradient_difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = float(torch.linalg.vector_norm(left - right))
    scale = max(
        float(torch.linalg.vector_norm(left)),
        float(torch.linalg.vector_norm(right)),
        1e-30,
    )
    return {
        "absolute_l2": difference,
        "relative_l2": difference / scale,
        "left_l2": float(torch.linalg.vector_norm(left)),
        "right_l2": float(torch.linalg.vector_norm(right)),
    }


def main(cfg: Config) -> None:
    run = cfg.run_dir.resolve()
    output = cfg.output_dir.resolve()
    assert run.is_dir()
    assert not output.exists()
    assert 0 < cfg.strict_forward_atol <= 1e-8
    assert 0 < cfg.refined_forward_atol < cfg.strict_forward_atol
    assert 0 < cfg.adjoint_rtol <= 1e-7
    assert cfg.probe_alpha > 0
    assert cfg.loss_resolution_abs >= 0
    assert cfg.wall_seconds > 0
    assert cfg.unshifted_wall_seconds > 0
    protocol_path = run / "protocol.json"
    summary_path = run / "summary.json"
    endpoint_path = run / "endpoint.npz"
    checkpoint_path = run / "checkpoint.pt"
    audit_path = run / cfg.audit_name
    protocol = json.loads(protocol_path.read_text())
    source_summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["collision_enabled"] is False
    assert source_summary["status"] != "running"
    assert audit["schema"] == "collision-off-expression-independent-audit-v1"
    assert audit["valid_forward"] is True
    assert audit["inputs"]["endpoint"]["sha256"] == sha256(endpoint_path)
    assert source_summary["endpoint"]["sha256"] == sha256(endpoint_path)
    policy = protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] <= 100
    assert policy["maximum_inverted_rest_volume_fraction"] <= 1e-4
    assert cfg.strict_forward_atol <= protocol["force_contract"]["atol"]
    target_path = bound(protocol["sources"]["blendshapes"])
    neutral_endpoint = bound(protocol["sources"]["neutral_endpoint"])
    reference = bound(protocol["sources"]["reference_repair"])
    neutral_audit_path = neutral_endpoint.parent / "independent-audit.json"
    neutral_audit = json.loads(neutral_audit_path.read_text())
    fields_path = neutral_endpoint.parent / "active-strain-fields.npz"
    assert (
        sha256(fields_path) == neutral_audit["run_inputs"][fields_path.name]["sha256"]
    )
    with np.load(endpoint_path, allow_pickle=False) as archive:
        endpoint_u = np.asarray(archive["displacement_m"], dtype=np.float64)
        endpoint_q = np.asarray(archive["activation_inv"], dtype=np.float64)
        endpoint_pose = np.asarray(archive["pose_rad_m"], dtype=np.float64)
        active_source_ids = np.asarray(archive["active_cell_ids"], dtype=np.int64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    np.testing.assert_array_equal(endpoint_u, checkpoint["displacement_m"].numpy())
    np.testing.assert_array_equal(endpoint_q, checkpoint["activation_inv"].numpy())
    np.testing.assert_array_equal(endpoint_pose, checkpoint["pose_rad_m"].numpy())
    pose_z_cpu = checkpoint["pose_normalized"]
    np.testing.assert_allclose(
        endpoint_pose,
        (pose_z_cpu * POSE_SCALE).numpy(),
        rtol=0,
        atol=1e-15,
    )
    assert np.isfinite(endpoint_u).all()
    assert np.isfinite(endpoint_q).all()
    assert np.isfinite(endpoint_pose).all()
    output.mkdir(parents=True)
    source_records = {}
    for directory, label in (
        (GROUP / "src", "collision-off-expressions"),
        (JOINT, "joint"),
        (SOLVERS, "solver-performance"),
        (ROOT / "src/liblaf/apple", "apple"),
    ):
        shutil.copytree(
            directory,
            output / "sources" / label,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for source in (output / "sources").rglob("*.py"):
        source_records[str(source.relative_to(output / "sources"))] = sha256(source)
    result: dict[str, Any] = {
        "schema": "collision-off-reduced-objective-probe-v1",
        "status": "running",
        "inverse_converged": False,
        "collision_enabled": False,
        "ccd_enabled": False,
        "inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "endpoint": record(endpoint_path),
            "checkpoint": record(checkpoint_path),
            "independent_audit": record(audit_path),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "reference_repair": record(reference),
            "neutral_active_strain_fields": record(fields_path),
        },
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "wall_seconds": cfg.wall_seconds,
        "unshifted_wall_seconds": cfg.unshifted_wall_seconds,
        "source_snapshot_sha256": source_records,
        "inversion_policy": policy,
        "expression_name": protocol["expression_name"],
        "shift_comparison": {},
        "probes": [],
    }
    write_json(output / "summary.json", result)

    configure_cuda()
    physics, _ = build_rebased_physics(reference.parent, inverse=True)
    model = physics.runtime.forward.model
    model.collision = None
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.concatenate(
        (
            np.repeat(is_fixed, 3),
            np.ones(model.dof_map.n_full - 3 * len(is_fixed), dtype=bool),
        )
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
    )
    lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    assert int(lip.sum()) == 3408
    assert not np.any(is_fixed[lip])
    _, strain_receipt, _ = install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    assert exclusion["excluded_tetrahedra"] == 2249
    baseline_materials = model.get_materials()
    with np.load(fields_path, allow_pickle=False) as fields:
        np.testing.assert_array_equal(
            baseline_materials["skin"]["activation_inv"].cpu().numpy(),
            fields["skin_activation_inverse"],
        )
        np.testing.assert_array_equal(
            baseline_materials["skin"]["mu"].cpu().numpy(), fields["skin_mu_mpa"]
        )
        np.testing.assert_array_equal(
            baseline_materials["skin"]["thickness"].cpu().numpy(),
            fields["skin_thickness_m"],
        )
    active_ids = physics.base.active_t
    np.testing.assert_array_equal(
        active_source_ids, physics.base.retained_active_cell_ids
    )
    assert endpoint_q.shape == (len(active_ids), 6)
    result["model_policy"] = {
        "isfixed_dof_map_exact": True,
        "all_3408_lip_vertices_free": True,
        "excluded_all_fixed_tetrahedra": exclusion["excluded_tetrahedra"],
        "retained_active_original_ids_match": True,
        "skin_fields_match_corrected_neutral": True,
        "active_strain_formulation": strain_receipt["formulation"],
    }

    device = baseline_materials["muscle"]["activation_inv"].device
    dtype = baseline_materials["muscle"]["activation_inv"].dtype
    q = torch.as_tensor(endpoint_q, device=device, dtype=dtype)
    pose_z = pose_z_cpu.to(device=device, dtype=dtype)
    saved_u = torch.as_tensor(endpoint_u, device=device, dtype=dtype)
    scale = POSE_SCALE.to(device=device, dtype=dtype)

    def material_at(value: torch.Tensor) -> dict:
        fields = {name: dict(rows) for name, rows in baseline_materials.items()}
        fields["muscle"]["activation_inv"] = baseline_materials["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return fields

    def boundary_at(value: torch.Tensor) -> torch.Tensor:
        return physics.boundary(value * scale)

    def physical_pose(value: torch.Tensor) -> torch.Tensor:
        return value * scale

    with np.load(target_path, allow_pickle=False) as archive:
        names = [str(name) for name in archive["expression_names"]]
        target_index = names.index(protocol["expression_name"])
        skin_ids_np = np.asarray(archive["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(archive["skin_triangles"], dtype=np.int64)
        neutral_skin = np.asarray(archive["new_neutral_points_m"], dtype=np.float64)
        target_skin = np.asarray(
            archive["target_points_m"][target_index], dtype=np.float64
        )
    with np.load(neutral_endpoint, allow_pickle=False) as archive:
        neutral_u = np.asarray(archive["displacement_m"], dtype=np.float64)
    np.testing.assert_array_equal(
        np.asarray(physics.points)[skin_ids_np] + neutral_u[skin_ids_np], neutral_skin
    )
    xyz = neutral_skin[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids_np), dtype=np.float64)
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    skin_ids = torch.as_tensor(skin_ids_np, device=device)
    weights_t = torch.as_tensor(weights, device=device, dtype=dtype)
    target = torch.as_tensor(
        target_skin - np.asarray(physics.points)[skin_ids_np],
        device=device,
        dtype=dtype,
    )
    delta = torch.as_tensor(target_skin - neutral_skin, device=device, dtype=dtype)
    scale2 = (weights_t[:, None] * delta.square()).sum()
    assert float(scale2) > 0

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (weights_t[:, None] * (u[skin_ids] - target).square()).sum() / scale2

    @torch.no_grad()
    def direct_free_force(u: torch.Tensor) -> float:
        state = model.State(u=u.detach().clone())
        assert state.collision is None
        torch.testing.assert_close(
            state.u.flatten()[model.dof_map.fixed_indices],
            model.dof_map.fixed_values,
            rtol=0,
            atol=1e-14,
        )
        free = model.dof_map.to_free_grad(model.grad(state))
        assert bool(torch.isfinite(free).all())
        return float(torch.linalg.vector_norm(free))

    runtime = install_collision_off_runtime(
        physics,
        forward_atol=cfg.strict_forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        adjoint_relative_shift=0.001,
        newton_max_steps=cfg.max_newton_steps,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
    )
    assert model.collision is None
    assert runtime.forward.state.collision is None
    runtime.deadline = time.perf_counter() + cfg.wall_seconds
    model.set_materials(material_at(q))
    model.dof_map.fixed_values = boundary_at(pose_z).detach().clone()
    saved_force = direct_free_force(saved_u)
    saved_loss = float(objective(saved_u))
    saved_geometry = geometry_metrics(physics, saved_u)
    saved_det = determinant_ratios(physics, saved_u)
    result["saved_baseline"] = {
        "force": saved_force,
        "loss": saved_loss,
        "geometry": saved_geometry,
        "within_declared_policy": geometry_allowed(saved_geometry, policy),
    }
    assert saved_force <= protocol["force_contract"]["atol"]
    assert geometry_allowed(saved_geometry, policy)
    write_json(output / "summary.json", result)

    try:
        strict_u = runtime.primal(material_at(q), boundary_at(pose_z), saved_u)
    except ForwardConvergenceError as error:
        result["status"] = "unresolved_baseline"
        result["strict_baseline"] = {
            "failure": str(error),
            "receipt": error.receipt,
            "forward": copy.deepcopy(runtime.last_forward),
        }
        if runtime.last_problem is not None and hasattr(
            runtime, "last_failed_displacement"
        ):
            failed = runtime.last_failed_displacement
            result["strict_baseline"]["geometry"] = geometry_metrics(physics, failed)
            result["strict_baseline"]["loss"] = float(objective(failed))
            failed_det = determinant_ratios(physics, failed)
            retained = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
            result["strict_baseline"]["newly_inverted_original_cell_ids"] = retained[
                (saved_det > 0) & (failed_det <= 0)
            ].tolist()
            save_npz(
                output / "strict-baseline-failed.npz",
                displacement_m=failed.cpu().numpy(),
            )
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    strict_force = direct_free_force(strict_u)
    strict_geometry = geometry_metrics(physics, strict_u)
    strict_loss = float(objective(strict_u))
    strict_det = determinant_ratios(physics, strict_u)
    retained = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
    save_npz(
        output / "strict-baseline.npz",
        displacement_m=strict_u.cpu().numpy(),
        activation_inv=q.cpu().numpy(),
        pose_normalized=pose_z.cpu().numpy(),
        pose_rad_m=physical_pose(pose_z).cpu().numpy(),
        active_cell_ids=active_source_ids,
    )
    result["strict_baseline"] = {
        "force": strict_force,
        "solver_reported_force": float(runtime.last_forward["grad_norm"]),
        "loss": strict_loss,
        "geometry": strict_geometry,
        "within_declared_policy": geometry_allowed(strict_geometry, policy),
        "maximum_change_from_saved_endpoint_m": float((strict_u - saved_u).abs().max()),
        "rms_change_from_saved_endpoint_m": float(
            (strict_u - saved_u).square().mean().sqrt()
        ),
        "loss_change_from_saved_endpoint": strict_loss - saved_loss,
        "newly_inverted_original_cell_ids": retained[
            (saved_det > 0) & (strict_det <= 0)
        ].tolist(),
        "recovered_original_cell_ids": retained[
            (saved_det <= 0) & (strict_det > 0)
        ].tolist(),
        "forward": copy.deepcopy(runtime.last_forward),
        "archive": record(output / "strict-baseline.npz"),
    }
    write_json(output / "summary.json", result)
    if strict_force > cfg.strict_forward_atol or not geometry_allowed(
        strict_geometry, policy
    ):
        result["status"] = "unresolved_baseline"
        result["reason"] = (
            "stricter re-equilibrium is outside the original physical gates"
        )
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        cherries.log_output(output / "strict-baseline.npz")
        return

    # The second force level measures objective sensitivity to a genuine
    # numerical refinement. Repeating the same tolerance would be tautological.
    runtime.tolerances["atol"] = cfg.refined_forward_atol
    try:
        refined_u = runtime.primal(material_at(q), boundary_at(pose_z), strict_u)
    except ForwardConvergenceError as error:
        result["status"] = "unresolved_baseline"
        result["reason"] = "1e-10 baseline refinement did not converge"
        result["refined_baseline"] = {
            "failure": str(error),
            "receipt": error.receipt,
            "forward": copy.deepcopy(runtime.last_forward),
        }
        if runtime.last_problem is not None and hasattr(
            runtime, "last_failed_displacement"
        ):
            failed = runtime.last_failed_displacement
            failed_det = determinant_ratios(physics, failed)
            result["refined_baseline"]["geometry"] = geometry_metrics(physics, failed)
            result["refined_baseline"]["loss"] = float(objective(failed))
            result["refined_baseline"]["newly_inverted_original_cell_ids"] = retained[
                (strict_det > 0) & (failed_det <= 0)
            ].tolist()
            save_npz(
                output / "refined-baseline-failed.npz",
                displacement_m=failed.cpu().numpy(),
            )
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    refined_loss = float(objective(refined_u))
    refined_geometry = geometry_metrics(physics, refined_u)
    refined_force = direct_free_force(refined_u)
    refined_det = determinant_ratios(physics, refined_u)
    save_npz(
        output / "refined-baseline.npz",
        displacement_m=refined_u.cpu().numpy(),
        activation_inv=q.cpu().numpy(),
        pose_normalized=pose_z.cpu().numpy(),
        pose_rad_m=physical_pose(pose_z).cpu().numpy(),
        active_cell_ids=active_source_ids,
    )
    result["refined_baseline"] = {
        "force": refined_force,
        "solver_reported_force": float(runtime.last_forward["grad_norm"]),
        "loss": refined_loss,
        "geometry": refined_geometry,
        "within_declared_policy": geometry_allowed(refined_geometry, policy),
        "loss_change_from_1e-9": refined_loss - strict_loss,
        "maximum_displacement_change_from_1e-9_m": float(
            (refined_u - strict_u).abs().max()
        ),
        "newly_inverted_original_cell_ids": retained[
            (strict_det > 0) & (refined_det <= 0)
        ].tolist(),
        "forward": copy.deepcopy(runtime.last_forward),
        "archive": record(output / "refined-baseline.npz"),
    }
    write_json(output / "summary.json", result)
    if refined_force > cfg.refined_forward_atol or not geometry_allowed(
        refined_geometry, policy
    ):
        result["status"] = "unresolved_baseline"
        result["reason"] = "1e-10 baseline refinement is outside physical gates"
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    baseline_refinement_drift = abs(refined_loss - strict_loss)
    result["loss_resolution_screen"] = {
        "absolute_floor": cfg.loss_resolution_abs,
        "baseline_refinement_loss_change_abs": baseline_refinement_drift,
        "screen_rule": "max(absolute_floor, 10*(baseline_refinement_drift + trial_refinement_drift))",
        "formal_error_bound": False,
    }
    write_json(output / "summary.json", result)

    gradients: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    def gradient_at_shift(shift: float) -> dict[str, Any]:
        runtime.adjoint_relative_shift = shift
        runtime.warm_adjoints.clear()
        q_value = q.detach().clone().requires_grad_(requires_grad=True)
        pose_value = pose_z.detach().clone().requires_grad_(requires_grad=True)
        u_value = runtime.solve(
            material_at(q_value), boundary_at(pose_value), refined_u, key="probe"
        )
        geometry = geometry_metrics(physics, u_value)
        force = direct_free_force(u_value)
        if force > cfg.refined_forward_atol or not geometry_allowed(geometry, policy):
            raise ForwardConvergenceError(
                "shift comparison moved outside physical gates",
                receipt={"force": force, "geometry": geometry},
            )
        state_change = float((u_value - refined_u).abs().max())
        if state_change > 1e-12:
            raise ForwardConvergenceError(
                "shift comparison changed the strict baseline state",
                receipt={"maximum_displacement_change_m": state_change},
            )
        loss = objective(u_value)
        q_gradient, pose_gradient = torch.autograd.grad(loss, (q_value, pose_value))
        sparse = copy.deepcopy(runtime.last_sparse_adjoint)
        adjoint = copy.deepcopy(runtime.last_adjoint)
        assert sparse["shifted_relative_residual"] <= cfg.adjoint_rtol
        assert sparse["native_shifted_relative_residual"] <= cfg.adjoint_rtol
        if shift == 0:
            assert sparse["original_unshifted_relative_residual"] <= cfg.adjoint_rtol
        gradients[shift] = (q_gradient.detach(), pose_gradient.detach())
        name = f"gradient-shift-{shift:.0e}.npz"
        save_npz(
            output / name,
            q_gradient=q_gradient.detach().cpu().numpy(),
            pose_gradient=pose_gradient.detach().cpu().numpy(),
        )
        return {
            "shift": shift,
            "loss": float(loss),
            "force": force,
            "solver_reported_force": float(runtime.last_forward["grad_norm"]),
            "geometry": geometry,
            "maximum_state_change_from_refined_baseline_m": state_change,
            "sparse_adjoint": sparse,
            "adjoint": adjoint,
            "gradient_archive": record(output / name),
        }

    for shift in (0.001, 0.0001):
        try:
            evidence = gradient_at_shift(shift)
        except (ForwardConvergenceError, AssertionError, RuntimeError) as error:
            result["status"] = "adjoint_unresolved"
            result["shift_comparison"][str(shift)] = {
                "failure": str(error),
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
            }
            write_json(output / "summary.json", result)
            cherries.log_output(output / "summary.json")
            return
        result["shift_comparison"][str(shift)] = evidence
        write_json(output / "summary.json", result)
    result["shift_comparison"]["gradient_difference"] = {
        "q": gradient_difference(gradients[0.001][0], gradients[0.0001][0]),
        "pose": gradient_difference(gradients[0.001][1], gradients[0.0001][1]),
    }
    # An unshifted CG attempt is optional. Its own strict true residual decides
    # whether it can be called an exact adjoint; failure does not justify a claim.
    global_deadline = runtime.deadline
    runtime.deadline = min(
        global_deadline, time.perf_counter() + cfg.unshifted_wall_seconds
    )
    try:
        try:
            result["shift_comparison"]["0.0"] = gradient_at_shift(0.0)
            best_shift = 0.0
        except (ForwardConvergenceError, AssertionError, RuntimeError) as error:
            result["shift_comparison"]["0.0"] = {
                "status": "unresolved",
                "failure": str(error),
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
            }
            best_shift = 0.0001
    finally:
        runtime.deadline = global_deadline
    result["probe_gradient_shift"] = best_shift
    write_json(output / "summary.json", result)

    gq, gp = gradients[best_shift]
    q_max = float(gq.abs().max())
    pose_max = float(gp.abs().max())
    dq = torch.zeros_like(gq) if q_max == 0 else -0.02 * gq / q_max
    dp = torch.zeros_like(gp) if pose_max == 0 else -0.1 * gp / pose_max
    directions = {
        "joint": (dq, dp),
        "q_only": (dq, torch.zeros_like(dp)),
        "pose_only": (torch.zeros_like(dq), dp),
    }
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
    witness = False
    runtime.adjoint_relative_shift = 0.0001
    for name, (q_direction, pose_direction) in directions.items():
        if time.perf_counter() >= runtime.deadline:
            result["status"] = "time_budget_exhausted_unresolved"
            break
        slope = float((gq * q_direction).sum() + (gp * pose_direction).sum())
        if slope >= 0:
            result["probes"].append(
                {"direction": name, "status": "zero_or_non_descent"}
            )
            continue
        for alpha in (cfg.probe_alpha, cfg.probe_alpha / 4):
            if time.perf_counter() >= runtime.deadline:
                result["status"] = "time_budget_exhausted_unresolved"
                break
            trial_q = q + alpha * q_direction
            row: dict[str, Any] = {
                "direction": name,
                "alpha": alpha,
                "q_step_max_abs": float((trial_q - q).abs().max()),
            }
            runtime.tolerances["atol"] = cfg.refined_forward_atol
            if name == "joint":
                projection_dir = output / "projections" / f"{name}-{alpha:.3e}"
                try:
                    projected_dp, projection = project_coupled_pose(
                        physics,
                        material_at,
                        physical_pose,
                        q,
                        trial_q,
                        pose_z,
                        alpha * pose_direction,
                        refined_u,
                        projection_dir,
                        activation_threshold=float(
                            protocol["config"]["projection_activation_threshold"]
                        ),
                        margin=float(
                            protocol["config"]["projection_determinant_margin"]
                        ),
                        epsilon=float(protocol["config"]["projection_probe_epsilon"]),
                        deadline=runtime.deadline,
                    )
                    row["projection"] = {
                        "receipt": record(projection_dir / "summary.json"),
                        "diagnostics": projection["projection"],
                    }
                except (
                    PoseProjectionError,
                    ForwardConvergenceError,
                    AssertionError,
                ) as error:
                    row["status"] = "projection_unresolved"
                    row["failure"] = str(error)
                    result["probes"].append(row)
                    write_json(output / "summary.json", result)
                    continue
                trial_pose = pose_z + projected_dp
            else:
                trial_pose = pose_z + alpha * pose_direction
            row["normalized_pose_step_max_abs"] = float(
                (trial_pose - pose_z).abs().max()
            )
            row["predicted_loss_change"] = float(
                (gq * (trial_q - q)).sum() + (gp * (trial_pose - pose_z)).sum()
            )
            if row["predicted_loss_change"] >= 0:
                row["status"] = "projected_non_descent_unresolved"
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            seed_dir = output / "seeds" / f"{name}-{alpha:.3e}"
            forward_started = False
            try:
                trial_seed, seed_receipt = prepare_coupled_seed(
                    physics,
                    material_at,
                    q,
                    trial_q,
                    pose_z * scale,
                    trial_pose * scale,
                    refined_u,
                    seed_dir,
                    predictor_relative_shift=runtime.adjoint_relative_shift,
                    predictor_rtol=cfg.adjoint_rtol,
                    deadline=runtime.deadline,
                )
                row["seed"] = {
                    "receipt": record(seed_dir / "summary.json"),
                    "geometry": geometry_metrics(physics, trial_seed),
                    "predictor": seed_receipt["predictor"],
                }
                runtime.tolerances["atol"] = cfg.strict_forward_atol
                forward_started = True
                trial_u_1e9 = runtime.primal(
                    material_at(trial_q), boundary_at(trial_pose), trial_seed
                )
            except ForwardConvergenceError as error:
                row["status"] = "forward_unresolved"
                row["failure"] = str(error)
                row["receipt"] = error.receipt
                if (
                    forward_started
                    and runtime.last_problem is not None
                    and hasattr(runtime, "last_failed_displacement")
                ):
                    row["failed_geometry"] = geometry_metrics(
                        physics, runtime.last_failed_displacement
                    )
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            trial_force_1e9 = direct_free_force(trial_u_1e9)
            trial_geometry_1e9 = geometry_metrics(physics, trial_u_1e9)
            trial_loss_1e9 = float(objective(trial_u_1e9))
            row["atol_1e9"] = {
                "force": trial_force_1e9,
                "solver_reported_force": float(runtime.last_forward["grad_norm"]),
                "geometry": trial_geometry_1e9,
                "loss": trial_loss_1e9,
                "loss_change_from_baseline": trial_loss_1e9 - strict_loss,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            save_npz(
                output / f"probe-{name}-{alpha:.3e}-1e-9.npz",
                displacement_m=trial_u_1e9.cpu().numpy(),
                activation_inv=trial_q.cpu().numpy(),
                pose_normalized=trial_pose.cpu().numpy(),
                pose_rad_m=physical_pose(trial_pose).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            row["atol_1e9"]["archive"] = record(
                output / f"probe-{name}-{alpha:.3e}-1e-9.npz"
            )
            runtime.tolerances["atol"] = cfg.refined_forward_atol
            try:
                trial_u_1e10 = runtime.primal(
                    material_at(trial_q), boundary_at(trial_pose), trial_u_1e9
                )
            except ForwardConvergenceError as error:
                row["status"] = "refinement_unresolved"
                row["refinement_failure"] = {
                    "message": str(error),
                    "receipt": error.receipt,
                }
                if runtime.last_problem is not None and hasattr(
                    runtime, "last_failed_displacement"
                ):
                    row["refinement_failed_geometry"] = geometry_metrics(
                        physics, runtime.last_failed_displacement
                    )
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            trial_force_1e10 = direct_free_force(trial_u_1e10)
            trial_geometry_1e10 = geometry_metrics(physics, trial_u_1e10)
            trial_loss_1e10 = float(objective(trial_u_1e10))
            trial_det = determinant_ratios(physics, trial_u_1e10)
            seed_det = determinant_ratios(physics, trial_seed)
            new_inverted = np.flatnonzero((refined_det > 0) & (trial_det <= 0))
            row["newly_inverted_original_cell_ids"] = retained_ids[
                new_inverted
            ].tolist()
            row["newly_inverted_seed_detF"] = seed_det[new_inverted].tolist()
            row["newly_inverted_corrected_detF"] = trial_det[new_inverted].tolist()
            row["seed_vs_corrected_detF_max_abs"] = float(
                np.max(np.abs(seed_det - trial_det))
            )
            row["atol_1e10"] = {
                "force": trial_force_1e10,
                "solver_reported_force": float(runtime.last_forward["grad_norm"]),
                "geometry": trial_geometry_1e10,
                "loss": trial_loss_1e10,
                "loss_change_from_baseline": trial_loss_1e10 - refined_loss,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            trial_refinement_drift = abs(trial_loss_1e10 - trial_loss_1e9)
            resolution = max(
                cfg.loss_resolution_abs,
                10 * (baseline_refinement_drift + trial_refinement_drift),
            )
            change_1e9 = trial_loss_1e9 - strict_loss
            change_1e10 = trial_loss_1e10 - refined_loss
            row["loss_resolution_screen"] = {
                "baseline_refinement_drift": baseline_refinement_drift,
                "trial_refinement_drift": trial_refinement_drift,
                "used": resolution,
                "formal_error_bound": False,
            }
            change_disagreement = abs(change_1e9 - change_1e10) / max(
                abs(change_1e9), abs(change_1e10), 1e-30
            )
            row["loss_change_relative_disagreement"] = change_disagreement
            row["within_declared_policy_at_both_force_levels"] = geometry_allowed(
                trial_geometry_1e9, policy
            ) and geometry_allowed(trial_geometry_1e10, policy)
            row["resolved_loss_decrease"] = (
                row["within_declared_policy_at_both_force_levels"]
                and trial_force_1e9 <= cfg.strict_forward_atol
                and trial_force_1e10 <= cfg.refined_forward_atol
                and change_1e9 < -resolution
                and change_1e10 < -resolution
                and change_disagreement <= 0.1
            )
            row["status"] = (
                "feasible_descent_witness"
                if row["resolved_loss_decrease"]
                else "no_resolved_witness"
            )
            save_npz(
                output / f"probe-{name}-{alpha:.3e}-1e-10.npz",
                displacement_m=trial_u_1e10.cpu().numpy(),
                activation_inv=trial_q.cpu().numpy(),
                pose_normalized=trial_pose.cpu().numpy(),
                pose_rad_m=physical_pose(trial_pose).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            row["atol_1e10"]["archive"] = record(
                output / f"probe-{name}-{alpha:.3e}-1e-10.npz"
            )
            witness |= row["resolved_loss_decrease"]
            result["probes"].append(row)
            write_json(output / "summary.json", result)

    if result["status"] != "time_budget_exhausted_unresolved":
        result["status"] = (
            "feasible_descent_witness" if witness else "no_witness_unresolved"
        )
    result["interpretation"] = (
        "A geometry-admissible loss decrease survives two force tolerances and the stated numerical screen; this is an empirical nonstationarity witness, not a formal error bound."
        if witness
        else "These finite probes did not establish stationarity or infeasibility."
    )
    write_json(output / "summary.json", result)
    cherries.log_output(output / "summary.json")
    cherries.log_output(output / "strict-baseline.npz")
    cherries.log_output(output / "refined-baseline.npz")
    cherries.log_metrics(
        {
            "probe/strict_baseline_force_n": strict_force * 1e6,
            "probe/witness": float(witness),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
