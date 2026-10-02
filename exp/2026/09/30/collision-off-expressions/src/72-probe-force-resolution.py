# Copyright (c) 2026 liblaf
# ruff: noqa: C901, PLR0915
"""Measure collision-off baseline drift at two stricter force tolerances.

This Cherries run reuses a hash-bound 71 baseline with exact audited parameters.
It changes no inverse parameter and makes no gradient or stationarity claim.
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
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import ForwardConvergenceError, configure_cuda  # noqa: E402
from mesh_step_scale import mean_rest_edge_length  # noqa: E402
from mouthopen_tet_policy import (  # noqa: E402
    exclude_fully_fixed_tetrahedra,
    geometry_metrics,
)
from neutral_active_strain import install_active_strain  # noqa: E402
from reference_rebase import build_rebased_physics  # noqa: E402

POSE_SCALE = torch.tensor([math.pi / 18.0] * 3 + [0.01] * 3, dtype=torch.float64)
EXPECTED_WARM_SHA256 = (
    "1ef03bff6fac7347652549eb67b6154fc0f93c8dee507e23cddc2e0c36e540f0"
)


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/mouthopen-001"
    prior_probe_dir: Path = GROUP / "data/mouthopen-resolution-001"
    output_dir: Path = GROUP / "data/mouthopen-force-resolution-001"
    expected_warm_sha256: str = EXPECTED_WARM_SHA256
    audit_name: str = "independent-audit.json"
    strict_forward_atol: float = 1e-13
    refined_forward_atol: float = 1e-14
    adjoint_rtol: float = 1e-7
    max_newton_steps: int = 3000
    wall_seconds: float = 3600


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


def main(cfg: Config) -> None:
    run = cfg.run_dir.resolve()
    prior_probe = cfg.prior_probe_dir.resolve()
    output = cfg.output_dir.resolve()
    assert run.is_dir()
    assert prior_probe.is_dir()
    assert not output.exists()
    assert 0 < cfg.strict_forward_atol <= 1e-8
    assert 0 < cfg.refined_forward_atol < cfg.strict_forward_atol
    assert cfg.strict_forward_atol == 1e-13
    assert cfg.refined_forward_atol == 1e-14
    assert cfg.expected_warm_sha256 == EXPECTED_WARM_SHA256
    assert 0 < cfg.adjoint_rtol <= 1e-7
    assert cfg.wall_seconds > 0
    strict_label = f"{cfg.strict_forward_atol:.0e}"
    refined_label = f"{cfg.refined_forward_atol:.0e}"
    assert float(strict_label) == cfg.strict_forward_atol
    assert float(refined_label) == cfg.refined_forward_atol
    strict_key = f"atol_{strict_label}"
    refined_key = f"atol_{refined_label}"
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
    prior_summary_path = prior_probe / "summary.json"
    prior_summary = json.loads(prior_summary_path.read_text())
    assert (
        prior_summary["schema"] == "collision-off-reduced-objective-resolution-probe-v2"
    )
    assert prior_summary["status"] == "no_witness_unresolved"
    assert prior_summary["collision_enabled"] is False
    assert prior_summary["ccd_enabled"] is False
    for key, current in (
        ("protocol", protocol_path),
        ("summary", summary_path),
        ("endpoint", endpoint_path),
        ("checkpoint", checkpoint_path),
        ("independent_audit", audit_path),
    ):
        assert prior_summary["inputs"][key]["sha256"] == sha256(current), key
    prior_source = (
        prior_probe / "sources/collision-off-expressions/71-probe-resolution.py"
    )
    assert prior_summary["source_snapshot_sha256"][
        "collision-off-expressions/71-probe-resolution.py"
    ] == sha256(prior_source)
    assert sha256(prior_source) == sha256(GROUP / "src/71-probe-resolution.py")
    warm_archive_path = bound(prior_summary["refined_baseline"]["archive"])
    assert warm_archive_path == prior_probe / "refined-baseline.npz"
    assert sha256(warm_archive_path) == cfg.expected_warm_sha256
    with np.load(warm_archive_path, allow_pickle=False) as archive:
        warm_u_np = np.asarray(archive["displacement_m"], dtype=np.float64)
        np.testing.assert_array_equal(archive["activation_inv"], endpoint_q)
        np.testing.assert_array_equal(archive["pose_normalized"], pose_z_cpu.numpy())
        np.testing.assert_array_equal(
            archive["pose_rad_m"], checkpoint["pose_rad_m"].numpy()
        )
        np.testing.assert_array_equal(archive["active_cell_ids"], active_source_ids)
    assert warm_u_np.shape == endpoint_u.shape
    assert np.isfinite(warm_u_np).all()
    assert prior_summary["refined_baseline"]["force"] <= 1e-12
    assert prior_summary["refined_baseline"]["within_declared_policy"] is True
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
        "schema": "collision-off-force-resolution-probe-v1",
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
            "prior_probe_summary": record(prior_summary_path),
            "prior_probe_refined_baseline": record(warm_archive_path),
            "prior_probe_71_source": record(prior_source),
        },
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "wall_seconds": cfg.wall_seconds,
        "source_snapshot_sha256": source_records,
        "inversion_policy": policy,
        "expression_name": protocol["expression_name"],
        "warmstart_method": "hash-bound prior 71-probe refined equilibrium at exact audited q and normalized pose",
        "expected_warm_sha256": cfg.expected_warm_sha256,
        "force_level_keys": {"strict": strict_key, "refined": refined_key},
        "parameter_probe_enabled": False,
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
    warm_u = torch.as_tensor(warm_u_np, device=device, dtype=dtype)
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

    def failed_state_evidence(
        u: torch.Tensor, reference_det: np.ndarray
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "displacement_finite": bool(torch.isfinite(u).all())
        }
        if not evidence["displacement_finite"]:
            return evidence
        try:
            evidence["direct_free_force"] = direct_free_force(u)
        except (AssertionError, RuntimeError, ValueError) as error:
            evidence["direct_force_failure"] = str(error)
        loss = float(objective(u))
        evidence["loss"] = loss if math.isfinite(loss) else None
        try:
            evidence["geometry"] = geometry_metrics(physics, u)
            det = determinant_ratios(physics, u)
            retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
            evidence["newly_inverted_original_cell_ids"] = retained_ids[
                (reference_det > 0) & (det <= 0)
            ].tolist()
        except (
            AssertionError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            evidence["geometry_evaluation_failure"] = str(error)
        return evidence

    runtime = install_collision_off_runtime(
        physics,
        forward_atol=cfg.strict_forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        adjoint_relative_shift=0.0,
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
    result["saved_baseline"] = {
        "force": saved_force,
        "loss": saved_loss,
        "geometry": saved_geometry,
        "within_declared_policy": geometry_allowed(saved_geometry, policy),
    }
    assert saved_force <= protocol["force_contract"]["atol"]
    assert geometry_allowed(saved_geometry, policy)
    warm_force = direct_free_force(warm_u)
    warm_loss = float(objective(warm_u))
    warm_geometry = geometry_metrics(physics, warm_u)
    warm_det = determinant_ratios(physics, warm_u)
    assert warm_force <= 1e-12
    np.testing.assert_allclose(
        warm_loss, prior_summary["refined_baseline"]["loss"], rtol=0, atol=1e-12
    )
    for key, value in warm_geometry.items():
        np.testing.assert_allclose(
            value,
            prior_summary["refined_baseline"]["geometry"][key],
            rtol=1e-12,
            atol=1e-12,
        )
    assert geometry_allowed(warm_geometry, policy)
    result["warm_baseline"] = {
        "force": warm_force,
        "loss": warm_loss,
        "geometry": warm_geometry,
        "maximum_change_from_saved_endpoint_m": float((warm_u - saved_u).abs().max()),
        "archive": record(warm_archive_path),
    }
    write_json(output / "summary.json", result)

    try:
        strict_u = runtime.primal(material_at(q), boundary_at(pose_z), warm_u)
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
            save_npz(
                output / "strict-baseline-failed.npz",
                displacement_m=failed.cpu().numpy(),
                activation_inv=q.cpu().numpy(),
                pose_normalized=pose_z.cpu().numpy(),
                pose_rad_m=physical_pose(pose_z).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            result["strict_baseline"].update(failed_state_evidence(failed, warm_det))
            result["strict_baseline"]["failed_archive"] = record(
                output / "strict-baseline-failed.npz"
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
        "maximum_change_from_warm_baseline_m": float((strict_u - warm_u).abs().max()),
        "rms_change_from_saved_endpoint_m": float(
            (strict_u - saved_u).square().mean().sqrt()
        ),
        "loss_change_from_warm_baseline": strict_loss - warm_loss,
        "loss_change_from_saved_endpoint": strict_loss - saved_loss,
        "newly_inverted_original_cell_ids": retained[
            (warm_det > 0) & (strict_det <= 0)
        ].tolist(),
        "recovered_original_cell_ids": retained[
            (warm_det <= 0) & (strict_det > 0)
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
        result["reason"] = f"{refined_label} baseline refinement did not converge"
        result["refined_baseline"] = {
            "failure": str(error),
            "receipt": error.receipt,
            "forward": copy.deepcopy(runtime.last_forward),
        }
        if runtime.last_problem is not None and hasattr(
            runtime, "last_failed_displacement"
        ):
            failed = runtime.last_failed_displacement
            save_npz(
                output / "refined-baseline-failed.npz",
                displacement_m=failed.cpu().numpy(),
                activation_inv=q.cpu().numpy(),
                pose_normalized=pose_z.cpu().numpy(),
                pose_rad_m=physical_pose(pose_z).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            result["refined_baseline"].update(failed_state_evidence(failed, strict_det))
            result["refined_baseline"]["failed_archive"] = record(
                output / "refined-baseline-failed.npz"
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
        "loss_change_from_strict": refined_loss - strict_loss,
        "maximum_displacement_change_from_strict_m": float(
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
        result["reason"] = (
            f"{refined_label} baseline refinement is outside physical gates"
        )
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    result["force_refinement"] = {
        "parameters_unchanged": True,
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "warm_to_strict_loss_change": strict_loss - warm_loss,
        "strict_to_refined_loss_change": refined_loss - strict_loss,
        "warm_to_refined_loss_change": refined_loss - warm_loss,
        "warm_to_strict_maximum_displacement_m": float((strict_u - warm_u).abs().max()),
        "strict_to_refined_maximum_displacement_m": float(
            (refined_u - strict_u).abs().max()
        ),
        "warm_to_refined_maximum_displacement_m": float(
            (refined_u - warm_u).abs().max()
        ),
        "warm_to_refined_rms_displacement_m": float(
            (refined_u - warm_u).square().mean().sqrt()
        ),
        "newly_inverted_original_cell_ids_warm_to_refined": retained[
            (warm_det > 0) & (refined_det <= 0)
        ].tolist(),
        "recovered_original_cell_ids_warm_to_refined": retained[
            (warm_det <= 0) & (refined_det > 0)
        ].tolist(),
        "formal_loss_error_bound": False,
    }
    result["status"] = "baseline_refined_requires_probe"
    result["interpretation"] = (
        "The unchanged-parameter equilibrium passes both tighter force and original "
        "geometry gates. Its measured loss and displacement drift set an empirical "
        "resolution scale for a separate parameter probe; no inverse convergence "
        "or stationarity is inferred."
    )
    write_json(output / "summary.json", result)
    cherries.log_output(output / "summary.json")
    cherries.log_output(output / "strict-baseline.npz")
    cherries.log_output(output / "refined-baseline.npz")
    cherries.log_metrics(
        {
            "probe/strict_baseline_force_n": strict_force * 1e6,
            "probe/refined_baseline_force_n": refined_force * 1e6,
            "probe/strict_to_refined_loss_drift": abs(refined_loss - strict_loss),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
