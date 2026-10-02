# Copyright (c) 2026 liblaf
# ruff: noqa: C901, PLR0915
"""Refine Smile004's saved equilibrium at fixed parameters and original policy.

The saved state is a seed, especially if its 1e-8 force sits just beyond the
gate on independent replay. Only separately validated tighter solves are
called refined baselines. No inverse update or convergence claim is made.
"""

from __future__ import annotations

import copy
import json
import math
import os
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
EXPECTED_PROTOCOL_SHA256 = (
    "fead9bd0afcbb0371c21c922804b5c2cc0fa6379ae6b44116796db3cc4e5b339"
)
EXPECTED_SUMMARY_SHA256 = (
    "fb9d765f53c61fe24045a88dfe34c31a229c1de19396cb7691bce586de6bbdbc"
)
EXPECTED_ENDPOINT_SHA256 = (
    "b0a2c80f825f9fc2cffe5d7ff2b867fe69a1049a208321d2ad7b3cedb229b625"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "76ccdc5e948c47debea1ce8632ea3b3746778d9e2c2ecfcba4df7604f8f1ad73"
)
EXPECTED_PROGRESS_SHA256 = (
    "24057dfeb1fa084606c091b26b32cd449c9b6b56d97c0a7a606d8e9f98ef3c1a"
)
EXPECTED_TRIALS_SHA256 = (
    "09cc31499a9cff1d6566005fe9d3ead17854bac7e4b31157ec707d0133c28e52"
)
EXPECTED_AUDIT_SHA256 = (
    "82b83aea48d15e0ff4f19e706e7a99d06f0c01d6be58999a4d6541081a2062a5"
)
EXPECTED_FAILURE_ANALYSIS_SHA256 = (
    "535c5bc4bd15934be319e04aab8d803bd125bb26379ea4782a6c93187459866e"
)
EXPECTED_RUNTIME_SHA256 = (
    "197d37f7b8167ae8186603026433fc178036d4a76ce752808f6b7456d68e700d"
)
EXPECTED_SEED_SHA256 = (
    "4632c9a546db15453cb74377baf5544265b02092115ec8af86a69c73b7c6223e"
)
EXPECTED_FIT_SHA256 = "9f1965dfff08e587cf8cbc64fa97f555cc696f7c8bdefb122d6d78ddb4110ced"
EXPECTED_QUEUE_SHA256 = (
    "b4fc4108d25a95e9bd79076e5c54d670951750c59a83bacf8bfa323989c22b7f"
)


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/smile-004"
    output_dir: Path = GROUP / "data/smile-baseline-resolution-001"
    audit_name: str = "independent-audit.json"
    strict_forward_atol: float = 1e-10
    refined_forward_atol: float = 1e-12
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


def preflight_run(run: Path) -> dict[str, Any]:
    """Report whether the frozen saved run and completed audit are available."""
    required = (
        "protocol.json",
        "summary.json",
        "endpoint.npz",
        "checkpoint.pt",
        "progress.jsonl",
        "trials.jsonl",
        "independent-audit.json",
    )
    paths = {name: run / name for name in required}
    present = {name: path.is_file() for name, path in paths.items()}
    sources = {
        name: sha256(GROUP / "src" / name) == expected
        for name, expected in (
            ("collision_off_runtime.py", EXPECTED_RUNTIME_SHA256),
            ("collision_off_seed.py", EXPECTED_SEED_SHA256),
            ("10-fit.py", EXPECTED_FIT_SHA256),
            ("40-run-queue.py", EXPECTED_QUEUE_SHA256),
        )
    }
    run_hashes = {
        name: present[name] and sha256(paths[name]) == expected
        for name, expected in (
            ("protocol.json", EXPECTED_PROTOCOL_SHA256),
            ("summary.json", EXPECTED_SUMMARY_SHA256),
            ("endpoint.npz", EXPECTED_ENDPOINT_SHA256),
            ("checkpoint.pt", EXPECTED_CHECKPOINT_SHA256),
            ("progress.jsonl", EXPECTED_PROGRESS_SHA256),
            ("trials.jsonl", EXPECTED_TRIALS_SHA256),
        )
    }
    audit_hash_bound = len(EXPECTED_AUDIT_SHA256) == 64
    audit_bound = (
        present["independent-audit.json"]
        and audit_hash_bound
        and sha256(paths["independent-audit.json"]) == EXPECTED_AUDIT_SHA256
    )
    failure_analysis_path = GROUP / "data/smile-004-failure-analysis.json"
    failure_analysis_bound = (
        failure_analysis_path.is_file()
        and sha256(failure_analysis_path) == EXPECTED_FAILURE_ANALYSIS_SHA256
    )
    return {
        "schema": "collision-off-smile-baseline-resolution-preflight-v1",
        "run_dir": str(run.resolve()),
        "present": present,
        "run_hashes_bound": run_hashes,
        "audit_hash_bound": audit_hash_bound,
        "audit_bound": audit_bound,
        "failure_analysis_bound": failure_analysis_bound,
        "sources_bound": sources,
        "ready": all(present.values())
        and all(run_hashes.values())
        and audit_bound
        and failure_analysis_bound
        and all(sources.values()),
    }


def main(cfg: Config) -> None:
    run = cfg.run_dir.resolve()
    output = cfg.output_dir.resolve()
    assert run.is_dir()
    assert not output.exists()
    preflight = preflight_run(run)
    assert preflight["ready"], preflight
    assert 0 < cfg.strict_forward_atol <= 1e-8
    assert 0 < cfg.refined_forward_atol < cfg.strict_forward_atol
    assert cfg.strict_forward_atol == 1e-10
    assert cfg.refined_forward_atol == 1e-12
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
    progress_path = run / "progress.jsonl"
    trials_path = run / "trials.jsonl"
    audit_path = run / cfg.audit_name
    failure_analysis_path = GROUP / "data/smile-004-failure-analysis.json"
    assert sha256(protocol_path) == EXPECTED_PROTOCOL_SHA256
    assert sha256(summary_path) == EXPECTED_SUMMARY_SHA256
    assert sha256(endpoint_path) == EXPECTED_ENDPOINT_SHA256
    assert sha256(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256
    assert sha256(progress_path) == EXPECTED_PROGRESS_SHA256
    assert sha256(trials_path) == EXPECTED_TRIALS_SHA256
    assert len(EXPECTED_AUDIT_SHA256) == 64, (
        "Bind the completed audit SHA before launch"
    )
    assert sha256(audit_path) == EXPECTED_AUDIT_SHA256
    assert sha256(failure_analysis_path) == EXPECTED_FAILURE_ANALYSIS_SHA256
    for name, expected in (
        ("collision_off_runtime.py", EXPECTED_RUNTIME_SHA256),
        ("collision_off_seed.py", EXPECTED_SEED_SHA256),
        ("10-fit.py", EXPECTED_FIT_SHA256),
        ("40-run-queue.py", EXPECTED_QUEUE_SHA256),
    ):
        assert sha256(GROUP / "src" / name) == expected, name
    protocol = json.loads(protocol_path.read_text())
    source_summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    failure_analysis = json.loads(failure_analysis_path.read_text())
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["collision_enabled"] is False
    assert protocol["expression_name"] == "Smile"
    assert source_summary["status"] == "pose_projection_failed"
    assert audit["schema"] == "collision-off-expression-independent-audit-v1"
    assert audit["expression_name"] == "Smile"
    assert audit["inputs"]["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
    assert audit["inputs"]["endpoint"]["sha256"] == sha256(endpoint_path)
    assert audit["inputs"]["summary"]["sha256"] == sha256(summary_path)
    assert source_summary["endpoint"]["sha256"] == sha256(endpoint_path)
    assert failure_analysis["run"] == "smile-004"
    assert failure_analysis["summary"]["sha256"] == EXPECTED_SUMMARY_SHA256
    assert failure_analysis["helper"]["sha256"] == EXPECTED_SEED_SHA256
    assert failure_analysis["fit_source"]["sha256"] == EXPECTED_FIT_SHA256
    assert failure_analysis["queue_source"]["sha256"] == EXPECTED_QUEUE_SHA256
    supporting = {
        Path(item["path"]).name: item["sha256"]
        for item in failure_analysis["supporting_files"]
    }
    assert supporting == {
        "progress.jsonl": EXPECTED_PROGRESS_SHA256,
        "trials.jsonl": EXPECTED_TRIALS_SHA256,
    }
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
    assert checkpoint["optimizer_steps"] == {"q": 86, "pose": 86}
    assert checkpoint["optimizer_step"] == 86
    assert checkpoint["iteration"] == 11
    assert checkpoint["local_iteration"] == 11
    assert len(checkpoint["moments"]) == 4
    assert all(bool(torch.isfinite(value).all()) for value in checkpoint["moments"])
    assert (
        checkpoint["moments"][0].shape
        == checkpoint["moments"][1].shape
        == checkpoint["activation_inv"].shape
    )
    assert (
        checkpoint["moments"][2].shape
        == checkpoint["moments"][3].shape
        == checkpoint["pose_normalized"].shape
    )
    assert isinstance(checkpoint["convergence_reference"], dict)
    assert len(checkpoint["convergence_history_tail"]) == 11
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
    final_progress = json.loads(progress_path.read_text().splitlines()[-1])
    assert final_progress == source_summary["final"]
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
        "schema": "collision-off-smile-saved-baseline-resolution-v1",
        "status": "running",
        "inverse_converged": False,
        "inverse_updates": 0,
        "collision_enabled": False,
        "ccd_enabled": False,
        "inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "endpoint": record(endpoint_path),
            "checkpoint": record(checkpoint_path),
            "progress": record(progress_path),
            "trials": record(trials_path),
            "independent_audit": record(audit_path),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "reference_repair": record(reference),
            "neutral_active_strain_fields": record(fields_path),
            "failure_analysis": record(failure_analysis_path),
            "fit_source": record(GROUP / "src/10-fit.py"),
            "queue_source": record(GROUP / "src/40-run-queue.py"),
            "seed_source": record(GROUP / "src/collision_off_seed.py"),
            "runtime_source": record(GROUP / "src/collision_off_runtime.py"),
        },
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "wall_seconds": cfg.wall_seconds,
        "source_snapshot_sha256": source_records,
        "inversion_policy": policy,
        "expression_name": protocol["expression_name"],
        "warmstart_method": "hash-bound saved Smile004 endpoint at unchanged q and normalized pose",
        "audited_seed_valid_forward": audit["valid_forward"],
        "audited_seed_force": audit["force"],
        "force_level_keys": {"strict": strict_key, "refined": refined_key},
        "parameter_probe_enabled": False,
    }
    write_json(output / "summary.json", result)
    assert preflight_run(run)["ready"]
    for key, current in (
        ("protocol", protocol_path),
        ("summary", summary_path),
        ("endpoint", endpoint_path),
        ("checkpoint", checkpoint_path),
        ("progress", progress_path),
        ("trials", trials_path),
        ("independent_audit", audit_path),
        ("failure_analysis", failure_analysis_path),
    ):
        assert result["inputs"][key]["sha256"] == sha256(current), key

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
    warm_u = saved_u
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
    warm_det = determinant_ratios(physics, saved_u)
    np.testing.assert_allclose(
        saved_loss, source_summary["final"]["loss"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        saved_force,
        audit["force"]["raw_free_force_mpa_m2"],
        rtol=1e-6,
        atol=1e-14,
    )
    result["saved_seed"] = {
        "force": saved_force,
        "loss": saved_loss,
        "geometry": saved_geometry,
        "within_declared_policy": geometry_allowed(saved_geometry, policy),
        "within_original_force_contract": saved_force
        <= protocol["force_contract"]["atol"],
        "audit_valid_forward": audit["valid_forward"],
        "source_checkpoint_optimizer_steps": checkpoint["optimizer_steps"],
        "source_checkpoint_local_iteration": checkpoint["local_iteration"],
    }
    for key, value in saved_geometry.items():
        np.testing.assert_allclose(
            value, failure_analysis["saved_geometry"][key], rtol=1e-10, atol=1e-12
        )
    np.testing.assert_allclose(
        saved_force * 1e6,
        failure_analysis["saved_force_norm_n"],
        rtol=1e-6,
        atol=1e-10,
    )
    write_json(output / "summary.json", result)
    if not geometry_allowed(saved_geometry, policy):
        result["status"] = "unresolved_seed_geometry"
        result["reason"] = (
            "saved Smile004 seed is outside the original retained-tet policy"
        )
        assert preflight_run(run)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    warm_loss = saved_loss

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
        assert preflight_run(run)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
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
        "maximum_change_from_saved_seed_m": float((strict_u - warm_u).abs().max()),
        "rms_change_from_saved_endpoint_m": float(
            (strict_u - saved_u).square().mean().sqrt()
        ),
        "loss_change_from_saved_seed": strict_loss - warm_loss,
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
        assert preflight_run(run)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
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
        assert preflight_run(run)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
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
        assert preflight_run(run)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    result["force_refinement"] = {
        "parameters_unchanged": True,
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "saved_to_strict_loss_change": strict_loss - warm_loss,
        "strict_to_refined_loss_change": refined_loss - strict_loss,
        "saved_to_refined_loss_change": refined_loss - warm_loss,
        "saved_to_strict_maximum_displacement_m": float(
            (strict_u - warm_u).abs().max()
        ),
        "strict_to_refined_maximum_displacement_m": float(
            (refined_u - strict_u).abs().max()
        ),
        "saved_to_refined_maximum_displacement_m": float(
            (refined_u - warm_u).abs().max()
        ),
        "saved_to_refined_rms_displacement_m": float(
            (refined_u - warm_u).square().mean().sqrt()
        ),
        "newly_inverted_original_cell_ids_saved_to_refined": retained[
            (warm_det > 0) & (refined_det <= 0)
        ].tolist(),
        "recovered_original_cell_ids_saved_to_refined": retained[
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
    assert preflight_run(run)["ready"]
    result["input_hashes_rechecked_at_terminal"] = True
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
    os.environ["CHERRIES_TAGS"] = "collision-off,independent-probe,Smile,paratera-4090"
    cherries.main(main, profile=ProfileJoint)
