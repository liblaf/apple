# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, PLR0912, PLR0915, TRY003
"""Probe finite Smile descent from two certified collision-off equilibria.

The archived 1e-10 and 1e-12 states are independently replayed at the exact
audited Smile004 parameters before fully re-equilibrated parameter trials.
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
PARENT_SHA256 = {
    "protocol.json": "fead9bd0afcbb0371c21c922804b5c2cc0fa6379ae6b44116796db3cc4e5b339",
    "summary.json": "fb9d765f53c61fe24045a88dfe34c31a229c1de19396cb7691bce586de6bbdbc",
    "progress.jsonl": "24057dfeb1fa084606c091b26b32cd449c9b6b56d97c0a7a606d8e9f98ef3c1a",
    "trials.jsonl": "09cc31499a9cff1d6566005fe9d3ead17854bac7e4b31157ec707d0133c28e52",
    "endpoint.npz": "b0a2c80f825f9fc2cffe5d7ff2b867fe69a1049a208321d2ad7b3cedb229b625",
    "checkpoint.pt": "76ccdc5e948c47debea1ce8632ea3b3746778d9e2c2ecfcba4df7604f8f1ad73",
    "independent-audit.json": "82b83aea48d15e0ff4f19e706e7a99d06f0c01d6be58999a4d6541081a2062a5",
}
EXPECTED_BASELINE_SUMMARY_SHA256 = (
    "0b429cfaad0dcf6340663d7fab533705c80fa06ee50c8e720d4532caccffd207"
)
EXPECTED_STRICT_SHA256 = (
    "d18792f362fc593e9d2e0f751d0fc561d44e65181dfe2474ae7277d9e9cb76e5"
)
EXPECTED_REFINED_SHA256 = (
    "b50cd8730e779ada304dcc02721aaad1f502d4adc5d8191cefbe077b2a94a852"
)
EXPECTED_BASELINE_SOURCE_SHA256 = (
    "dfae02b344f4ccc6c3e769b50a36e803bafb7bf5a01ea44b4dd2e2b07e3bf67c"
)
EXPECTED_RUNTIME_SHA256 = (
    "197d37f7b8167ae8186603026433fc178036d4a76ce752808f6b7456d68e700d"
)
EXPECTED_SEED_SHA256 = (
    "4632c9a546db15453cb74377baf5544265b02092115ec8af86a69c73b7c6223e"
)


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/smile-004"
    baseline_dir: Path = GROUP / "data/smile-baseline-resolution-001"
    output_dir: Path = GROUP / "data/smile-feasible-descent-001"
    audit_name: str = "independent-audit.json"
    strict_forward_atol: float = 1e-10
    refined_forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    probe_alpha: float = 0.04
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


def preflight_inputs(run: Path, baseline_dir: Path) -> dict[str, Any]:
    """Check frozen source and completed input bytes before CUDA is used."""
    parent = {
        name: (run / name).is_file() and sha256(run / name) == expected
        for name, expected in PARENT_SHA256.items()
    }
    baseline = {
        name: (baseline_dir / name).is_file()
        and sha256(baseline_dir / name) == expected
        for name, expected in (
            ("summary.json", EXPECTED_BASELINE_SUMMARY_SHA256),
            ("strict-baseline.npz", EXPECTED_STRICT_SHA256),
            ("refined-baseline.npz", EXPECTED_REFINED_SHA256),
            (
                "sources/collision-off-expressions/82-smile-baseline-resolution.py",
                EXPECTED_BASELINE_SOURCE_SHA256,
            ),
        )
    }
    sources = {
        name: sha256(GROUP / "src" / name) == expected
        for name, expected in (
            ("82-smile-baseline-resolution.py", EXPECTED_BASELINE_SOURCE_SHA256),
            ("collision_off_runtime.py", EXPECTED_RUNTIME_SHA256),
            ("collision_off_seed.py", EXPECTED_SEED_SHA256),
        )
    }
    return {
        "schema": "collision-off-smile-feasible-descent-preflight-v1",
        "run_dir": str(run),
        "baseline_dir": str(baseline_dir),
        "parent_hashes_match": parent,
        "baseline_hashes_match": baseline,
        "source_hashes_match": sources,
        "ready": all(parent.values())
        and all(baseline.values())
        and all(sources.values()),
    }


def main(cfg: Config) -> None:
    run = cfg.run_dir.resolve()
    baseline_dir = cfg.baseline_dir.resolve()
    output = cfg.output_dir.resolve()
    assert run.is_dir()
    assert baseline_dir.is_dir()
    assert not output.exists()
    assert preflight_inputs(run, baseline_dir)["ready"]
    assert 0 < cfg.strict_forward_atol <= 1e-8
    assert 0 < cfg.refined_forward_atol < cfg.strict_forward_atol
    assert cfg.strict_forward_atol == 1e-10
    assert cfg.refined_forward_atol == 1e-12
    assert cfg.probe_alpha == 0.04
    assert 0 < cfg.adjoint_rtol <= 1e-7
    assert cfg.probe_alpha > 0
    assert cfg.loss_resolution_abs >= 0
    assert cfg.wall_seconds > 0
    assert cfg.unshifted_wall_seconds > 0
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
    assert protocol["expression_name"] == "Smile"
    assert source_summary["status"] == "pose_projection_failed"
    assert audit["schema"] == "collision-off-expression-independent-audit-v1"
    assert audit["expression_name"] == "Smile"
    assert audit["valid_forward"] is True
    assert audit["inputs"]["protocol"]["sha256"] == PARENT_SHA256["protocol.json"]
    assert audit["inputs"]["summary"]["sha256"] == PARENT_SHA256["summary.json"]
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
    baseline_summary_path = baseline_dir / "summary.json"
    baseline_source = (
        baseline_dir
        / "sources/collision-off-expressions/82-smile-baseline-resolution.py"
    )
    strict_archive_path = baseline_dir / "strict-baseline.npz"
    refined_archive_path = baseline_dir / "refined-baseline.npz"
    assert sha256(baseline_summary_path) == EXPECTED_BASELINE_SUMMARY_SHA256
    assert sha256(baseline_source) == EXPECTED_BASELINE_SOURCE_SHA256
    assert (
        sha256(GROUP / "src/82-smile-baseline-resolution.py")
        == EXPECTED_BASELINE_SOURCE_SHA256
    )
    assert sha256(strict_archive_path) == EXPECTED_STRICT_SHA256
    assert sha256(refined_archive_path) == EXPECTED_REFINED_SHA256
    assert sha256(GROUP / "src/collision_off_runtime.py") == EXPECTED_RUNTIME_SHA256
    assert sha256(GROUP / "src/collision_off_seed.py") == EXPECTED_SEED_SHA256
    baseline_summary = json.loads(baseline_summary_path.read_text())
    assert (
        baseline_summary["schema"] == "collision-off-smile-saved-baseline-resolution-v1"
    )
    assert baseline_summary["status"] == "baseline_refined_requires_probe"
    assert baseline_summary["inverse_updates"] == 0
    assert baseline_summary["collision_enabled"] is False
    assert baseline_summary["ccd_enabled"] is False
    assert (
        baseline_summary["inputs"]["runtime_source"]["sha256"]
        == EXPECTED_RUNTIME_SHA256
    )
    assert baseline_summary["inputs"]["seed_source"]["sha256"] == EXPECTED_SEED_SHA256
    assert (
        baseline_summary["source_snapshot_sha256"][
            "collision-off-expressions/82-smile-baseline-resolution.py"
        ]
        == EXPECTED_BASELINE_SOURCE_SHA256
    )
    for key, current in (
        ("protocol", protocol_path),
        ("summary", summary_path),
        ("endpoint", endpoint_path),
        ("checkpoint", checkpoint_path),
        ("progress", run / "progress.jsonl"),
        ("trials", run / "trials.jsonl"),
        ("independent_audit", audit_path),
    ):
        assert baseline_summary["inputs"][key]["sha256"] == sha256(current), key
    assert (
        baseline_summary["strict_baseline"]["archive"]["sha256"]
        == EXPECTED_STRICT_SHA256
    )
    assert (
        baseline_summary["refined_baseline"]["archive"]["sha256"]
        == EXPECTED_REFINED_SHA256
    )
    assert baseline_summary["strict_baseline"]["within_declared_policy"] is True
    assert baseline_summary["refined_baseline"]["within_declared_policy"] is True
    assert baseline_summary["strict_force_atol"] == cfg.strict_forward_atol
    assert baseline_summary["refined_force_atol"] == cfg.refined_forward_atol
    assert baseline_summary["saved_seed"]["within_declared_policy"] is True
    assert baseline_summary["saved_seed"]["within_original_force_contract"] is False
    assert baseline_summary["saved_seed"]["source_checkpoint_optimizer_steps"] == {
        "q": 86,
        "pose": 86,
    }
    assert checkpoint["optimizer_steps"] == {"q": 86, "pose": 86}
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
    assert checkpoint["convergence_reference"] is not None
    assert len(checkpoint["convergence_history_tail"]) == 11

    def bound_baseline(path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as archive:
            displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
            np.testing.assert_array_equal(archive["activation_inv"], endpoint_q)
            np.testing.assert_array_equal(
                archive["pose_normalized"], pose_z_cpu.numpy()
            )
            np.testing.assert_array_equal(
                archive["pose_rad_m"], checkpoint["pose_rad_m"].numpy()
            )
            np.testing.assert_array_equal(archive["active_cell_ids"], active_source_ids)
        assert displacement.shape == endpoint_u.shape
        assert np.isfinite(displacement).all()
        return displacement

    coarse_u_np = bound_baseline(strict_archive_path)
    fine_u_np = bound_baseline(refined_archive_path)
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
        "schema": "collision-off-smile-finite-feasible-descent-probe-v1",
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
            "independent_audit": record(audit_path),
            "blendshapes": record(target_path),
            "neutral_endpoint": record(neutral_endpoint),
            "reference_repair": record(reference),
            "neutral_active_strain_fields": record(fields_path),
            "progress": record(run / "progress.jsonl"),
            "trials": record(run / "trials.jsonl"),
            "baseline_summary": record(baseline_summary_path),
            "strict_baseline_archive": record(strict_archive_path),
            "refined_baseline_archive": record(refined_archive_path),
            "baseline_82_source": record(baseline_source),
            "runtime_source": record(GROUP / "src/collision_off_runtime.py"),
            "seed_source": record(GROUP / "src/collision_off_seed.py"),
        },
        "strict_force_atol": cfg.strict_forward_atol,
        "refined_force_atol": cfg.refined_forward_atol,
        "wall_seconds": cfg.wall_seconds,
        "unshifted_wall_seconds": cfg.unshifted_wall_seconds,
        "source_snapshot_sha256": source_records,
        "inversion_policy": policy,
        "expression_name": protocol["expression_name"],
        "baseline_method": "independent replay of hash-bound Smile82 at 1e-10 and 1e-12 at exact audited Smile004 parameters",
        "trial_method": "1e-10 solve from strict baseline tangent seed, followed by 1e-12 refinement of that trial",
        "force_level_keys": {"strict": strict_key, "refined": refined_key},
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
    coarse_u = torch.as_tensor(coarse_u_np, device=device, dtype=dtype)
    fine_u = torch.as_tensor(fine_u_np, device=device, dtype=dtype)
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

    def failed_trial_evidence(
        u: torch.Tensor,
        trial_q: torch.Tensor,
        trial_pose: torch.Tensor,
        reference_det: np.ndarray,
        path: Path,
    ) -> dict[str, Any]:
        save_npz(
            path,
            displacement_m=u.detach().cpu().numpy(),
            activation_inv=trial_q.detach().cpu().numpy(),
            pose_normalized=trial_pose.detach().cpu().numpy(),
            pose_rad_m=physical_pose(trial_pose).detach().cpu().numpy(),
            active_cell_ids=active_source_ids,
        )
        evidence: dict[str, Any] = {
            "archive": record(path),
            "displacement_finite": bool(torch.isfinite(u).all()),
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
            retained = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
            evidence["newly_inverted_original_cell_ids"] = retained[
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
    result["saved_seed"] = {
        "force": saved_force,
        "loss": saved_loss,
        "geometry": saved_geometry,
        "within_declared_policy": geometry_allowed(saved_geometry, policy),
    }
    result["saved_seed"]["within_original_force_contract"] = (
        saved_force <= protocol["force_contract"]["atol"]
    )
    np.testing.assert_allclose(
        saved_force, baseline_summary["saved_seed"]["force"], rtol=1e-6, atol=1e-14
    )
    np.testing.assert_allclose(
        saved_loss, baseline_summary["saved_seed"]["loss"], rtol=0, atol=1e-12
    )
    for key, value in saved_geometry.items():
        np.testing.assert_allclose(
            value,
            baseline_summary["saved_seed"]["geometry"][key],
            rtol=1e-10,
            atol=1e-12,
        )
    assert geometry_allowed(saved_geometry, policy)
    strict_u = coarse_u
    refined_u = fine_u
    retained = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
    result["baseline_replay"] = {}
    try:
        for label, state, archive_path, source_baseline, tolerance in (
            (
                "strict",
                strict_u,
                strict_archive_path,
                baseline_summary["strict_baseline"],
                cfg.strict_forward_atol,
            ),
            (
                "refined",
                refined_u,
                refined_archive_path,
                baseline_summary["refined_baseline"],
                cfg.refined_forward_atol,
            ),
        ):
            force = direct_free_force(state)
            loss = float(objective(state))
            geometry = geometry_metrics(physics, state)
            determinants = determinant_ratios(physics, state)
            replay = {
                "force": force,
                "source_reported_force": source_baseline["force"],
                "loss": loss,
                "source_reported_loss": source_baseline["loss"],
                "geometry": geometry,
                "within_declared_policy": geometry_allowed(geometry, policy),
                "archive": record(archive_path),
                "maximum_change_from_saved_endpoint_m": float(
                    (state - saved_u).abs().max()
                ),
            }
            result["baseline_replay"][label] = replay
            np.testing.assert_allclose(
                force, source_baseline["force"], rtol=1e-6, atol=1e-14
            )
            np.testing.assert_allclose(
                loss, source_baseline["loss"], rtol=0, atol=1e-12
            )
            for key, value in geometry.items():
                np.testing.assert_allclose(
                    value, source_baseline["geometry"][key], rtol=1e-12, atol=1e-12
                )
            assert len(determinants) == len(retained)
            assert force <= tolerance, (label, force, tolerance)
            assert replay["within_declared_policy"], label
    except (AssertionError, RuntimeError, ValueError) as error:
        result["status"] = "unresolved_baseline"
        result["reason"] = f"archived baseline replay failed: {error}"
        assert preflight_inputs(run, baseline_dir)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    strict_force = result["baseline_replay"]["strict"]["force"]
    strict_loss = result["baseline_replay"]["strict"]["loss"]
    refined_loss = result["baseline_replay"]["refined"]["loss"]
    strict_det = determinant_ratios(physics, strict_u)
    refined_det = determinant_ratios(physics, refined_u)
    shutil.copy2(strict_archive_path, output / "strict-baseline.npz")
    shutil.copy2(refined_archive_path, output / "refined-baseline.npz")
    assert sha256(output / "strict-baseline.npz") == EXPECTED_STRICT_SHA256
    assert sha256(output / "refined-baseline.npz") == EXPECTED_REFINED_SHA256
    result["strict_baseline"] = {
        **result["baseline_replay"]["strict"],
        "archive": record(output / "strict-baseline.npz"),
        "source_archive": record(strict_archive_path),
    }
    result["refined_baseline"] = {
        **result["baseline_replay"]["refined"],
        "archive": record(output / "refined-baseline.npz"),
        "source_archive": record(refined_archive_path),
        "loss_change_from_strict": refined_loss - strict_loss,
        "maximum_displacement_change_from_strict_m": float(
            (refined_u - strict_u).abs().max()
        ),
        "newly_inverted_original_cell_ids": retained[
            (strict_det > 0) & (refined_det <= 0)
        ].tolist(),
        "recovered_original_cell_ids": retained[
            (strict_det <= 0) & (refined_det > 0)
        ].tolist(),
    }
    baseline_refinement_drift = abs(refined_loss - strict_loss)
    result["loss_resolution_screen"] = {
        "absolute_floor": cfg.loss_resolution_abs,
        "baseline_refinement_loss_change_abs": baseline_refinement_drift,
        "baseline_refinement_max_displacement_m": float(
            (refined_u - strict_u).abs().max()
        ),
        "screen_rule": "max(absolute_floor, 10*(baseline_refinement_drift + trial_refinement_drift))",
        "formal_error_bound": False,
    }
    write_json(output / "summary.json", result)
    runtime.tolerances["atol"] = cfg.refined_forward_atol

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
                "shift comparison changed the refined baseline state",
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
            assert preflight_inputs(run, baseline_dir)["ready"]
            result["input_hashes_rechecked_at_terminal"] = True
            write_json(output / "summary.json", result)
            cherries.log_output(output / "summary.json")
            return
        result["shift_comparison"][str(shift)] = evidence
        write_json(output / "summary.json", result)
    result["shift_comparison"]["gradient_difference"] = {
        "q": gradient_difference(gradients[0.001][0], gradients[0.0001][0]),
        "pose": gradient_difference(gradients[0.001][1], gradients[0.0001][1]),
    }
    # Probe directions require the unshifted physical adjoint. A damped
    # fallback would retain the ambiguity that this diagnostic must resolve.
    global_deadline = runtime.deadline
    runtime.deadline = min(
        global_deadline, time.perf_counter() + cfg.unshifted_wall_seconds
    )
    try:
        try:
            result["shift_comparison"]["0.0"] = gradient_at_shift(0.0)
        except (ForwardConvergenceError, AssertionError, RuntimeError) as error:
            result["shift_comparison"]["0.0"] = {
                "status": "unresolved",
                "failure": str(error),
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
            }
    finally:
        runtime.deadline = global_deadline
    if 0.0 not in gradients:
        result["status"] = "unshifted_adjoint_unresolved"
        assert preflight_inputs(run, baseline_dir)["ready"]
        result["input_hashes_rechecked_at_terminal"] = True
        write_json(output / "summary.json", result)
        cherries.log_output(output / "summary.json")
        return
    result["probe_gradient_shift"] = 0.0
    write_json(output / "summary.json", result)

    gq, gp = gradients[0.0]
    q_max = float(gq.abs().max())
    pose_max = float(gp.abs().max())
    dq = torch.zeros_like(gq) if q_max == 0 else -0.02 * gq / q_max
    dp = torch.zeros_like(gp) if pose_max == 0 else -0.1 * gp / pose_max
    directions = {
        "pose_only": (torch.zeros_like(dq), dp),
        "joint": (dq, dp),
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
            runtime.tolerances["atol"] = cfg.strict_forward_atol
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
                        strict_u,
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
                    strict_u,
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
                trial_u_strict = runtime.primal(
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
                    row["failed_state"] = failed_trial_evidence(
                        runtime.last_failed_displacement,
                        trial_q,
                        trial_pose,
                        strict_det,
                        output / f"probe-{name}-{alpha:.3e}-{strict_label}-failed.npz",
                    )
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            trial_force_strict = direct_free_force(trial_u_strict)
            trial_geometry_strict = geometry_metrics(physics, trial_u_strict)
            trial_loss_strict = float(objective(trial_u_strict))
            row[strict_key] = {
                "force": trial_force_strict,
                "solver_reported_force": float(runtime.last_forward["grad_norm"]),
                "geometry": trial_geometry_strict,
                "loss": trial_loss_strict,
                "loss_change_from_baseline": trial_loss_strict - strict_loss,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            strict_archive = output / f"probe-{name}-{alpha:.3e}-{strict_label}.npz"
            save_npz(
                strict_archive,
                displacement_m=trial_u_strict.cpu().numpy(),
                activation_inv=trial_q.cpu().numpy(),
                pose_normalized=trial_pose.cpu().numpy(),
                pose_rad_m=physical_pose(trial_pose).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            row[strict_key]["archive"] = record(strict_archive)
            if trial_force_strict > cfg.strict_forward_atol or not geometry_allowed(
                trial_geometry_strict, policy
            ):
                row["status"] = "strict_trial_outside_physical_gates_unresolved"
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            runtime.tolerances["atol"] = cfg.refined_forward_atol
            try:
                trial_u_refined = runtime.primal(
                    material_at(trial_q), boundary_at(trial_pose), trial_u_strict
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
                    row["refinement_failed_state"] = failed_trial_evidence(
                        runtime.last_failed_displacement,
                        trial_q,
                        trial_pose,
                        determinant_ratios(physics, trial_u_strict),
                        output / f"probe-{name}-{alpha:.3e}-{refined_label}-failed.npz",
                    )
                    row["refinement_failed_state"][
                        "refined_from_strict_trial_archive"
                    ] = row[strict_key]["archive"]
                result["probes"].append(row)
                write_json(output / "summary.json", result)
                continue
            trial_force_refined = direct_free_force(trial_u_refined)
            trial_geometry_refined = geometry_metrics(physics, trial_u_refined)
            trial_loss_refined = float(objective(trial_u_refined))
            trial_det = determinant_ratios(physics, trial_u_refined)
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
            row[refined_key] = {
                "force": trial_force_refined,
                "solver_reported_force": float(runtime.last_forward["grad_norm"]),
                "geometry": trial_geometry_refined,
                "loss": trial_loss_refined,
                "loss_change_from_baseline": trial_loss_refined - refined_loss,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            trial_refinement_drift = abs(trial_loss_refined - trial_loss_strict)
            resolution = max(
                cfg.loss_resolution_abs,
                10 * (baseline_refinement_drift + trial_refinement_drift),
            )
            change_strict = trial_loss_strict - strict_loss
            change_refined = trial_loss_refined - refined_loss
            row["loss_resolution_screen"] = {
                "baseline_refinement_drift": baseline_refinement_drift,
                "trial_refinement_drift": trial_refinement_drift,
                "used": resolution,
                "formal_error_bound": False,
            }
            change_disagreement = abs(change_strict - change_refined) / max(
                abs(change_strict), abs(change_refined), 1e-30
            )
            row["loss_change_relative_disagreement"] = change_disagreement
            row["within_declared_policy_at_both_force_levels"] = geometry_allowed(
                trial_geometry_strict, policy
            ) and geometry_allowed(trial_geometry_refined, policy)
            seed_response_m = float((trial_seed - strict_u).abs().max())
            correction_m = float((trial_u_refined - trial_seed).abs().max())
            parameter_response_strict_m = float((trial_u_strict - strict_u).abs().max())
            parameter_response_refined_m = float(
                (trial_u_refined - refined_u).abs().max()
            )
            baseline_refinement_m = float((refined_u - strict_u).abs().max())
            trial_refinement_m = float((trial_u_refined - trial_u_strict).abs().max())
            correction_ratio = correction_m / max(seed_response_m, 1e-15)
            row["correction_guard"] = {
                "seed_parameter_response_max_m": seed_response_m,
                "seed_to_refined_correction_max_m": correction_m,
                "strict_baseline_to_strict_trial_response_max_m": parameter_response_strict_m,
                "refined_baseline_to_refined_trial_response_max_m": parameter_response_refined_m,
                "baseline_refinement_max_m": baseline_refinement_m,
                "trial_refinement_max_m": trial_refinement_m,
                "correction_to_seed_response_ratio": correction_ratio,
                "finite_parameter_response_resolved": parameter_response_refined_m
                > 10 * (baseline_refinement_m + trial_refinement_m),
                "seed_correction_ratio_within_10": correction_ratio <= 10,
                "formal_error_bound": False,
            }
            row["resolved_loss_decrease"] = (
                row["within_declared_policy_at_both_force_levels"]
                and trial_force_strict <= cfg.strict_forward_atol
                and trial_force_refined <= cfg.refined_forward_atol
                and change_strict < -resolution
                and change_refined < -resolution
                and change_disagreement <= 0.1
            )
            row["status"] = (
                "feasible_descent_witness"
                if row["resolved_loss_decrease"]
                else "no_resolved_witness"
            )
            refined_archive = output / f"probe-{name}-{alpha:.3e}-{refined_label}.npz"
            save_npz(
                refined_archive,
                displacement_m=trial_u_refined.cpu().numpy(),
                activation_inv=trial_q.cpu().numpy(),
                pose_normalized=trial_pose.cpu().numpy(),
                pose_rad_m=physical_pose(trial_pose).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            row[refined_key]["archive"] = record(refined_archive)
            witness |= row["resolved_loss_decrease"]
            result["probes"].append(row)
            write_json(output / "summary.json", result)

    result["finite_scale_checks"] = {}
    for name in directions:
        rows = [
            row
            for row in result["probes"]
            if row["direction"] == name and refined_key in row
        ]
        if len(rows) != 2:
            result["finite_scale_checks"][name] = {
                "status": "unresolved_incomplete_pair"
            }
            continue
        rows.sort(key=lambda row: row["alpha"], reverse=True)
        actual_slopes = [
            row[refined_key]["loss_change_from_baseline"] / row["alpha"] for row in rows
        ]
        predicted_slopes = [row["predicted_loss_change"] / row["alpha"] for row in rows]
        actual_scale_disagreement = abs(actual_slopes[0] - actual_slopes[1]) / max(
            abs(actual_slopes[0]), abs(actual_slopes[1]), 1e-30
        )
        predicted_scale_disagreement = abs(
            predicted_slopes[0] - predicted_slopes[1]
        ) / max(abs(predicted_slopes[0]), abs(predicted_slopes[1]), 1e-30)
        gradient_slope_disagreement = [
            abs(actual - predicted) / max(abs(actual), abs(predicted), 1e-30)
            for actual, predicted in zip(actual_slopes, predicted_slopes, strict=True)
        ]
        result["finite_scale_checks"][name] = {
            "alphas": [row["alpha"] for row in rows],
            "refined_finite_difference_slopes": actual_slopes,
            "unshifted_adjoint_linear_predictions_per_alpha": predicted_slopes,
            "actual_scale_relative_disagreement": actual_scale_disagreement,
            "predicted_scale_relative_disagreement": predicted_scale_disagreement,
            "actual_vs_predicted_relative_disagreement": gradient_slope_disagreement,
            "scope": "finite scales only; no local gradient conclusion",
            "formal_error_bound": False,
        }
    if result["status"] != "time_budget_exhausted_unresolved":
        result["status"] = (
            "feasible_descent_witness" if witness else "no_witness_unresolved"
        )
    if witness:
        result["interpretation"] = (
            "A finite feasible lower-loss parameter endpoint survives both force levels and the empirical resolution screen. Finite-scale slopes do not establish a local derivative or stationarity claim."
        )
    else:
        result["interpretation"] = (
            "These finite probes did not establish stationarity, infeasibility, or a resolved feasible descent."
        )
    assert preflight_inputs(run, baseline_dir)["ready"]
    result["input_hashes_rechecked_at_terminal"] = True
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
    os.environ["CHERRIES_TAGS"] = "collision-off,independent-probe,Smile,paratera-4090"
    cherries.main(main, profile=ProfileJoint)
