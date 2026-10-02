# Copyright (c) 2026 liblaf
# ruff: noqa: C901, E402, PLR0912, PLR0915, FBT001, FBT003, PT018
"""Bounded strict MouthOpen continuation from an audited pilot checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)
POSE_SCALE = torch.tensor([math.pi / 18.0] * 3 + [0.01] * 3, dtype=torch.float64)
MAX_ROTATION_INCREMENT_DEG = 1.0
MAX_TRANSLATION_INCREMENT_M = 0.001
PARENT_SHA256 = {
    "protocol.json": "098fdebd7473d2c6ab453a0fa0b27bcb748ed3449e6543c23708951dd61ca027",
    "summary.json": "ac88d5d18d2b6967883851d7568e4a127c8befe001a6b6422c2017fff1ddbaae",
    "progress.jsonl": "62e9a560e8cd6e42dbcced0f19e935f44ee1ee4a2dc1dfeedbe9d385fb8b2b14",
    "endpoint.npz": "bc46375f796164941e52b764a19f14f8b88236d4b3e7446bcb997967bd836992",
    "checkpoint.pt": "a85d193ea8010ff370735121c555119c14890fdfd9dc20b66968de2381eed29f",
    "independent-audit.json": "3fdb1db00a73b5d4f14e2384b082e589a8b222f54e116e9b5bda7c9b194c8930",
    "equilibrium-refinement.json": "7a4a72e594d0fab6c5dcf3ac42c45e013cb475fddff4ba65b1aa09b58b088283",
}
PARENT_FIT_SOURCE_SHA256 = (
    "98fb0d6d0d792ae2b9e44843fa61acf537825b95fce27b2e16e0444257f2d39e"
)
FIT_SOURCE_SHA256 = "9f1965dfff08e587cf8cbc64fa97f555cc696f7c8bdefb122d6d78ddb4110ced"
AUDIT_SOURCE_SHA256 = "8158f6413eb932d3df21edd28d1cf287f15cd898b24dc0eaa5b98a5adb56ed83"


class Config(cherries.BaseConfig):
    expression_name: str = "MouthOpen"
    output_dir: Path = GROUP / "data/mouthopen-strict-continuation-001"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    blendshape_dir: Path = GROUP / "data/blendshapes-isfixed-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    initialization_checkpoint: Path | None = None
    pilot_source_sha256: str = ""
    seed_method: str = "collision_off_tangent"
    exclude_fully_fixed_tets: bool = True
    maximum_inverted_tetrahedra: int = 100
    maximum_inverted_rest_volume_fraction: float = 0.0001
    max_rotation_increment_deg: float | None = None
    max_translation_increment_m: float | None = None
    predictor_relative_shift: float = 0.0
    predictor_rtol: float = 1e-7
    maximum_iterations: int = 25
    learning_rate: float = 0.02
    pose_learning_rate: float = 0.1
    forward_atol: float = 1e-12
    adjoint_rtol: float = 1e-7
    adjoint_relative_shift: float = 0.0
    max_newton_steps: int = 3000
    max_backtracks: int = 24
    wall_seconds: float | None = 3600
    no_contact_linear_max_steps: int = 3000
    initializer_wall_seconds: float = 3600
    ipc_threads: int = 4
    resume: bool = False
    continue_optimizer_state: bool = False
    q_only_iterations: int = 0
    project_pose_at_inversion_limit: bool = True
    projection_activation_threshold: float = 0.05
    projection_determinant_margin: float = 1e-6
    projection_probe_epsilon: float = 1e-4
    adaptive_trial_alpha: bool = True
    initial_trial_alpha: float = 0.04
    minimum_trial_alpha: float = 0.001
    convergence_patience: int = 10
    convergence_loss_rtol: float = 1e-6
    convergence_gradient_rtol: float = 1e-3
    convergence_gradient_atol: float = 1e-8
    enable_stationarity_monitor: bool = False


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def append(path: Path, row: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def save_torch(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def _bounded_pose(
    normalized: torch.Tensor,
    proposed_delta: torch.Tensor,
    *,
    max_rotation_deg: float | None = MAX_ROTATION_INCREMENT_DEG,
    max_translation_m: float | None = MAX_TRANSLATION_INCREMENT_M,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one physical SE(3) increment bounded on SO(3) and translation."""
    from mouthopen_pose_path import pose_waypoints

    assert normalized.shape == proposed_delta.shape == (6,)
    scale = POSE_SCALE.to(device=normalized.device, dtype=normalized.dtype)
    old = (normalized * scale).detach().cpu().numpy()
    requested = ((normalized + proposed_delta) * scale).detach().cpu().numpy()
    if max_rotation_deg is None and max_translation_m is None:
        from scipy.spatial.transform import Rotation

        angle = float(
            (
                Rotation.from_rotvec(requested[:3])
                * Rotation.from_rotvec(old[:3]).inv()
            ).magnitude()
        )
        assert angle < math.pi
        return normalized + proposed_delta, {
            "rotation_deg": math.degrees(angle),
            "translation_m": float(np.linalg.norm(requested[3:] - old[3:])),
        }
    assert max_rotation_deg is not None and max_translation_m is not None
    path, receipt = pose_waypoints(
        old,
        requested,
        max_rotation_deg=max_rotation_deg,
        max_translation_m=max_translation_m,
    )
    new = path[min(1, len(path) - 1)]
    result = (
        torch.as_tensor(new, dtype=normalized.dtype, device=normalized.device) / scale
    )
    increment = (
        receipt["steps"][0]
        if receipt["steps"]
        else {"rotation_deg": 0.0, "translation_m": 0.0}
    )
    assert increment["rotation_deg"] <= max_rotation_deg + 1e-10
    assert increment["translation_m"] <= max_translation_m + 1e-12
    return result, increment


def adam_update(
    moments: list[torch.Tensor],
    gradients: tuple[torch.Tensor, torch.Tensor],
    *,
    optimizer_step: int,
    learning_rate: float,
    pose_learning_rate: float,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return the unchanged Adam proposal at its explicit global step."""
    assert optimizer_step >= 1
    assert len(moments) == 4
    gq, gp = gradients
    mq, vq, mp, vp = moments
    assert mq.shape == vq.shape == gq.shape
    assert mp.shape == vp.shape == gp.shape
    proposed_moments = [
        0.9 * mq + 0.1 * gq,
        0.999 * vq + 0.001 * gq.square(),
        0.9 * mp + 0.1 * gp,
        0.999 * vp + 0.001 * gp.square(),
    ]
    next_mq, next_vq, next_mp, next_vp = proposed_moments
    dq = (
        -learning_rate
        * (next_mq / (1 - 0.9**optimizer_step))
        / ((next_vq / (1 - 0.999**optimizer_step)).sqrt() + 1e-12)
    )
    dp = (
        -pose_learning_rate
        * (next_mp / (1 - 0.9**optimizer_step))
        / ((next_vp / (1 - 0.999**optimizer_step)).sqrt() + 1e-12)
    )
    return proposed_moments, dq, dp


def validate_linear_contract(cfg: Config) -> None:
    """Reject inconsistent predictor/adjoint settings before CUDA initialization."""
    assert cfg.predictor_relative_shift == cfg.adjoint_relative_shift == 0.0
    assert cfg.predictor_rtol == cfg.adjoint_rtol == 1e-7


def preflight_parent(parent: Path) -> dict:
    """CPU-only readiness check; pending hashes prevent an accidental launch."""
    expected_parent = (GROUP / "data/mouthopen-refined-pilot-002").resolve()
    present = {name: (parent / name).is_file() for name in PARENT_SHA256}
    hashes_bound = {
        name: len(expected) == 64 and all(ch in "0123456789abcdef" for ch in expected)
        for name, expected in PARENT_SHA256.items()
    }
    hashes_match = {
        name: present[name] and hashes_bound[name] and sha256(parent / name) == expected
        for name, expected in PARENT_SHA256.items()
    }
    source_match = (
        sha256(GROUP / "src/10-fit.py") == FIT_SOURCE_SHA256
        and sha256(GROUP / "src/20-audit.py") == AUDIT_SOURCE_SHA256
        and sha256(GROUP / "src/81-run-unshifted-pilot.py") == PARENT_FIT_SOURCE_SHA256
    )
    return {
        "schema": "collision-off-strict-continuation-preflight-v1",
        "parent_run": str(parent),
        "expected_parent": str(expected_parent),
        "parent_path_exact": parent == expected_parent,
        "present": present,
        "hashes_bound": hashes_bound,
        "hashes_match": hashes_match,
        "source_match": source_match,
        "ready": parent == expected_parent
        and all(hashes_match.values())
        and source_match,
    }


def continuation_inputs(parent: Path) -> dict:
    """Bind the completed audited pilot, its exact saved state, and lineage."""
    preflight = preflight_parent(parent)
    assert preflight["ready"], preflight
    summary = json.loads((parent / "summary.json").read_text())
    protocol = json.loads((parent / "protocol.json").read_text())
    audit = json.loads((parent / "independent-audit.json").read_text())
    refinement = json.loads((parent / "equilibrium-refinement.json").read_text())
    final = summary["final"]
    assert summary["status"] in {"finite_budget_exhausted", "time_budget_exhausted"}
    assert summary["inverse_converged"] is False
    assert final["valid_forward"] is True
    assert final["optimizer_phase"] == "joint"
    assert final["local_iteration"] >= 1
    assert final["optimizer_steps"]["q"] == final["optimizer_steps"]["pose"]
    assert protocol["expression_name"] == "MouthOpen"
    assert protocol["collision_enabled"] is False
    assert protocol["force_contract"]["atol"] == 1e-12
    assert protocol["adjoint_contract"]["relative_shift"] == 0
    assert protocol["inversion_policy"] == {
        "maximum_inverted_tetrahedra": 100,
        "maximum_inverted_rest_volume_fraction": 0.0001,
        "orientation_floor": None,
        "scope": "retained mechanical tetrahedra",
    }
    assert protocol["initialization"]["baseline_refinement"] is not None
    assert (
        protocol["source_sha256"]["new-neutral/81-run-unshifted-pilot.py"]
        == PARENT_FIT_SOURCE_SHA256
    )
    assert refinement["kind"] == "equilibrium_refinement_at_unchanged_parameters"
    assert refinement["optimizer_state_preserved_exactly"] is True
    assert audit["schema"] == "collision-off-expression-independent-audit-v1"
    assert audit["valid_forward"] is True
    assert audit["expression_name"] == "MouthOpen"
    for key, name in (
        ("protocol", "protocol.json"),
        ("summary", "summary.json"),
        ("endpoint", "endpoint.npz"),
    ):
        assert audit["inputs"][key]["sha256"] == PARENT_SHA256[name]
    assert summary["endpoint"]["sha256"] == PARENT_SHA256["endpoint.npz"]
    progress = [
        json.loads(line)
        for line in (parent / "progress.jsonl").read_text().splitlines()
    ]
    assert progress[-1] == final
    assert len(progress) == final["local_iteration"] + 1
    state = torch.load(parent / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["local_iteration"] == final["local_iteration"]
    assert state["optimizer_steps"] == final["optimizer_steps"]
    assert state["optimizer_step"] == max(final["optimizer_steps"].values())
    assert len(state["moments"]) == 4
    assert all(bool(torch.isfinite(value).all()) for value in state["moments"])
    assert (
        state["moments"][0].shape
        == state["moments"][1].shape
        == state["activation_inv"].shape
    )
    assert (
        state["moments"][2].shape
        == state["moments"][3].shape
        == state["pose_normalized"].shape
    )
    assert state["convergence_reference"] is not None
    assert len(state["convergence_history_tail"]) > 0
    with np.load(parent / "endpoint.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["displacement_m"], state["displacement_m"].numpy()
        )
        np.testing.assert_array_equal(
            archive["activation_inv"], state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            archive["pose_rad_m"], state["pose_rad_m"].numpy()
        )
        assert np.isfinite(archive["displacement_m"]).all()
    next_alpha = min(1.0, 2.0 * float(final["alpha"]))
    assert 0.001 <= next_alpha <= 1.0
    return {
        "kind": "ordinary_checkpoint_continuation",
        "optimizer_updates": 0,
        "parent_run": str(parent),
        "parent_inputs": {name: record(parent / name) for name in PARENT_SHA256},
        "parent_final": final,
        "parent_optimizer_steps": dict(state["optimizer_steps"]),
        "next_trial_alpha": next_alpha,
        "stationarity_monitor": "disabled; inherited reference/history cross the earlier equilibrium refinement",
    }


def main(cfg: Config) -> None:
    assert len(cfg.pilot_source_sha256) == 64
    assert sha256(Path(__file__)) == cfg.pilot_source_sha256
    assert cfg.expression_name == "MouthOpen"
    assert cfg.initialization_checkpoint is not None
    assert cfg.continue_optimizer_state and not cfg.resume
    assert cfg.maximum_iterations == 25
    assert cfg.forward_atol == 1e-12
    validate_linear_contract(cfg)
    assert cfg.minimum_trial_alpha == 0.001
    assert cfg.wall_seconds == 3600
    assert cfg.enable_stationarity_monitor is False
    assert cfg.exclude_fully_fixed_tets
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 0.0001
    continuation_edge = continuation_inputs(
        cfg.initialization_checkpoint.resolve().parent
    )
    assert cfg.initial_trial_alpha == continuation_edge["next_trial_alpha"]
    assert (
        cfg.initialization_checkpoint.resolve()
        == (GROUP / "data/mouthopen-refined-pilot-002/checkpoint.pt").resolve()
    )
    from collision_off_runtime import install_collision_off_runtime
    from collision_off_seed import prepare_coupled_seed
    from mesh_step_scale import mean_rest_edge_length
    from mouthopen_block_optimizer import block_adam_update
    from mouthopen_convergence import gradient_metrics, stationarity_candidate

    output = cfg.output_dir.resolve()
    assert cfg.maximum_iterations >= 0
    assert cfg.q_only_iterations >= 0
    assert 0 < cfg.initial_trial_alpha <= 1
    assert 0 <= cfg.minimum_trial_alpha <= cfg.initial_trial_alpha
    assert cfg.convergence_patience >= 0
    assert not (cfg.resume and cfg.continue_optimizer_state)
    assert cfg.resume or not output.exists(), output
    output.mkdir(parents=True, exist_ok=cfg.resume)
    configure_cuda()
    assert cfg.expression_name in {"MouthOpen", "Smile"}
    assert cfg.seed_method == "collision_off_tangent"
    assert cfg.exclude_fully_fixed_tets
    assert cfg.forward_atol <= 1e-8
    assert cfg.maximum_inverted_tetrahedra <= 100
    assert cfg.maximum_inverted_rest_volume_fraction <= 1e-4

    neutral_audit = json.loads((cfg.neutral_dir / "independent-audit.json").read_text())
    for name in ("endpoint.npz", "protocol.json", "summary.json"):
        assert (
            sha256(cfg.neutral_dir / name)
            == neutral_audit["run_inputs"][name]["sha256"]
        )
    transfer = json.loads((cfg.blendshape_dir / "manifest.json").read_text())
    assert transfer["transfer_success"]
    target_path = cfg.blendshape_dir / "blendshapes.npz"
    assert sha256(target_path) == transfer["artifacts"]["blendshapes.npz"]["sha256"]

    physics, _ = build_rebased_physics(cfg.reference_dir, inverse=True)
    model = physics.runtime.forward.model
    # Display geometry remains in physics; mechanics never receives collision.
    model.collision = None
    is_fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    is_lip = np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)
    assert is_lip.sum() == 3408
    assert not np.any(is_lip & is_fixed)
    expected_fixed = np.concatenate(
        (
            np.repeat(is_fixed, 3),
            np.ones((model.dof_map.n_full - len(is_fixed) * 3), dtype=bool),
        )
    )
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu().numpy(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu().numpy(), np.flatnonzero(~expected_fixed)
    )
    baseline, strain_receipt, _ = install_active_strain(model)
    tetrahedron_policy = {"excluded_tetrahedra": 0, "method": "original complete FEM"}
    if cfg.exclude_fully_fixed_tets:
        from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra

        # All-fixed cells must contribute exactly zero to the free equations.
        # Prove this on the actual neutral, including a free Hessian product.
        with np.load(cfg.neutral_dir / "endpoint.npz", allow_pickle=False) as archive:
            probe_u = torch.zeros_like(physics.runtime.forward.state.u)
            probe_u[: len(is_fixed)] = torch.as_tensor(
                archive["displacement_m"][: len(is_fixed)],
                device=probe_u.device,
                dtype=probe_u.dtype,
            )
        model.dof_map.fixed_values = physics.boundary(
            torch.zeros(6, device=probe_u.device, dtype=probe_u.dtype)
        )
        probe_u = model.dof_map.to_full(model.dof_map.to_free(probe_u))
        generator = torch.Generator(device=probe_u.device).manual_seed(20260930)
        probe_direction = model.dof_map.to_full_grad(
            torch.randn(
                (len(model.dof_map.free_indices),),
                device=probe_u.device,
                dtype=probe_u.dtype,
                generator=generator,
            )
        )

        def probe_free_equations():
            state = model.State(u=probe_u)
            assert model.collision is None
            return (
                model.dof_map.to_free_grad(model.grad(state)).detach().clone(),
                model.dof_map.to_free_grad(model.hess_prod(state, probe_direction))
                .detach()
                .clone(),
            )

        original_probe = probe_free_equations()
        tetrahedron_policy = exclude_fully_fixed_tetrahedra(physics)
        baseline = model.get_materials()
        retained_probe = probe_free_equations()
        proof = {}
        for name, before, after in zip(
            ("free_gradient", "free_hessian_product"),
            original_probe,
            retained_probe,
            strict=True,
        ):
            difference = float(torch.linalg.vector_norm(after - before))
            scale = float(torch.linalg.vector_norm(before))
            proof[name] = {
                "absolute_error": difference,
                "relative_error": difference / max(scale, 1e-30),
            }
            torch.testing.assert_close(after, before, rtol=1e-10, atol=1e-15)
        tetrahedron_policy["neutral_free_equation_proof"] = proof
        write_json(output / "tetrahedron-policy.json", tetrahedron_policy)

    inversion_policy = {
        "maximum_inverted_tetrahedra": cfg.maximum_inverted_tetrahedra,
        "maximum_inverted_rest_volume_fraction": cfg.maximum_inverted_rest_volume_fraction,
        "orientation_floor": None,
        "scope": "retained mechanical tetrahedra",
    }
    assert cfg.maximum_inverted_tetrahedra >= 0
    assert 0 <= cfg.maximum_inverted_rest_volume_fraction <= 1

    def geometry_metrics(u: torch.Tensor) -> dict:
        if cfg.exclude_fully_fixed_tets:
            from mouthopen_tet_policy import geometry_metrics as retained_metrics

            return retained_metrics(physics, u)
        return physics.metrics(u)

    def geometry_allowed(geometry: dict) -> bool:
        return (
            geometry["inverted_tetrahedra"] <= cfg.maximum_inverted_tetrahedra
            and geometry.get("inverted_rest_volume_fraction", 0.0)
            <= cfg.maximum_inverted_rest_volume_fraction
        )

    def bounded_pose(pose: torch.Tensor, delta: torch.Tensor):
        return _bounded_pose(
            pose,
            delta,
            max_rotation_deg=cfg.max_rotation_increment_deg,
            max_translation_m=cfg.max_translation_increment_m,
        )

    with np.load(
        cfg.neutral_dir / "active-strain-fields.npz", allow_pickle=False
    ) as saved_strain:
        np.testing.assert_array_equal(
            baseline["skin"]["activation_inv"].cpu().numpy(),
            saved_strain["skin_activation_inverse"],
        )
        np.testing.assert_array_equal(
            baseline["skin"]["mu"].cpu().numpy(), saved_strain["skin_mu_mpa"]
        )
        np.testing.assert_array_equal(
            baseline["skin"]["thickness"].cpu().numpy(),
            saved_strain["skin_thickness_m"],
        )
    strain_receipt["neutral_skin_prestretch_mu_thickness_preserved_exactly"] = True
    model.set_materials(baseline)
    runtime = install_collision_off_runtime(
        physics,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        adjoint_relative_shift=cfg.adjoint_relative_shift,
        newton_max_steps=cfg.max_newton_steps,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
    )
    with np.load(target_path, allow_pickle=False) as data:
        target_index = list(data["expression_names"]).index(cfg.expression_name)
        skin_ids = data["skin_global_ids"].copy()
        tri = data["skin_triangles"].copy()
        neutral_points = data["new_neutral_points_m"].copy()
        target_points = data["target_points_m"][target_index].copy()
    with np.load(cfg.neutral_dir / "endpoint.npz") as data:
        neutral_u = torch.as_tensor(data["displacement_m"].copy())
    np.testing.assert_array_equal(
        physics.points[skin_ids] + neutral_u.cpu().numpy()[skin_ids], neutral_points
    )
    ids = torch.as_tensor(skin_ids, dtype=torch.int64)
    active_ids = physics.base.active_t
    active_source_ids = (
        physics.base.retained_active_cell_ids
        if cfg.exclude_fully_fixed_tets
        else active_ids.cpu().numpy()
    )
    if cfg.initialization_checkpoint is None:
        initialization_dir = output / "neutral-initialization"
        initialization_dir.mkdir(exist_ok=cfg.resume)
        initialization_checkpoint = initialization_dir / "endpoint.pt"
        zero_pose = torch.zeros(6, dtype=neutral_u.dtype, device="cuda")
        full_seed = torch.zeros_like(runtime.forward.state.u)
        full_seed[: len(neutral_u)] = neutral_u.to(device="cuda")
        model.dof_map.fixed_values = physics.boundary(zero_pose)
        full_seed = model.dof_map.to_full(model.dof_map.to_free(full_seed))
        initial_activation = (
            baseline["muscle"]["activation_inv"][active_ids].detach().clone()
        )
        if not cfg.resume:
            save_torch(
                initialization_checkpoint,
                {
                    "activation_inv": initial_activation.cpu(),
                    "pose_rad_m": zero_pose.cpu(),
                    "displacement_m": full_seed.cpu(),
                },
            )
            save_npz(
                initialization_checkpoint.with_suffix(".npz"),
                activation_inv=initial_activation.cpu().numpy(),
                pose_rad_m=zero_pose.cpu().numpy(),
                displacement_m=full_seed.cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
        initialization_method = "saved corrected collision-on neutral as seed; fully re-equilibrated with collision disabled before any fit"
    else:
        initialization_checkpoint = cfg.initialization_checkpoint.resolve()
        initialization_method = "saved collision-off checkpoint; force and retained tetrahedra revalidated before optimization"
    initialization_endpoint = initialization_checkpoint.with_name("endpoint.npz")
    assert initialization_checkpoint.is_file()
    assert initialization_endpoint.is_file()
    with np.load(initialization_endpoint, allow_pickle=False) as data:
        initialization_active_ids = np.asarray(data["active_cell_ids"])
    np.testing.assert_array_equal(initialization_active_ids, active_source_ids)
    target = torch.as_tensor(target_points - physics.points[skin_ids])
    delta = torch.as_tensor(target_points - neutral_points)
    xyz = neutral_points[tri]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
    weights /= weights.sum()
    weights_t = torch.as_tensor(weights)
    scale2 = (weights_t[:, None] * delta.square()).sum()
    assert float(scale2) > 0

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return result

    def physical_pose(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == (6,)
        return value * POSE_SCALE.to(device=value.device, dtype=value.dtype)

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (weights_t[:, None] * (u[ids] - target).square()).sum() / scale2

    protocol = None
    if cfg.resume:
        protocol = json.loads((output / "protocol.json").read_text())
        assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
        assert protocol["sources"]["blendshapes"] == record(target_path)
        assert protocol["sources"]["neutral_endpoint"] == record(
            cfg.neutral_dir / "endpoint.npz"
        )
        assert protocol["sources"]["reference_repair"] == record(
            cfg.reference_dir / "reference-clearance.npz"
        )
        assert protocol["initialization"]["checkpoint"] == record(
            initialization_checkpoint
        )
        assert protocol["initialization"]["endpoint"] == record(initialization_endpoint)

    seed_attempt = len(list((output / "initialization").glob("attempt-*")))

    def evaluate(
        value: torch.Tensor,
        pose_value: torch.Tensor,
        initial: torch.Tensor,
        gradient: bool,
        *,
        source_q: torch.Tensor | None = None,
        source_pose: torch.Tensor | None = None,
    ) -> dict:
        nonlocal seed_attempt
        if source_q is not None:
            assert source_pose is not None
            seed_dir = output / "initialization" / f"attempt-{seed_attempt:05d}"
            seed_attempt += 1
            try:
                initial, seed_receipt = prepare_coupled_seed(
                    physics,
                    materials,
                    source_q,
                    value,
                    physical_pose(source_pose),
                    physical_pose(pose_value),
                    initial,
                    seed_dir,
                    forward_atol=cfg.forward_atol,
                    max_newton_steps=cfg.max_newton_steps,
                    off_wall_seconds=cfg.initializer_wall_seconds,
                    no_contact_linear_max_steps=cfg.no_contact_linear_max_steps,
                    deadline=deadline,
                    predictor_relative_shift=cfg.predictor_relative_shift,
                    predictor_rtol=cfg.predictor_rtol,
                )
            except ForwardConvergenceError as error:
                summary = seed_dir / "summary.json"
                row = {
                    "success": False,
                    "output_dir": str(seed_dir.resolve()),
                    "summary": record(summary) if summary.is_file() else None,
                    "error_receipt": error.receipt,
                }
                append(output / "predictors.jsonl", row)
                error.seed_dir = str(seed_dir.resolve())
                raise
            seed_receipt["output_dir"] = str(seed_dir.resolve())
            append(output / "predictors.jsonl", seed_receipt)
            seed_geometry = geometry_metrics(initial[: len(physics.points)])
            if not geometry_allowed(seed_geometry):
                message = "predictor exceeds declared inversion allowance"
                raise ForwardConvergenceError(message, receipt=seed_geometry)
        value = value.detach().requires_grad_(gradient)
        pose_value = pose_value.detach().requires_grad_(gradient)
        u = runtime.solve(
            materials(value),
            physics.boundary(physical_pose(pose_value)),
            initial.detach(),
            key=cfg.expression_name,
        )
        state_for_force = model.State(u=u.detach().clone())
        assert state_for_force.collision is None
        torch.testing.assert_close(
            state_for_force.u.flatten()[model.dof_map.fixed_indices],
            model.dof_map.fixed_values,
            rtol=0,
            atol=1e-14,
        )
        free_force = model.dof_map.to_free_grad(model.grad(state_for_force))
        direct_force = float(torch.linalg.vector_norm(free_force))
        if not math.isfinite(direct_force) or direct_force > cfg.forward_atol:
            message = "refined pilot direct free-force gate failed"
            raise ForwardConvergenceError(
                message,
                receipt={
                    "direct_free_force": direct_force,
                    "threshold": cfg.forward_atol,
                },
            )
        geometry = geometry_metrics(u[: len(physics.points)])
        if not geometry_allowed(geometry):
            message = (
                f"{cfg.expression_name} candidate exceeds declared inversion allowance"
            )
            raise ForwardConvergenceError(
                message,
                receipt={
                    "geometry": geometry,
                    "forward": copy.deepcopy(runtime.last_forward),
                },
            )
        loss = objective(u)
        forward = copy.deepcopy(runtime.last_forward)
        grads = torch.autograd.grad(loss, (value, pose_value)) if gradient else None
        if gradient:
            sparse = runtime.last_sparse_adjoint
            assert cfg.adjoint_relative_shift == 0
            assert sparse["relative_shift"] == 0
            assert sparse["shifted_relative_residual"] <= cfg.adjoint_rtol, sparse
            assert sparse["native_shifted_relative_residual"] <= cfg.adjoint_rtol, (
                sparse
            )
            assert sparse["original_unshifted_relative_residual"] <= cfg.adjoint_rtol, (
                sparse
            )
            runtime.last_adjoint["sparse_solver"] = copy.deepcopy(
                runtime.last_sparse_adjoint
            )
        return {
            "q": value.detach(),
            "pose": pose_value.detach(),
            "u": u.detach(),
            "loss": float(loss),
            "grads": grads,
            "forward": forward,
            "direct_free_force": direct_force,
            "adjoint": copy.deepcopy(runtime.last_adjoint) if gradient else None,
        }

    def metric(
        candidate: dict,
        local_iteration: int,
        optimizer_steps: dict[str, int],
        optimizer_phase: str,
        elapsed: float,
        **extra: object,
    ) -> dict:
        assert set(optimizer_steps) == {"q", "pose"}
        assert optimizer_phase in {"initial", "q_only", "joint"}
        u = candidate["u"][: len(physics.points)]
        geometry = geometry_metrics(u)
        forward = candidate["forward"]
        pose_m = physical_pose(candidate["pose"]).detach().cpu()
        return {
            "iteration": local_iteration,
            "local_iteration": local_iteration,
            "optimizer_step": max(optimizer_steps.values()),
            "optimizer_steps": dict(optimizer_steps),
            "optimizer_phase": optimizer_phase,
            "loss": candidate["loss"],
            "fit_rms_mm": math.sqrt(candidate["loss"] * float(scale2)) * 1000,
            "pose_rad_m": pose_m.tolist(),
            "pose_rotation_degrees": float(
                torch.linalg.vector_norm(pose_m[:3]) * 180 / math.pi
            ),
            "pose_translation_mm": float(torch.linalg.vector_norm(pose_m[3:]) * 1000),
            "force_norm_n": candidate["direct_free_force"] * 1e6,
            "force_threshold_n": cfg.forward_atol * 1e6,
            "forward_converged": forward["success"]
            and candidate["direct_free_force"] <= cfg.forward_atol,
            "collision_enabled": False,
            "contact_valid": None,
            "geometry": geometry,
            "inversion_free": geometry["inverted_tetrahedra"] == 0,
            "geometry_within_declared_allowance": geometry_allowed(geometry),
            "valid_forward": geometry_allowed(geometry)
            and forward["success"]
            and candidate["direct_free_force"] <= cfg.forward_atol
            and model.collision is None,
            "inverse_converged": False,
            "gradient": gradient_metrics(candidate["grads"]),
            "adjoint_relative_shift": cfg.adjoint_relative_shift,
            "adjoint_gradient_type": (
                "damped_approximation"
                if cfg.adjoint_relative_shift > 0
                else "unshifted_implicit"
            ),
            "elapsed_seconds": elapsed,
            "activation_rms": float(candidate["q"].square().mean().sqrt()),
            "activation_max_abs": float(candidate["q"].abs().max()),
            **extra,
        }

    moments: list[torch.Tensor]
    history: list[dict] = []
    start_iteration = 0
    optimizer_steps = {"q": 0, "pose": 0}
    optimizer_state_source = "fresh_zero_moments"
    if cfg.resume:
        state = torch.load(output / "checkpoint.pt", weights_only=False)
        q = state["activation_inv"].to(device="cuda")
        pose_z = state["pose_normalized"].to(device="cuda")
        seed = state["displacement_m"].to(device="cuda")
        moments = [value.to(device="cuda") for value in state["moments"]]
        assert len(moments) == 4
        start_iteration = int(state.get("local_iteration", state["iteration"]))
        legacy_step = int(state.get("optimizer_step", state["iteration"]))
        optimizer_steps = dict(
            state.get("optimizer_steps", {"q": legacy_step, "pose": legacy_step})
        )
        assert set(optimizer_steps) == {"q", "pose"}
        assert all(
            isinstance(value, int) and value >= 0 for value in optimizer_steps.values()
        )
        optimizer_state_source = "same_run_checkpoint"
        history = [
            json.loads(line)
            for line in (output / "progress.jsonl").read_text().splitlines()
        ]
    else:
        state = torch.load(
            initialization_checkpoint, map_location="cpu", weights_only=False
        )
        assert {"activation_inv", "pose_rad_m", "displacement_m"} <= state.keys()
        q = state["activation_inv"].to(device="cuda")
        initial_pose = state["pose_rad_m"].to(device="cuda")
        seed = state["displacement_m"].to(device="cuda")
        assert q.shape == (len(active_ids), 6) and initial_pose.shape == (6,)
        assert seed.shape == physics.full_skull.full_reference_points_m.shape
        assert bool(torch.isfinite(q).all())
        assert bool(torch.isfinite(initial_pose).all())
        assert bool(torch.isfinite(seed).all())
        with np.load(initialization_endpoint, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["activation_inv"], q.cpu().numpy())
            np.testing.assert_array_equal(
                data["pose_rad_m"], initial_pose.cpu().numpy()
            )
            np.testing.assert_array_equal(data["displacement_m"], seed.cpu().numpy())
        pose_z = initial_pose / POSE_SCALE.to(device="cuda", dtype=initial_pose.dtype)
        if cfg.continue_optimizer_state:
            # Preserve the actual optimized coordinates: divide/multiply
            # roundoff can change the sign of an almost singular boundary tet.
            assert "pose_normalized" in state
            pose_z = state["pose_normalized"].to(device="cuda")
            torch.testing.assert_close(
                physical_pose(pose_z), initial_pose, rtol=0, atol=0
            )
            assert "moments" in state
            assert "iteration" in state
            moments = [value.to(device="cuda") for value in state["moments"]]
            assert len(moments) == 4
            legacy_step = int(state.get("optimizer_step", state["iteration"]))
            optimizer_steps = dict(
                state.get("optimizer_steps", {"q": legacy_step, "pose": legacy_step})
            )
            assert set(optimizer_steps) == {"q", "pose"}
            assert all(
                isinstance(value, int) and value >= 0
                for value in optimizer_steps.values()
            )
            optimizer_state_source = (
                "initialization_checkpoint_optimizer_step"
                if "optimizer_step" in state
                else "initialization_checkpoint_legacy_iteration"
            )
        else:
            moments = [
                torch.zeros_like(q),
                torch.zeros_like(q),
                torch.zeros_like(pose_z),
                torch.zeros_like(pose_z),
            ]
        assert all(bool(torch.isfinite(value).all()) for value in moments)
        assert moments[0].shape == moments[1].shape == q.shape
        assert moments[2].shape == moments[3].shape == pose_z.shape

    monitor_history = (
        list(state.get("convergence_history_tail", []))
        if (cfg.resume or cfg.continue_optimizer_state)
        else []
    )
    convergence_reference = (
        state.get("convergence_reference")
        if (cfg.resume or cfg.continue_optimizer_state)
        else None
    )

    def checkpoint(candidate: dict, row: dict, optimizer_steps: dict[str, int]) -> None:
        pose_m = physical_pose(candidate["pose"])
        save_npz(
            output / "endpoint.npz",
            displacement_m=candidate["u"].cpu().numpy(),
            activation_inv=candidate["q"].cpu().numpy(),
            active_cell_ids=active_source_ids,
            pose_rad_m=pose_m.cpu().numpy(),
        )
        save_torch(
            output / "checkpoint.pt",
            {
                "iteration": row["iteration"],
                "local_iteration": row["local_iteration"],
                "optimizer_step": max(optimizer_steps.values()),
                "optimizer_steps": dict(optimizer_steps),
                "activation_inv": candidate["q"].cpu(),
                "pose_normalized": candidate["pose"].cpu(),
                "pose_rad_m": pose_m.cpu(),
                "displacement_m": candidate["u"].cpu(),
                "moments": [value.cpu() for value in moments],
                "convergence_reference": convergence_reference,
                "convergence_history_tail": monitor_history,
            },
        )
        write_json(
            output / "summary.json",
            {
                "status": "running",
                "initial": history[0],
                "final": row,
                "inverse_converged": False,
                "endpoint": record(output / "endpoint.npz"),
            },
        )

    if not cfg.resume:
        rendering = output / "rendering.npz"
        geometry = physics.full_skull.geometry
        save_npz(
            rendering,
            full_reference_points_m=physics.full_skull.full_reference_points_m,
            skin_global_ids=skin_ids,
            skin_triangles=tri,
            cranium_global_ids=geometry.cranium_global_ids,
            cranium_triangles=geometry.cranium_faces,
            mandible_global_ids=geometry.mandible_global_ids,
            mandible_triangles=geometry.mandible_faces,
            eye_global_ids=physics.full_skull.eye_global_ids,
            eye_triangles=physics.eyes.triangles,
        )
        source_records = {}
        source_dir = output / "sources"
        for directory, label in (
            (GROUP / "src", "new-neutral"),
            (SOLVERS, "solver-performance"),
            (JOINT, "joint"),
            (ROOT / "src/liblaf/apple", "apple"),
        ):
            shutil.copytree(
                directory,
                source_dir / label,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for path in source_dir.rglob("*.py"):
            source_records[str(path.relative_to(source_dir))] = sha256(path)
        write_json(
            output / "protocol.json",
            {
                "schema": "corrected-neutral-collision-off-rigid6-inverse-v1",
                "config": cfg.model_dump(mode="json"),
                "expression_name": cfg.expression_name,
                "expression_index": target_index,
                "initialization": {
                    "method": initialization_method,
                    "checkpoint": record(initialization_checkpoint),
                    "endpoint": record(initialization_endpoint),
                    "optimizer_state": {
                        "continued": cfg.continue_optimizer_state,
                        "source": optimizer_state_source,
                        "initial_steps": dict(optimizer_steps),
                    },
                    "continuation_edge": continuation_edge,
                },
                "sources": {
                    "blendshapes": record(target_path),
                    "blendshape_manifest": record(cfg.blendshape_dir / "manifest.json"),
                    "neutral_endpoint": record(cfg.neutral_dir / "endpoint.npz"),
                    "neutral_summary": record(cfg.neutral_dir / "summary.json"),
                    "reference_repair": record(
                        cfg.reference_dir / "reference-clearance.npz"
                    ),
                },
                "rendering": {
                    "archive": record(rendering),
                    "layout": "Full runtime coordinates; each mesh uses named global IDs and local zero-based triangles.",
                },
                "parameterization": {
                    "activation": "B = I + symmetric(q); six independent dimensionless components per active muscle tetrahedron; no projection, clipping, positivity, rank, magnitude or smoothness penalty",
                    "active_cells": len(active_ids),
                    "activation_dofs": q.numel(),
                    "jaw_dofs": 6,
                    "jaw": "world rotation-vector radians and world translation metres about existing mandible pivot; no total-pose bound",
                    "normalization_scales": [math.pi / 18.0] * 3 + [0.01] * 3,
                    "adjacent_proposed_increment": {
                        "rotation_degrees": cfg.max_rotation_increment_deg,
                        "translation_m": cfg.max_translation_increment_m,
                        "rotation_interpolation": "SO(3) geodesic",
                    },
                    "jaw_pivot_m": physics.pivot_t.cpu().tolist(),
                    "packing": "xx, yy, zz, xy, yz, xz",
                },
                "optimizer_policy": {
                    "q_only_accepted_iterations": cfg.q_only_iterations,
                    "q_only": "updates q Adam state only; pose moments and counter remain frozen",
                    "joint": "updates q and pose Adam blocks with separate bias-correction counters",
                    "stationarity_monitor": "disabled for this bounded continuation because inherited history spans the earlier equilibrium refinement; independent post-run convergence audit required",
                },
                "objective": "authoritative loaded-neutral skin-area-weighted squared position error divided by weighted squared target motion; no regularization",
                "materials": strain_receipt,
                "tetrahedron_policy": tetrahedron_policy,
                "inversion_policy": inversion_policy,
                "boundary_policy": "original FEM IsFixed only; rigid obstacle nodes prescribed; runtime fixed and free DOF arrays verified",
                "seed_method": cfg.seed_method,
                "collision_enabled": False,
                "ipc_policy": "collision=None in forward energy, residual, Hessian, adjoint and admission; CCD disabled",
                "coordinate_contract": "X is the repaired constitutive reference; the exact audited pilot checkpoint is the ordinary continuation state",
                "force_contract": {
                    "rtol": 0.0,
                    "atol": cfg.forward_atol,
                    "effective_threshold": cfg.forward_atol,
                    "relative_anchor": "none; each trial uses the fixed absolute gate",
                },
                "adjoint_contract": {
                    "relative_shift": cfg.adjoint_relative_shift,
                    "shift_definition": "lambda = relative_shift * mean(abs(diag(H_ff)))",
                    "system": "(H_ff + lambda I) p = -L_f",
                    "approximate_gradient": cfg.adjoint_relative_shift > 0,
                    "relative_tolerance": cfg.adjoint_rtol,
                    "physical_forward_unchanged": True,
                },
                "source_sha256": source_records,
                "commit_enabled": False,
            },
        )
    else:
        assert protocol is not None

    assert cfg.wall_seconds is None or cfg.wall_seconds > 0
    started = time.perf_counter()
    deadline = None if cfg.wall_seconds is None else started + cfg.wall_seconds
    runtime.deadline = deadline
    from mouthopen_gradient_check import check_joint_pullback

    try:
        replayed = evaluate(q, pose_z, seed, False)
        np.testing.assert_array_equal(
            replayed["q"].cpu().numpy(), state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            replayed["pose"].cpu().numpy(), state["pose_normalized"].numpy()
        )
        replay_change = float((replayed["u"] - seed).abs().max())
        replay_geometry = geometry_metrics(replayed["u"][: len(physics.points)])
        replay_receipt = {
            "schema": "collision-off-strict-continuation-replay-v1",
            "parent_checkpoint": record(initialization_checkpoint),
            "parent_endpoint": record(initialization_endpoint),
            "parameters_unchanged": True,
            "optimizer_updates": 0,
            "maximum_displacement_change_m": replay_change,
            "loss": replayed["loss"],
            "direct_free_force": replayed["direct_free_force"],
            "geometry": replay_geometry,
            "saved_parent_loss": continuation_edge["parent_final"]["loss"],
            "saved_parent_force_n": continuation_edge["parent_final"]["force_norm_n"],
            "saved_parent_geometry": continuation_edge["parent_final"]["geometry"],
        }
        if not torch.equal(replayed["u"], seed):
            save_npz(
                output / "replayed-moved-state.npz",
                displacement_m=replayed["u"].cpu().numpy(),
                activation_inv=replayed["q"].cpu().numpy(),
                pose_normalized=replayed["pose"].cpu().numpy(),
                pose_rad_m=physical_pose(replayed["pose"]).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            replay_receipt["status"] = "unresolved_replay_changed"
            replay_receipt["moved_state"] = record(output / "replayed-moved-state.npz")
            write_json(output / "continuation-replay.json", replay_receipt)
            write_json(
                output / "summary.json",
                {
                    "status": "unresolved_replay_changed",
                    "inverse_converged": False,
                    "optimizer_updates": 0,
                    "continuation_replay": record(output / "continuation-replay.json"),
                },
            )
            cherries.log_output(output / "summary.json")
            return
        np.testing.assert_allclose(
            replayed["loss"],
            continuation_edge["parent_final"]["loss"],
            rtol=1e-10,
            atol=1e-12,
        )
        for key, value in replay_geometry.items():
            np.testing.assert_allclose(
                value,
                continuation_edge["parent_final"]["geometry"][key],
                rtol=1e-10,
                atol=1e-12,
            )
        current = evaluate(q, pose_z, seed, True)
        np.testing.assert_array_equal(
            current["q"].cpu().numpy(), state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            current["pose"].cpu().numpy(), state["pose_normalized"].numpy()
        )
        if not torch.equal(current["u"], seed):
            save_npz(
                output / "gradient-replayed-moved-state.npz",
                displacement_m=current["u"].cpu().numpy(),
                activation_inv=current["q"].cpu().numpy(),
                pose_normalized=current["pose"].cpu().numpy(),
                pose_rad_m=physical_pose(current["pose"]).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            replay_receipt["status"] = "unresolved_gradient_replay_changed"
            replay_receipt["gradient_moved_state"] = record(
                output / "gradient-replayed-moved-state.npz"
            )
            replay_receipt["maximum_gradient_replay_displacement_change_m"] = float(
                (current["u"] - seed).abs().max()
            )
            write_json(output / "continuation-replay.json", replay_receipt)
            write_json(
                output / "summary.json",
                {
                    "status": "unresolved_gradient_replay_changed",
                    "inverse_converged": False,
                    "optimizer_updates": 0,
                    "continuation_replay": record(output / "continuation-replay.json"),
                },
            )
            cherries.log_output(output / "summary.json")
            return
        replay_receipt["status"] = "identical_saved_state"
        replay_receipt["unshifted_adjoint"] = copy.deepcopy(runtime.last_sparse_adjoint)
        replay_receipt["optimizer_steps_unchanged"] = dict(optimizer_steps)
        replay_receipt["normalized_pose_exact"] = True
        write_json(output / "continuation-replay.json", replay_receipt)
        if cfg.seed_method == "collision_off_tangent":
            identity_seed, identity_receipt = prepare_coupled_seed(
                physics,
                materials,
                current["q"],
                current["q"],
                physical_pose(current["pose"]),
                physical_pose(current["pose"]),
                current["u"],
                output
                / ("resume-zero-update-check" if cfg.resume else "zero-update-check"),
                deadline=deadline,
                predictor_relative_shift=cfg.predictor_relative_shift,
                predictor_rtol=cfg.predictor_rtol,
            )
            torch.testing.assert_close(identity_seed, current["u"], rtol=0, atol=0)
            assert identity_receipt["predictor"]["rhs_norm"] == 0
            assert identity_receipt["predictor"]["maximum_free_displacement_m"] == 0
        if not cfg.resume and cfg.initialization_checkpoint is None:
            save_npz(
                output / "collision-off-start.npz",
                displacement_m=current["u"].cpu().numpy(),
                activation_inv=current["q"].cpu().numpy(),
                pose_normalized=current["pose"].cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            write_json(
                output / "collision-off-start.json",
                {
                    "collision_enabled": False,
                    "source_neutral": record(cfg.neutral_dir / "endpoint.npz"),
                    "re_equilibrated": True,
                    "forward": current["forward"],
                    "geometry": geometry_metrics(current["u"][: len(physics.points)]),
                    "maximum_change_from_saved_neutral_m": float(
                        (current["u"] - seed).abs().max()
                    ),
                    "target_geometry_preserved": True,
                },
            )
        check = check_joint_pullback(
            physics,
            runtime,
            materials,
            physical_pose,
            current["q"],
            current["pose"],
            current["u"],
            grads=current["grads"],
            objective=objective,
            key=cfg.expression_name,
        )
        write_json(
            output
            / ("resume-gradient-check.json" if cfg.resume else "gradient-check.json"),
            check,
        )
        assert min(row["relative_error"] for row in check["q"].values()) < 1e-3, check[
            "q"
        ]
        for coordinate in check["jaw_coordinates"].values():
            assert (
                min(
                    coordinate[f"epsilon_{epsilon:.0e}"]["relative_error"]
                    for epsilon in (1e-4, 1e-5, 1e-6)
                )
                < 1e-3
            ), coordinate
    except Exception as error:
        failed_state = None
        if runtime.last_problem is not None and hasattr(
            runtime, "last_failed_displacement"
        ):
            failed = runtime.last_failed_displacement
            save_npz(
                output / "initial-replay-failed-state.npz",
                displacement_m=failed.cpu().numpy(),
                activation_inv=q.cpu().numpy(),
                pose_normalized=pose_z.cpu().numpy(),
                pose_rad_m=physical_pose(pose_z).cpu().numpy(),
                active_cell_ids=active_source_ids,
            )
            failed_state = record(output / "initial-replay-failed-state.npz")
        write_json(
            output / "summary.json",
            {
                "status": "unresolved_initial_replay",
                "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
                "inverse_converged": False,
                "optimizer_updates": 0,
                "failed_state": failed_state,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "receipt": getattr(error, "receipt", None),
                },
            },
        )
        cherries.log_output(output / "summary.json")
        raise
    if not cfg.resume:
        row = metric(
            current,
            0,
            optimizer_steps,
            "initial",
            time.perf_counter() - started,
        )
        history.append(row)
        if convergence_reference is None:
            convergence_reference = row["gradient"]
        if not monitor_history:
            monitor_history.append(row)
        append(output / "progress.jsonl", row)
        checkpoint(current, row, optimizer_steps)
        shutil.copy2(output / "checkpoint.pt", output / "initial-checkpoint.pt")
        shutil.copy2(output / "endpoint.npz", output / "initial-endpoint.npz")
        initial_state = torch.load(
            output / "initial-checkpoint.pt", map_location="cpu", weights_only=False
        )
        for key in ("activation_inv", "pose_normalized", "pose_rad_m"):
            torch.testing.assert_close(initial_state[key], state[key], rtol=0, atol=0)
        for kept, original in zip(
            initial_state["moments"], state["moments"], strict=True
        ):
            torch.testing.assert_close(kept, original, rtol=0, atol=0)
        assert initial_state["optimizer_steps"] == state["optimizer_steps"]
        assert initial_state["optimizer_step"] == state["optimizer_step"]
        assert initial_state["convergence_reference"] == state["convergence_reference"]
        assert (
            initial_state["convergence_history_tail"]
            == state["convergence_history_tail"]
        )
        np.testing.assert_array_equal(
            initial_state["displacement_m"].numpy(), state["displacement_m"].numpy()
        )
        replay_receipt["saved_initial_checkpoint"] = record(
            output / "initial-checkpoint.pt"
        )
        replay_receipt["saved_initial_endpoint"] = record(
            output / "initial-endpoint.npz"
        )
        replay_receipt["optimizer_state_preserved_exactly"] = True
        write_json(output / "continuation-replay.json", replay_receipt)
        write_json(output / "initial-adjoint.json", current["adjoint"])

    status = "finite_budget_exhausted"
    trial_alpha_start = cfg.initial_trial_alpha
    if cfg.adaptive_trial_alpha and cfg.resume and "alpha" in history[-1]:
        trial_alpha_start = min(1.0, 2 * history[-1]["alpha"])
    previous_phase = history[-1].get("optimizer_phase", "joint")
    for iteration in range(start_iteration + 1, cfg.maximum_iterations + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            status = "time_budget_exhausted"
            break
        pose_m = physical_pose(current["pose"])
        LOG.info(
            "Iteration %d start: loss %.7g, rotation %.5f deg, translation %.5f mm",
            iteration,
            current["loss"],
            float(torch.linalg.vector_norm(pose_m[:3]) * 180 / math.pi),
            float(torch.linalg.vector_norm(pose_m[3:]) * 1000),
        )
        optimizer_phase = "q_only" if iteration <= cfg.q_only_iterations else "joint"
        if previous_phase == "q_only" and optimizer_phase == "joint":
            trial_alpha_start = cfg.initial_trial_alpha
        gq, gp = current["grads"]
        proposed_moments, next_optimizer_steps, dq, dp = block_adam_update(
            moments,
            (gq, gp),
            q_optimizer_step=optimizer_steps["q"],
            pose_optimizer_step=optimizer_steps["pose"],
            update_q=True,
            update_pose=optimizer_phase == "joint",
            learning_rate=cfg.learning_rate,
            pose_learning_rate=cfg.pose_learning_rate,
        )
        if optimizer_phase == "q_only":
            proposed_pose = current["pose"]
            increment = {"rotation_deg": 0.0, "translation_m": 0.0}
        else:
            proposed_pose, increment = bounded_pose(current["pose"], dp)
        dp = proposed_pose - current["pose"]
        directional = float((gq * dq).sum() + (gp * dp).sum())
        if directional >= 0:
            dq = -cfg.learning_rate * gq / (gq.abs() + 1e-12)
            if optimizer_phase == "joint":
                proposed_pose, increment = bounded_pose(
                    current["pose"], -cfg.pose_learning_rate * gp.sign()
                )
            else:
                proposed_pose = current["pose"]
                increment = {"rotation_deg": 0.0, "translation_m": 0.0}
            dp = proposed_pose - current["pose"]
            directional = float((gq * dq).sum() + (gp * dp).sum())
        assert directional < 0
        projection_receipt = None
        if (
            cfg.project_pose_at_inversion_limit
            and optimizer_phase == "joint"
            and history[-1]["geometry"]["inverted_tetrahedra"]
            >= max(1, cfg.maximum_inverted_tetrahedra - 5)
        ):
            from mouthopen_constrained_direction import project_coupled_pose

            try:
                dp, projection_receipt = project_coupled_pose(
                    physics,
                    materials,
                    physical_pose,
                    current["q"],
                    current["q"] + dq,
                    current["pose"],
                    dp,
                    current["u"],
                    output / "pose-projections" / f"{iteration:05d}",
                    activation_threshold=cfg.projection_activation_threshold,
                    margin=cfg.projection_determinant_margin,
                    epsilon=cfg.projection_probe_epsilon,
                    deadline=deadline,
                )
            except Exception as error:
                summary = json.loads((output / "summary.json").read_text())
                summary["status"] = "pose_projection_failed"
                summary["failure"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "iteration": iteration,
                }
                write_json(output / "summary.json", summary)
                raise
            proposed_pose, increment = bounded_pose(current["pose"], dp)
            dp = proposed_pose - current["pose"]
            directional = float((gq * dq).sum() + (gp * dp).sum())
            if directional >= 0:
                status = "projected_direction_not_descent"
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "success": False,
                        "failure": status,
                        "directional": directional,
                    },
                )
                break
        accepted = None
        for trial in range(cfg.max_backtracks + 1):
            alpha = trial_alpha_start * 0.5**trial
            if alpha < cfg.minimum_trial_alpha:
                status = "line_search_below_declared_resolution"
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": status,
                        "minimum_trial_alpha": cfg.minimum_trial_alpha,
                    },
                )
                break
            if deadline is not None and time.perf_counter() >= deadline:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": "declared forward wall budget exhausted before trial",
                    },
                )
                status = "time_budget_exhausted"
                break
            if optimizer_phase == "joint":
                trial_pose, trial_increment = bounded_pose(current["pose"], alpha * dp)
            else:
                trial_pose = current["pose"]
                trial_increment = {"rotation_deg": 0.0, "translation_m": 0.0}
            trial_q = current["q"] + alpha * dq
            trial_directional = float(
                (gq * (trial_q - current["q"])).sum()
                + (gp * (trial_pose - current["pose"])).sum()
            )
            if trial_directional >= 0:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": "non_descent_capped_trial",
                        "directional": trial_directional,
                        "proposed_increment": trial_increment,
                    },
                )
                continue
            try:
                candidate = evaluate(
                    trial_q,
                    trial_pose,
                    current["u"],
                    False,
                    source_q=current["q"],
                    source_pose=current["pose"],
                )
            except ForwardConvergenceError as error:
                append(
                    output / "trials.jsonl",
                    {
                        "iteration": iteration,
                        "trial": trial,
                        "alpha": alpha,
                        "success": False,
                        "failure": str(error),
                        "receipt": error.receipt,
                        "initializer_output_dir": getattr(error, "seed_dir", None),
                    },
                )
                if deadline is not None and time.perf_counter() >= deadline:
                    status = "time_budget_exhausted"
                    break
                continue
            passed = candidate["loss"] <= current["loss"] + 1e-4 * trial_directional
            append(
                output / "trials.jsonl",
                {
                    "iteration": iteration,
                    "trial": trial,
                    "alpha": alpha,
                    "success": True,
                    "accepted": passed,
                    "loss": candidate["loss"],
                    "forward": candidate["forward"],
                    "proposed_increment": trial_increment,
                    "directional": trial_directional,
                },
            )
            if passed:
                try:
                    accepted = evaluate(
                        candidate["q"], candidate["pose"], candidate["u"], True
                    )
                except ForwardConvergenceError as error:
                    append(
                        output / "trials.jsonl",
                        {
                            "iteration": iteration,
                            "trial": trial,
                            "alpha": alpha,
                            "success": False,
                            "failure": str(error),
                            "receipt": error.receipt,
                            "phase": "accepted_gradient_re_evaluation",
                        },
                    )
                    if deadline is not None and time.perf_counter() >= deadline:
                        status = "time_budget_exhausted"
                        break
                    raise
                except Exception as error:
                    summary = json.loads((output / "summary.json").read_text())
                    summary["status"] = "accepted_gradient_failed"
                    summary["failure"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "iteration": iteration,
                        "trial": trial,
                        "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
                    }
                    write_json(output / "summary.json", summary)
                    cherries.log_output(output / "summary.json")
                    raise
                break
        if status == "time_budget_exhausted":
            break
        if accepted is None:
            if status not in (
                "time_budget_exhausted",
                "line_search_below_declared_resolution",
            ):
                status = "line_search_stalled"
            break
        current = accepted
        moments = proposed_moments
        optimizer_steps = next_optimizer_steps
        row = metric(
            current,
            iteration,
            optimizer_steps,
            optimizer_phase,
            time.perf_counter() - started,
            alpha=alpha,
            line_search_backtracks=trial,
            trial_alpha_start=trial_alpha_start,
            pose_projection=(
                None
                if projection_receipt is None
                else {
                    "active_cell_count": projection_receipt["active_cell_count"],
                    "minimum_linearized_projected_J": projection_receipt[
                        "minimum_linearized_projected_J"
                    ],
                    "projection": projection_receipt["projection"],
                }
            ),
            proposed_increment=increment,
            accepted_increment=trial_increment,
            adjoint_relative_residual=current["adjoint"]["relative_residual"],
        )
        history.append(row)
        monitor_history.append(row)
        monitor_history = monitor_history[-(cfg.convergence_patience + 1) :]
        if cfg.enable_stationarity_monitor and cfg.convergence_patience:
            row["convergence_monitor"] = stationarity_candidate(
                monitor_history,
                gradient_reference=convergence_reference,
                patience=cfg.convergence_patience,
                loss_rtol=cfg.convergence_loss_rtol,
                gradient_rtol=cfg.convergence_gradient_rtol,
                gradient_atol=cfg.convergence_gradient_atol,
            )
        append(output / "progress.jsonl", row)
        checkpoint(current, row, optimizer_steps)
        cherries.set_step(iteration)
        cherries.log_metrics(
            {
                key: row[key]
                for key in (
                    "loss",
                    "fit_rms_mm",
                    "pose_rotation_degrees",
                    "pose_translation_mm",
                    "force_norm_n",
                    "activation_rms",
                )
            }
        )
        LOG.info(
            "Accepted %d: fit %.6f mm, rotation %.5f deg, translation %.5f mm, force %.6g N, inverted %d",
            iteration,
            row["fit_rms_mm"],
            row["pose_rotation_degrees"],
            row["pose_translation_mm"],
            row["force_norm_n"],
            row["geometry"]["inverted_tetrahedra"],
        )
        if cfg.adaptive_trial_alpha:
            trial_alpha_start = min(1.0, 2 * alpha)
        previous_phase = optimizer_phase
        if (
            cfg.enable_stationarity_monitor
            and cfg.convergence_patience
            and row["convergence_monitor"]["candidate_requires_independent_audit"]
        ):
            status = "stationarity_candidate_requires_audit"
            break
    summary = json.loads((output / "summary.json").read_text())
    summary["status"] = status
    summary["inverse_converged"] = False
    write_json(output / "summary.json", summary)
    cherries.log_output(output / "summary.json")
    cherries.log_output(output / "progress.jsonl")
    cherries.log_output(output / "endpoint.npz")


def gpu_compute_apps() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
    ).strip()


def drive() -> int:
    """Run the numerical worker and independent auditor in separate processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    parent = args.run_dir.resolve()
    output = args.output_dir.resolve()
    assert output.is_relative_to(GROUP / "data")
    assert output != parent
    assert not output.exists()
    driver_source = record(Path(__file__))
    edge = continuation_inputs(parent)
    assert not gpu_compute_apps(), "GPU must be idle before pilot fit"
    control = output.with_name(output.name + "-control")
    assert not control.exists()
    control.mkdir(parents=True)
    receipt_path = control / "driver.json"
    sources = {}
    for name in (Path(__file__).name, "10-fit.py", "20-audit.py"):
        origin = GROUP / "src" / name
        destination = control / "sources" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(origin, destination)
        assert sha256(destination) == sha256(origin)
        sources[name] = record(destination)
    receipt = {
        "schema": "collision-off-strict-continuation-driver-v1",
        "status": "preflight_complete",
        "parent_run": str(parent),
        "continuation_run": str(output),
        "continuation_edge": edge,
        "sources": sources,
        "executing_pilot_source": driver_source,
        "fit_exit_code": None,
        "audit_exit_code": None,
        "gpu_compute_apps_before_fit": "",
    }
    write_json(receipt_path, receipt)
    fit_command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--fit-worker",
        "--pilot-source-sha256",
        driver_source["sha256"],
        "--expression-name",
        "MouthOpen",
        "--output-dir",
        str(output),
        "--initialization-checkpoint",
        str(parent / "checkpoint.pt"),
        "--continue-optimizer-state",
        "true",
        "--maximum-iterations",
        "25",
        "--forward-atol",
        "1e-12",
        "--adjoint-relative-shift",
        "0",
        "--predictor-relative-shift",
        "0",
        "--predictor-rtol",
        "1e-7",
        "--adjoint-rtol",
        "1e-7",
        "--initial-trial-alpha",
        str(edge["next_trial_alpha"]),
        "--minimum-trial-alpha",
        "0.001",
        "--wall-seconds",
        "3600",
    ]
    env = os.environ.copy()
    env.pop("DEBUG", None)
    env.update(
        CHERRIES_NAME="MouthOpen strict collision-off continuation",
        CHERRIES_TAGS="mouthopen,collision-off,strict-continuation,isfixed,unshifted-adjoint",
        OMP_NUM_THREADS="4",
    )
    interrupted: list[int] = []

    def on_signal(signum: int, _frame: object) -> None:
        interrupted.append(signum)
        receipt["signals_received"] = interrupted
        write_json(receipt_path, receipt)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    completed = False
    try:
        assert preflight_parent(parent)["ready"]
        assert record(Path(__file__)) == driver_source
        assert sha256(GROUP / "src/10-fit.py") == FIT_SOURCE_SHA256
        assert sha256(GROUP / "src/20-audit.py") == AUDIT_SOURCE_SHA256
        receipt["sources_immediately_before_fit"] = {
            "pilot": record(Path(__file__)),
            "original_fit": record(GROUP / "src/10-fit.py"),
            "original_audit": record(GROUP / "src/20-audit.py"),
        }
        receipt["status"] = "fit_running"
        receipt["fit_command"] = fit_command
        write_json(receipt_path, receipt)
        with (control / "fit.log").open("x") as stream:
            fit = subprocess.Popen(
                fit_command,
                cwd=GROUP,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            receipt["fit_pid"] = fit.pid
            write_json(receipt_path, receipt)
            receipt["fit_exit_code"] = fit.wait()
        receipt["status"] = "fit_exited"
        write_json(receipt_path, receipt)
        assert not gpu_compute_apps(), (
            "GPU compute process remains after fit worker exit"
        )
        if (output / "summary.json").is_file():
            fit_summary = json.loads((output / "summary.json").read_text())
            receipt["fit_status"] = fit_summary["status"]
            receipt["fit_summary"] = record(output / "summary.json")
        else:
            fit_summary = {}
        if (output / "endpoint.npz").is_file() and "final" in fit_summary:
            assert record(Path(__file__)) == driver_source
            assert sha256(GROUP / "src/20-audit.py") == AUDIT_SOURCE_SHA256
            receipt["sources_immediately_before_audit"] = {
                "pilot": record(Path(__file__)),
                "original_audit": record(GROUP / "src/20-audit.py"),
            }
            audit_command = [
                sys.executable,
                "-u",
                str(GROUP / "src/20-audit.py"),
                "--run-dir",
                str(output),
            ]
            audit_env = dict(env)
            audit_env["CHERRIES_NAME"] = (
                "MouthOpen strict continuation independent audit"
            )
            receipt["status"] = "audit_running"
            receipt["audit_command"] = audit_command
            write_json(receipt_path, receipt)
            with (control / "audit.log").open("x") as stream:
                audit = subprocess.Popen(
                    audit_command,
                    cwd=GROUP,
                    env=audit_env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                receipt["audit_pid"] = audit.pid
                write_json(receipt_path, receipt)
                receipt["audit_exit_code"] = audit.wait()
            assert not gpu_compute_apps(), (
                "GPU compute process remains after audit exit"
            )
            if (output / "independent-audit.json").is_file():
                audit_result = json.loads(
                    (output / "independent-audit.json").read_text()
                )
                receipt["audit"] = {
                    "file": record(output / "independent-audit.json"),
                    "valid_forward": audit_result["valid_forward"],
                }
        receipt["parent_inputs_after"] = {
            name: record(parent / name) for name in PARENT_SHA256
        }
        assert all(
            item["sha256"] == PARENT_SHA256[name]
            for name, item in receipt["parent_inputs_after"].items()
        )
        assert preflight_parent(parent)["ready"]
        parent_steps = edge["parent_optimizer_steps"]
        completed_updates = fit_summary.get("final", {}).get("local_iteration", 0)
        completed = (
            receipt["fit_exit_code"] == 0
            and receipt.get("fit_status")
            in {"finite_budget_exhausted", "time_budget_exhausted"}
            and 1 <= completed_updates <= 25
            and fit_summary.get("final", {}).get("optimizer_steps")
            == {
                "q": parent_steps["q"] + completed_updates,
                "pose": parent_steps["pose"] + completed_updates,
            }
            and receipt["audit_exit_code"] == 0
            and receipt.get("audit", {}).get("valid_forward") is True
            and not interrupted
        )
        receipt["completed_updates"] = completed_updates
        receipt["status"] = "completed_audited_chunk" if completed else "unresolved"
        write_json(receipt_path, receipt)
        if output.is_dir():
            shutil.copy2(receipt_path, output / "pilot-driver.json")
            shutil.copytree(control, output / "pilot-control")
    except BaseException as error:
        receipt["status"] = "driver_failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_json(receipt_path, receipt)
        if output.is_dir():
            shutil.copy2(receipt_path, output / "pilot-driver.json")
            if not (output / "pilot-control").exists():
                shutil.copytree(control, output / "pilot-control")
        raise
    return 0 if completed else 1


if __name__ == "__main__":
    if "--fit-worker" in sys.argv:
        sys.argv.remove("--fit-worker")
        cherries.main(main, profile=ProfileJoint)
    else:
        raise SystemExit(drive())
