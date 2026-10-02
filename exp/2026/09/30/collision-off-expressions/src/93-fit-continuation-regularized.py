# Copyright (c) 2026 liblaf
# ruff: noqa: C901, E402, PLR0912, PLR0915, FBT001, FBT003, PT018
"""Optional bounded continuation from an independently audited regularized endpoint."""

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
from collections.abc import Callable, Mapping
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

from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)
POSE_SCALE = torch.tensor([math.pi / 18.0] * 3 + [0.01] * 3, dtype=torch.float64)
MAX_ROTATION_INCREMENT_DEG = 1.0
MAX_TRANSLATION_INCREMENT_M = 0.001
PARENT_SHA256 = {
    "protocol.json": "f96ff8771d825b6871518d7a2e0768293bc192ff5d29af1964cdaa261d8183c9",
    "summary.json": "8230a955124d0535d1c53eaf92ba9053bc991f99af5506a5d613ba8ee9b4abbd",
    "progress.jsonl": "acaa90754a9698fba293e38b55be6efd0ea8c658a1b4a1dc86c46aa6c2390684",
    "endpoint.npz": "2d61772610f284daf893ef02f2c99cb69343e50a109001ec5d52bc09ea7c2f11",
    "checkpoint.pt": "37ba9ddc89bb3eb256a8c958f25f65f45d398875929e9d626b1b4cd9e69de1b4",
    "independent-audit.json": "69338baf9a3071d0c7a4157bcbfa5603dee191e2b93c8b30e8345c9ed8f58518",
    "continuation-replay.json": "d27e1583b4b99a2ae9cc064d4bcee6b4c7898321df5171c0d2eb5a79a7c5633c",
    "initial-checkpoint.pt": "39c9cdd768f30d05270c32707db13b53fe4853c7303a1e2f2a07979e510a4d46",
    "initial-endpoint.npz": "bc46375f796164941e52b764a19f14f8b88236d4b3e7446bcb997967bd836992",
    "sources/new-neutral/83-run-strict-continuation.py": "21cd28184ddca0546d66ba100eb03bb5763e9afe812ce662641d13bc1820f11c",
}
PARENT_FIT_SOURCE_SHA256 = (
    "21cd28184ddca0546d66ba100eb03bb5763e9afe812ce662641d13bc1820f11c"
)
FIT_SOURCE_SHA256 = "9f1965dfff08e587cf8cbc64fa97f555cc696f7c8bdefb122d6d78ddb4110ced"
AUDIT_SOURCE_SHA256 = "8158f6413eb932d3df21edd28d1cf287f15cd898b24dc0eaa5b98a5adb56ed83"
OBJECTIVE_SOURCE_SHA256 = (
    "9366c6b6da00ed0355200c37c4de6e206c2f2ba9b70c550a33a834769531f9b3"
)
SMILE_PARENT_SHA256 = {
    "protocol.json": "fead9bd0afcbb0371c21c922804b5c2cc0fa6379ae6b44116796db3cc4e5b339",
    "summary.json": "fb9d765f53c61fe24045a88dfe34c31a229c1de19396cb7691bce586de6bbdbc",
    "progress.jsonl": "24057dfeb1fa084606c091b26b32cd449c9b6b56d97c0a7a606d8e9f98ef3c1a",
    "endpoint.npz": "b0a2c80f825f9fc2cffe5d7ff2b867fe69a1049a208321d2ad7b3cedb229b625",
    "checkpoint.pt": "76ccdc5e948c47debea1ce8632ea3b3746778d9e2c2ecfcba4df7604f8f1ad73",
    "independent-audit.json": "82b83aea48d15e0ff4f19e706e7a99d06f0c01d6be58999a4d6541081a2062a5",
}
SMILE_REFINEMENT_SUMMARY_SHA256 = (
    "0b429cfaad0dcf6340663d7fab533705c80fa06ee50c8e720d4532caccffd207"
)
SMILE_REFINEMENT_ARCHIVE_SHA256 = (
    "b50cd8730e779ada304dcc02721aaad1f502d4adc5d8191cefbe077b2a94a852"
)
SMILE_REFINEMENT_SOURCE_SHA256 = (
    "dfae02b344f4ccc6c3e769b50a36e803bafb7bf5a01ea44b4dd2e2b07e3bf67c"
)


class Config(cherries.BaseConfig):
    expression_name: str = "MouthOpen"
    output_dir: Path = GROUP / "data/regularized-expression-001"
    parent_kind: str = "regularized_audited"
    calibration_only: bool = False
    calibration_control: bool = False
    refinement_archive: Path | None = None
    candidate_id: str = "UNSET_CANDIDATE"
    normal_weight: float | None = None
    smooth_weight: float | None = None
    objective_epoch: str = "regularized"
    objective_source_sha256: str = OBJECTIVE_SOURCE_SHA256
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    blendshape_dir: Path = GROUP / "data/blendshapes-isfixed-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    initialization_checkpoint: Path | None = None
    parent_protocol_sha256: str = ""
    parent_checkpoint_sha256: str = ""
    parent_audit_sha256: str = ""
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
    deadline_unix_s: float | None = None
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
    expected_parent = (GROUP / "data/mouthopen-strict-continuation-001").resolve()
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
        and sha256(GROUP / "src/83-run-strict-continuation.py")
        == PARENT_FIT_SOURCE_SHA256
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
    """Bind the completed audited continuation and its exact saved state."""
    preflight = preflight_parent(parent)
    assert preflight["ready"], preflight
    summary = json.loads((parent / "summary.json").read_text())
    protocol = json.loads((parent / "protocol.json").read_text())
    audit = json.loads((parent / "independent-audit.json").read_text())
    replay = json.loads((parent / "continuation-replay.json").read_text())
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
    initialization = protocol["initialization"]
    assert "baseline_refinement" not in initialization
    previous_edge = initialization["continuation_edge"]
    assert previous_edge["kind"] == "ordinary_checkpoint_continuation"
    assert previous_edge["optimizer_updates"] == 0
    assert Path(previous_edge["parent_run"]).name == "mouthopen-refined-pilot-002"
    assert initialization["optimizer_state"]["continued"] is True
    assert (
        protocol["source_sha256"]["new-neutral/83-run-strict-continuation.py"]
        == PARENT_FIT_SOURCE_SHA256
    )
    assert replay["schema"] == "collision-off-strict-continuation-replay-v1"
    assert replay["status"] == "identical_saved_state"
    assert replay["parameters_unchanged"] is True
    assert replay["optimizer_updates"] == 0
    assert replay["maximum_displacement_change_m"] == 0
    assert replay["normalized_pose_exact"] is True
    assert replay["optimizer_state_preserved_exactly"] is True
    for key, name in (
        ("saved_initial_checkpoint", "initial-checkpoint.pt"),
        ("saved_initial_endpoint", "initial-endpoint.npz"),
    ):
        assert Path(replay[key]["path"]).name == name
        assert replay[key]["sha256"] == PARENT_SHA256[name]
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
    initial_state = torch.load(
        parent / "initial-checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert replay["optimizer_steps_unchanged"] == initial_state["optimizer_steps"]
    assert (
        initialization["optimizer_state"]["initial_steps"]
        == initial_state["optimizer_steps"]
    )
    assert state["local_iteration"] == final["local_iteration"]
    assert state["optimizer_steps"] == final["optimizer_steps"]
    assert state["optimizer_step"] == max(final["optimizer_steps"].values())
    torch.testing.assert_close(
        state["pose_normalized"] * POSE_SCALE,
        state["pose_rad_m"],
        rtol=0,
        atol=0,
    )
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
    assert next_alpha == 0.25
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


def smile_refinement_inputs(
    parent: Path, refinement_archive: Path
) -> tuple[dict, np.ndarray]:
    """Bind Smile004 and its certified unchanged-parameter 82 equilibrium."""
    expected_parent = (GROUP / "data/smile-004").resolve()
    expected_refinement = (
        GROUP / "data/smile-baseline-resolution-001/refined-baseline.npz"
    ).resolve()
    assert parent == expected_parent
    assert refinement_archive.resolve() == expected_refinement
    for name, digest in SMILE_PARENT_SHA256.items():
        assert sha256(parent / name) == digest, name
    refinement_run = expected_refinement.parent
    assert sha256(refinement_run / "summary.json") == SMILE_REFINEMENT_SUMMARY_SHA256
    assert sha256(expected_refinement) == SMILE_REFINEMENT_ARCHIVE_SHA256
    assert (
        sha256(GROUP / "src/82-smile-baseline-resolution.py")
        == SMILE_REFINEMENT_SOURCE_SHA256
    )
    summary = json.loads((parent / "summary.json").read_text())
    protocol = json.loads((parent / "protocol.json").read_text())
    audit = json.loads((parent / "independent-audit.json").read_text())
    refined_summary = json.loads((refinement_run / "summary.json").read_text())
    assert summary["status"] == "pose_projection_failed"
    assert summary["inverse_converged"] is False
    assert (
        protocol["expression_name"] == "Smile"
        and protocol["collision_enabled"] is False
    )
    assert audit["valid_forward"] is True and audit["expression_name"] == "Smile"
    assert refined_summary["status"] == "baseline_refined_requires_probe"
    refined = refined_summary["refined_baseline"]
    assert refined["force"] <= 1e-12 and refined["within_declared_policy"] is True
    state = torch.load(parent / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["optimizer_steps"] == {"q": 86, "pose": 86}
    assert len(state["moments"]) == 4
    with np.load(parent / "endpoint.npz", allow_pickle=False) as archive:
        parent_u = np.asarray(archive["displacement_m"], dtype=np.float64)
        parent_ids = np.asarray(archive["active_cell_ids"], dtype=np.int64)
        np.testing.assert_array_equal(
            archive["activation_inv"], state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            archive["pose_rad_m"], state["pose_rad_m"].numpy()
        )
    with np.load(expected_refinement, allow_pickle=False) as archive:
        refined_u = np.asarray(archive["displacement_m"], dtype=np.float64)
        np.testing.assert_array_equal(
            archive["activation_inv"], state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            archive["pose_normalized"], state["pose_normalized"].numpy()
        )
        np.testing.assert_array_equal(
            archive["pose_rad_m"], state["pose_rad_m"].numpy()
        )
        np.testing.assert_array_equal(archive["active_cell_ids"], parent_ids)
    assert refined_u.shape == parent_u.shape and np.isfinite(refined_u).all()
    event = {
        "kind": "equilibrium_refinement_at_unchanged_parameters",
        "optimizer_updates": 0,
        "parent_run": str(parent),
        "parent_inputs": {name: record(parent / name) for name in SMILE_PARENT_SHA256},
        "refinement_summary": record(refinement_run / "summary.json"),
        "refinement_archive": record(expected_refinement),
        "refinement_source": record(GROUP / "src/82-smile-baseline-resolution.py"),
        "parent_loss": summary["final"]["loss"],
        "refined_loss": refined["loss"],
        "loss_change": refined["loss"] - summary["final"]["loss"],
        "maximum_displacement_change_m": float(np.max(np.abs(refined_u - parent_u))),
        "refined_force": refined["force"],
        "refined_geometry": refined["geometry"],
        "optimizer_state": "Exact parent q, normalized pose, four moments, counters, convergence reference and history are retained; only the certified same-parameter equilibrium displacement is substituted.",
        "stationarity_monitor": "disabled; inherited historical monitor crosses the zero-update equilibrium refinement",
    }
    return event, refined_u


def regularized_parent_inputs(cfg: Config) -> dict:
    """Hash-bind an independently audited regularized parent checkpoint."""
    assert cfg.initialization_checkpoint is not None
    parent = cfg.initialization_checkpoint.resolve().parent
    protocol_path, summary_path, audit_path = (
        parent / "protocol.json",
        parent / "summary.json",
        parent / "independent-audit.json",
    )
    for expected, path in (
        (cfg.parent_protocol_sha256, protocol_path),
        (cfg.parent_checkpoint_sha256, cfg.initialization_checkpoint),
        (cfg.parent_audit_sha256, audit_path),
    ):
        assert len(expected) == 64 and all(
            char in "0123456789abcdef" for char in expected
        )
        assert sha256(path) == expected, path
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert protocol["schema"] == "corrected-neutral-collision-off-rigid6-inverse-v1"
    assert protocol["expression_name"] == cfg.expression_name
    assert protocol["collision_enabled"] is False
    assert protocol["objective_epoch"] == "regularized"
    objective = protocol["objective"]
    assert (
        objective["kind"]
        == "l2_position_plus_oriented_normal_plus_activation_smoothness"
    )
    assert objective["weights"] == {
        "normal": cfg.normal_weight,
        "smooth": cfg.smooth_weight,
    }
    assert objective["source"]["sha256"] == OBJECTIVE_SOURCE_SHA256
    assert protocol["force_contract"]["atol"] == 1e-12
    assert protocol["adjoint_contract"]["relative_shift"] == 0
    assert (
        audit["schema"] == "collision-off-regularized-expression-independent-audit-v1"
    )
    assert audit["valid_forward"] is True
    assert audit["inputs"]["protocol"]["sha256"] == cfg.parent_protocol_sha256
    assert audit["inputs"]["endpoint"]["sha256"] == summary["endpoint"]["sha256"]
    final = summary["final"]
    assert summary["inverse_converged"] is False
    assert final["valid_forward"] is True
    assert final["objective_epoch"] == "regularized"
    assert final["objective_components"]["total"] == final["loss"]
    progress = [
        json.loads(line)
        for line in (parent / "progress.jsonl").read_text().splitlines()
    ]
    assert progress[-1] == final
    state = torch.load(
        cfg.initialization_checkpoint, map_location="cpu", weights_only=False
    )
    with np.load(parent / "endpoint.npz", allow_pickle=False) as endpoint:
        np.testing.assert_array_equal(
            endpoint["displacement_m"], state["displacement_m"].numpy()
        )
        np.testing.assert_array_equal(
            endpoint["activation_inv"], state["activation_inv"].numpy()
        )
        np.testing.assert_array_equal(
            endpoint["pose_rad_m"], state["pose_rad_m"].numpy()
        )
    assert len(state["moments"]) == 4
    assert state["optimizer_steps"] == final["optimizer_steps"]
    assert state["convergence_reference"] is not None
    assert len(state["convergence_history_tail"]) > 0
    return {
        "kind": "ordinary_regularized_checkpoint_continuation",
        "optimizer_updates": 0,
        "parent_run": str(parent),
        "parent_inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "progress": record(parent / "progress.jsonl"),
            "endpoint": record(parent / "endpoint.npz"),
            "checkpoint": record(cfg.initialization_checkpoint),
            "audit": record(audit_path),
        },
        "parent_final": final,
        "parent_optimizer_steps": dict(state["optimizer_steps"]),
        "next_trial_alpha": cfg.initial_trial_alpha,
        "stationarity_monitor": "disabled for this bounded continuation; inverse convergence remains false pending independent assessment",
    }


def check_regularized_pullback(
    physics: Any,
    runtime: Any,
    materials: Callable[[torch.Tensor], Mapping[str, Mapping[str, torch.Tensor]]],
    pose: Callable[[torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    jaw: torch.Tensor,
    u: torch.Tensor,
    *,
    grads: tuple[torch.Tensor, torch.Tensor],
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    key: str,
) -> dict[str, Any]:
    """Check the full regularized implicit gradient at fixed free coordinates.

    Unlike the historic position-only helper, this includes the direct-q
    activation regularizer in both the Lagrangian and finite differences.
    """
    model = runtime.forward.model
    q_grad, jaw_grad = grads
    p_free = runtime.warm_adjoints[key].detach()
    free_u = model.dof_map.to_free(u.detach()).clone()
    assert q.shape[1] == 6 and jaw.shape == (6,)
    assert q_grad.shape == q.shape and jaw_grad.shape == jaw.shape
    assert p_free.shape == free_u.shape
    generator = torch.Generator(device=q.device).manual_seed(20260930)
    q_direction = torch.randn(
        q.shape, generator=generator, device=q.device, dtype=q.dtype
    )
    q_direction /= torch.linalg.vector_norm(q_direction)
    jaw_direction = torch.ones_like(jaw) / math.sqrt(jaw.numel())
    original_materials = model.get_materials()
    original_fixed = model.dof_map.fixed_values

    def comparison(
        predicted: torch.Tensor, finite_difference: torch.Tensor
    ) -> dict[str, float]:
        absolute = float((predicted - finite_difference).abs())
        scale = max(abs(float(predicted)), abs(float(finite_difference)), 1e-12)
        return {
            "predicted": float(predicted),
            "finite_difference": float(finite_difference),
            "absolute_error": absolute,
            "relative_error": absolute / scale,
        }

    try:

        def lagrangian(q_value: torch.Tensor, jaw_value: torch.Tensor) -> torch.Tensor:
            model.set_materials(materials(q_value))
            model.dof_map.fixed_values = physics.boundary(pose(jaw_value))
            full_u = model.dof_map.to_full(free_u)
            state = model.State(u=full_u)
            assert model.collision is None
            residual = model.dof_map.to_free_grad(model.grad(state))
            value = objective(full_u, q_value) + torch.dot(p_free, residual)
            assert value.ndim == 0 and bool(torch.isfinite(value))
            return value

        result: dict[str, Any] = {
            "schema": "collision-off-regularized-fixed-state-lagrangian-pullback-v1",
            "definition": "L_regularized(u_fixed_free,q) + p_free dot r_free; q includes the direct activation-smoothness derivative and jaw rebuilds original fixed DOFs.",
            "free_coordinates_held_fixed": True,
            "fixed_coordinates_rebuilt_from_pose": True,
            "collision_enabled": False,
            "q": {},
            "jaw": {},
        }
        predicted_q = torch.sum(q_grad * q_direction)
        predicted_jaw = torch.sum(jaw_grad * jaw_direction)
        for epsilon in (1e-4, 1e-5, 1e-6):
            q_fd = (
                lagrangian(q + epsilon * q_direction, jaw)
                - lagrangian(q - epsilon * q_direction, jaw)
            ) / (2 * epsilon)
            jaw_fd = (
                lagrangian(q, jaw + epsilon * jaw_direction)
                - lagrangian(q, jaw - epsilon * jaw_direction)
            ) / (2 * epsilon)
            label = f"epsilon_{epsilon:.0e}"
            result["q"][label] = comparison(predicted_q, q_fd)
            result["jaw"][label] = comparison(predicted_jaw, jaw_fd)
        return result
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed


def main(cfg: Config) -> None:
    assert len(cfg.pilot_source_sha256) == 64
    assert sha256(Path(__file__)) == cfg.pilot_source_sha256
    assert cfg.expression_name in {"MouthOpen", "Smile"}
    assert cfg.parent_kind == "regularized_audited"
    assert cfg.candidate_id != "UNSET_CANDIDATE"
    assert cfg.normal_weight is not None and cfg.normal_weight > 0
    assert cfg.smooth_weight is not None and cfg.smooth_weight >= 0
    assert cfg.smooth_weight > 0 or cfg.calibration_control, (
        "zero smoothness is a labelled calibration control only"
    )
    assert cfg.initialization_checkpoint is not None
    assert cfg.continue_optimizer_state and not cfg.resume
    assert 0 <= cfg.maximum_iterations <= 25
    assert cfg.forward_atol == 1e-12
    validate_linear_contract(cfg)
    assert cfg.minimum_trial_alpha == 0.001
    assert cfg.wall_seconds is not None and 0 < cfg.wall_seconds <= 3600
    if cfg.deadline_unix_s is not None:
        assert math.isfinite(cfg.deadline_unix_s)
        assert cfg.deadline_unix_s > time.time(), (
            "absolute worker deadline already elapsed"
        )
    assert cfg.enable_stationarity_monitor is False
    assert cfg.objective_epoch == "regularized"
    assert cfg.objective_source_sha256 == OBJECTIVE_SOURCE_SHA256
    assert (
        sha256(GROUP / "src/face_shape_activation_objective.py")
        == cfg.objective_source_sha256
    )
    assert cfg.exclude_fully_fixed_tets
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 0.0001
    assert cfg.refinement_archive is None
    continuation_edge = regularized_parent_inputs(cfg)
    assert 0.001 <= cfg.initial_trial_alpha <= 1
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
    from face_shape_activation_objective import (
        ActivationSmoothness,
        FaceShapeActivationObjective,
        ObjectiveWeights,
        SkinShapeLoss,
        calibrate_smooth_weight,
        normal_anchor_coefficient,
        normal_pose_gradient_ratio,
    )

    skin_loss = SkinShapeLoss(
        np.asarray(physics.points)[skin_ids],
        neutral_points,
        target_points,
        skin_ids,
        tri,
        device="cuda",
        dtype=torch.float64,
    )
    smoothness = ActivationSmoothness(
        np.asarray(physics.points),
        np.asarray(physics.base.tets),
        active_source_ids,
        np.asarray(physics.mesh.cell_data["MuscleFraction"]),
        np.asarray(physics.mesh.cell_data["MuscleId"]),
        device="cuda",
        dtype=torch.float64,
    )
    assert cfg.normal_weight is not None and cfg.smooth_weight is not None
    objective_model = FaceShapeActivationObjective(
        skin_loss,
        smoothness,
        ObjectiveWeights(normal=cfg.normal_weight, smooth=cfg.smooth_weight),
    )
    normal_anchor = normal_anchor_coefficient(float(skin_loss.target_motion_scale2_m2))

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, active_ids, value)
        return result

    def physical_pose(value: torch.Tensor) -> torch.Tensor:
        assert value.shape == (6,)
        return value * POSE_SCALE.to(device=value.device, dtype=value.dtype)

    def objective(u: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return objective_model(u[: len(physics.points)], value)

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
        components = objective_model.components(u[: len(physics.points)], value)
        loss = objective_model(u[: len(physics.points)], value)
        forward = copy.deepcopy(runtime.last_forward)
        component_grads = None
        if gradient:
            component_adjoint = None
            if cfg.calibration_only:
                position_grads = torch.autograd.grad(
                    components["position"], (value, pose_value), retain_graph=True
                )
                position_adjoint = copy.deepcopy(runtime.last_sparse_adjoint)
                normal_grads = torch.autograd.grad(
                    components["normal"], (value, pose_value), retain_graph=True
                )
                normal_adjoint = copy.deepcopy(runtime.last_sparse_adjoint)
                smooth_q = torch.autograd.grad(
                    components["smooth"], value, retain_graph=True
                )[0]
                component_adjoint = {
                    "position": position_adjoint,
                    "normal": normal_adjoint,
                }
            grads = torch.autograd.grad(loss, (value, pose_value))
            if cfg.calibration_only:
                component_grads = {
                    "position_q": position_grads[0].detach(),
                    "position_pose": position_grads[1].detach(),
                    "normal_q": normal_grads[0].detach(),
                    "normal_pose": normal_grads[1].detach(),
                    "smooth_q": smooth_q.detach(),
                    "unshifted_adjoint": component_adjoint,
                }
        else:
            grads = None
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
            if component_grads is not None:
                for name, receipt in component_grads["unshifted_adjoint"].items():
                    assert receipt["relative_shift"] == 0, (name, receipt)
                    assert (
                        receipt["original_unshifted_relative_residual"]
                        <= cfg.adjoint_rtol
                    ), (
                        name,
                        receipt,
                    )
            runtime.last_adjoint["sparse_solver"] = copy.deepcopy(
                runtime.last_sparse_adjoint
            )
        return {
            "q": value.detach(),
            "pose": pose_value.detach(),
            "u": u.detach(),
            "loss": float(loss),
            "objective_components": {
                name: float(term.detach()) for name, term in components.items()
            },
            "objective_metrics": objective_model.metrics(
                u[: len(physics.points)], value
            ),
            "component_grads": component_grads,
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
            "objective_epoch": cfg.objective_epoch,
            "objective_components": {
                "position": candidate["objective_components"]["position"],
                "normal": candidate["objective_components"]["normal"],
                "smooth": candidate["objective_components"]["smooth"],
                "normal_contribution": cfg.normal_weight
                * candidate["objective_components"]["normal"],
                "smooth_contribution": cfg.smooth_weight
                * candidate["objective_components"]["smooth"],
                "total": candidate["loss"],
            },
            "fit_rms_mm": candidate["objective_metrics"]["position_rms_mm"],
            "normal_angle_rms_deg": candidate["objective_metrics"][
                "normal_angle_rms_deg"
            ],
            "activation_smoothness": candidate["objective_metrics"][
                "activation_smoothness"
            ],
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
                    "activation": "B = I + symmetric(q); six independent dimensionless components per active muscle tetrahedron; no projection, clipping, positivity, rank or magnitude constraint; the declared objective adds within-muscle smoothness.",
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
                "objective": {
                    "kind": "l2_position_plus_oriented_normal_plus_activation_smoothness",
                    "weights": {
                        "normal": cfg.normal_weight,
                        "smooth": cfg.smooth_weight,
                    },
                    "skin_contract": skin_loss.contract(),
                    "smoothness_contract": smoothness.contract(),
                    "source": record(GROUP / "src/face_shape_activation_objective.py"),
                    "normal_anchor_2mm_5deg": normal_anchor,
                },
                "objective_epoch": cfg.objective_epoch,
                "candidate_id": cfg.candidate_id,
                "materials": strain_receipt,
                "tetrahedron_policy": tetrahedron_policy,
                "inversion_policy": inversion_policy,
                "boundary_policy": "original FEM IsFixed only; rigid obstacle nodes prescribed; runtime fixed and free DOF arrays verified",
                "seed_method": cfg.seed_method,
                "collision_enabled": False,
                "ipc_policy": "collision=None in forward energy, residual, Hessian, adjoint and admission; CCD disabled",
                "coordinate_contract": "X is the repaired constitutive reference; q, normalized pose and optimizer state come from the exact bound parent, while Smile may use the separately recorded certified same-parameter equilibrium displacement.",
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

    assert cfg.wall_seconds is not None and cfg.wall_seconds > 0
    started = time.perf_counter()
    deadline = started + cfg.wall_seconds
    if cfg.deadline_unix_s is not None:
        deadline = min(deadline, started + (cfg.deadline_unix_s - time.time()))
    assert deadline > time.perf_counter(), "no worker time remains after setup"
    runtime.deadline = deadline
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
        assert math.isfinite(replayed["loss"])
        replay_receipt["parent_loss_comparison"] = (
            "not applicable: this run starts a declared regularized objective epoch"
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
        check = check_regularized_pullback(
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
        assert min(row["relative_error"] for row in check["jaw"].values()) < 1e-3, (
            check["jaw"]
        )
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
            initial_state["displacement_m"].numpy(), seed.cpu().numpy()
        )
        replay_receipt["saved_initial_checkpoint"] = record(
            output / "initial-checkpoint.pt"
        )
        replay_receipt["saved_initial_endpoint"] = record(
            output / "initial-endpoint.npz"
        )
        replay_receipt["optimizer_state_preserved_exactly"] = True
        replay_receipt["initial_displacement_source"] = "parent_checkpoint"
        write_json(output / "continuation-replay.json", replay_receipt)
        write_json(output / "initial-adjoint.json", current["adjoint"])

    if cfg.calibration_only:
        assert cfg.calibration_control and cfg.smooth_weight == 0
        assert current["component_grads"] is not None
        calibration = calibrate_smooth_weight(
            current["component_grads"]["position_q"],
            current["component_grads"]["normal_q"],
            current["component_grads"]["smooth_q"],
            smoothness.mass_weight,
            normal_coefficient=cfg.normal_weight,
            smooth_target_ratio=0.1,
        )
        calibration.update(
            {
                "schema": "collision-off-regularized-objective-calibration-v1",
                "candidate_id": cfg.candidate_id,
                "expression_name": cfg.expression_name,
                "objective_epoch": cfg.objective_epoch,
                "optimizer_updates": 0,
                "parameters_unchanged": True,
                "force": current["direct_free_force"],
                "force_threshold": cfg.forward_atol,
                "geometry": geometry_metrics(current["u"][: len(physics.points)]),
                "components": current["objective_components"],
                "metrics": current["objective_metrics"],
                "normal_pose_to_position_gradient_ratio": normal_pose_gradient_ratio(
                    current["component_grads"]["position_pose"],
                    current["component_grads"]["normal_pose"],
                    normal_coefficient=cfg.normal_weight,
                ),
                "component_unshifted_adjoint": current["component_grads"][
                    "unshifted_adjoint"
                ],
                "combined_unshifted_adjoint": current["adjoint"],
                "initial_checkpoint": record(output / "initial-checkpoint.pt"),
                "initial_endpoint": record(output / "initial-endpoint.npz"),
                "parent_checkpoint": record(initialization_checkpoint),
                "parent_endpoint": record(initialization_endpoint),
                "objective_source": record(
                    GROUP / "src/face_shape_activation_objective.py"
                ),
                "skin_contract": skin_loss.contract(),
                "smoothness_contract": smoothness.contract(),
            }
        )
        write_json(output / "objective-calibration.json", calibration)
        summary = json.loads((output / "summary.json").read_text())
        summary.update(
            {
                "status": "calibration_complete_no_optimizer_updates",
                "inverse_converged": False,
                "optimizer_updates": 0,
                "objective_calibration": record(output / "objective-calibration.json"),
            }
        )
        write_json(output / "summary.json", summary)
        cherries.log_output(output / "objective-calibration.json")
        cherries.log_output(output / "summary.json")
        return

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
        message = "86 is a worker only; launch it through the serial calibration driver with --fit-worker"
        raise SystemExit(message)
