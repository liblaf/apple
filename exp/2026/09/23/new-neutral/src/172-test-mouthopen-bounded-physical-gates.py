"""Correct the original saved bounded-joint seed under original physical gates."""

# ruff: noqa: C901, E402, PLR0912, PLR0915, SLF001
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

from joint_common import ProfileJoint, sha256, write_json
from joint_coupled_predictor import audit_coupled_motion, rotation_sagitta
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from mouthopen_coupled_seed import _mandible_arc_radius
from mouthopen_pose_projection import (
    determinant_ratio,
)
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-012"
    output_dir: Path = GROUP / "data/mouthopen-bounded-joint-002"
    seed_run: Path = GROUP / "data/mouthopen-bounded-joint-001"
    gradient_run: Path = GROUP / "data/mouthopen-joint-constraints-001"
    strain_increment_limit: float = 0.01
    relative_shift: float = 1e-5
    epsilon: float = 0.00005
    determinant_margin: float = 1e-6
    activation_threshold: float = 0.05
    descent_fraction: float = 0.1
    candidate_alpha: float = 0.125
    case_wall_seconds: float = 600.0
    internal_atol: float = 1e-9
    ipc_threads: int = 4


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def bound(item: dict) -> Path:
    path = Path(item["path"]).resolve()
    assert record(path) == item, path
    return path


def main(cfg: Config) -> None:
    assert cfg.relative_shift == 1e-5
    assert cfg.strain_increment_limit == 0.01
    assert cfg.candidate_alpha == 0.125
    assert cfg.internal_atol == 1e-9
    assert 0 < cfg.epsilon < 1
    assert 0 < cfg.determinant_margin < cfg.activation_threshold
    assert 0 < cfg.descent_fraction <= 1
    assert 0 < cfg.candidate_alpha <= 1
    assert 0 < cfg.case_wall_seconds <= 600
    source, output = cfg.source_run.resolve(), cfg.output_dir.resolve()
    assert not output.exists(), output
    refs = {
        name: record(source / name)
        for name in (
            "summary.json",
            "protocol.json",
            "endpoint.npz",
            "checkpoint.pt",
            "independent-audit.json",
        )
    }
    protocol = json.loads((source / "protocol.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    audit = json.loads((source / "independent-audit.json").read_text())
    assert summary["status"] != "running"
    assert summary["endpoint"] == refs["endpoint.npz"]
    assert audit["valid_forward"]
    for item in audit["inputs"].values():
        bound(item)
    for key in ("summary", "protocol", "endpoint"):
        assert (
            audit["inputs"][key]
            == refs[key + (".npz" if key == "endpoint" else ".json")]
        )
    assert protocol["force_contract"]["atol"] == 1e-8
    policy = protocol["inversion_policy"]
    assert policy["maximum_inverted_tetrahedra"] == 100
    assert policy["maximum_inverted_rest_volume_fraction"] == 0.0001
    assert policy["orientation_floor"] is None
    checkpoint = torch.load(
        source / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["optimizer_steps"] == {"q": 268, "pose": 268}
    assert checkpoint["iteration"] == summary["final"]["iteration"]
    assert checkpoint["optimizer_steps"] == summary["final"]["optimizer_steps"]
    with np.load(source / "endpoint.npz", allow_pickle=False) as saved:
        for key in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        active_ids = saved["active_cell_ids"].copy()
    reference_path = bound(protocol["sources"]["reference_repair"])
    neutral_path = bound(protocol["sources"]["neutral_endpoint"])
    target_path = bound(protocol["sources"]["blendshapes"])
    model_dir = source / "affine-projections/00002/model"
    trial_dir = source / "affine-projections/00002/trial-01"
    cached_summary = json.loads((model_dir / "summary.json").read_text())
    assert cached_summary["status"] == "model_ready_for_trial_projection"
    assert cached_summary["relative_shift"] == cfg.relative_shift
    assert cached_summary["epsilon"] == cfg.epsilon == 5e-5
    cache_path = bound(cached_summary["cache"])
    assert cache_path == model_dir / "cache/coefficients.npz"
    cached_trial = json.loads((trial_dir / "summary.json").read_text())
    assert cached_trial["status"] == "certified_linear_increment"
    assert cached_trial["alpha"] == cfg.candidate_alpha
    assert cached_trial["margin"] == cfg.determinant_margin
    assert cached_trial["cache"] == cached_summary["cache"]
    with np.load(model_dir / "source-and-response.npz") as saved:
        for key in ("activation_inv", "pose_normalized", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
    calibration_refs = {
        "model_summary": record(model_dir / "summary.json"),
        "model_source_state": record(model_dir / "source-and-response.npz"),
        "coefficients": record(cache_path),
        "trial_summary": record(trial_dir / "summary.json"),
        "trial_input_receipt": record(trial_dir / "input-receipt.json"),
        "trial_constraints": record(trial_dir / "constraints-00.npz"),
        "trial_predictions": record(trial_dir / "predictions.npz"),
    }
    frozen, frozen_refs = load_frozen(cfg.gradient_run.resolve())
    assert frozen["protocol"]["source_refs"] == refs
    assert frozen["protocol"]["calibration_refs"] == calibration_refs
    replay, replay_refs = load_saved_seed(cfg.seed_run.resolve(), frozen, frozen_refs)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    physics, _ = build_rebased_physics(reference_path.parent, inverse=True)
    model = physics.runtime.forward.model
    isfixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    expected_fixed = np.r_[
        np.repeat(isfixed, 3),
        np.ones(model.dof_map.n_full - 3 * len(isfixed), dtype=bool),
    ]
    np.testing.assert_array_equal(
        model.dof_map.fixed_indices.cpu(), np.flatnonzero(expected_fixed)
    )
    np.testing.assert_array_equal(
        model.dof_map.free_indices.cpu(), np.flatnonzero(~expected_fixed)
    )
    assert not isfixed[np.asarray(physics.mesh.point_data["IsLip"], dtype=bool)].any()
    install_active_strain(model)
    exclusion = exclude_fully_fixed_tetrahedra(physics)
    expected_exclusion = dict(protocol["tetrahedron_policy"])
    expected_exclusion.pop("neutral_free_equation_proof")
    assert exclusion == expected_exclusion
    baseline = model.get_materials()
    with np.load(
        neutral_path.parent / "active-strain-fields.npz", allow_pickle=False
    ) as saved:
        for key, saved_key in (
            ("activation_inv", "skin_activation_inverse"),
            ("mu", "skin_mu_mpa"),
            ("thickness", "skin_thickness_m"),
        ):
            np.testing.assert_array_equal(baseline["skin"][key].cpu(), saved[saved_key])
    np.testing.assert_array_equal(active_ids, physics.base.retained_active_cell_ids)
    q = checkpoint["activation_inv"].clone().cuda()
    pose = checkpoint["pose_normalized"].clone().cuda()
    old_u = checkpoint["displacement_m"].clone().cuda()
    scales = torch.tensor(
        [math.pi / 18] * 3 + [0.01] * 3, device=pose.device, dtype=pose.dtype
    )
    torch.testing.assert_close(
        pose * scales, checkpoint["pose_rad_m"].cuda(), rtol=0, atol=0
    )
    fixed = physics.boundary(pose * scales).detach().clone()
    torch.testing.assert_close(
        old_u.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=5e-16
    )
    moments = [value.clone().cuda() for value in checkpoint["moments"]]
    assert len(moments) == 4

    def materials(value: torch.Tensor) -> dict:
        result = {name: dict(fields) for name, fields in baseline.items()}
        result["muscle"]["activation_inv"] = baseline["muscle"][
            "activation_inv"
        ].index_copy(0, physics.base.active_t, value)
        return result

    old_materials = materials(q)
    kappa = float(protocol["ipc_stiffness_mpa"])
    assert kappa == 1.3544
    collision = model.collision
    assert collision is not None
    potential = collision.potential
    collision.potential = ipctk.BarrierPotential(
        type(potential.barrier)(), potential.dhat, kappa, collision.use_physical_barrier
    )
    physics.contact_definition["config"]["stiffness_mpa"] = kappa
    with np.load(target_path, allow_pickle=False) as data:
        index = list(data["expression_names"]).index("MouthOpen")
        skin_ids = data["skin_global_ids"].copy()
        triangles = data["skin_triangles"].copy()
        neutral = data["new_neutral_points_m"].copy()
        target = data["target_points_m"][index].copy()
    xyz = neutral[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
    )
    weights = np.zeros(len(skin_ids))
    np.add.at(weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights /= weights.sum()
    scale2 = float(np.sum(weights * np.sum((target - neutral) ** 2, axis=1)))
    assert scale2 > 0
    target_u = torch.as_tensor(target - physics.points[skin_ids], device=q.device)
    weights_t = torch.as_tensor(weights, device=q.device)

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (weights_t[:, None] * (u[skin_ids] - target_u).square()).sum() / scale2

    reference = np.asarray(physics.points)
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)
    retained_tets = np.asarray(physics.base.tets)[retained_ids]
    old_np = old_u.cpu().numpy()
    old_all_j = determinant_ratio(reference, retained_tets, old_np)
    active = (old_all_j > 0) & (old_all_j <= cfg.activation_threshold)
    assert active.any()
    output.mkdir(parents=True)
    source_manifest = {}
    for label, source_root in (
        ("new-neutral", GROUP / "src"),
        ("joint", JOINT),
        ("solvers", SOLVERS),
        ("apple", ROOT / "src/liblaf/apple"),
    ):
        for path in source_root.rglob("*.py"):
            relative = path.relative_to(source_root)
            destination = output / "sources" / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            assert sha256(path) == sha256(destination)
            source_manifest[str(Path(label) / relative)] = record(destination)
    np.testing.assert_array_equal(frozen["source_q"], q.cpu().numpy())
    np.testing.assert_array_equal(frozen["source_pose"], pose.cpu().numpy())
    with np.load(cache_path) as saved:
        np.testing.assert_array_equal(saved["original_ids"], retained_ids)
        np.testing.assert_array_equal(saved["old_j"], old_all_j)
        residual_delta = saved["residual_delta_j"].copy()
    positive = old_all_j > 0
    write_json(
        output / "protocol.json",
        {
            "schema": "mouthopen-saved-bounded-seed-physical-gates-v1",
            "config": cfg.model_dump(mode="json"),
            "source_refs": refs,
            "calibration_refs": calibration_refs,
            "frozen_projection_refs": frozen_refs,
            "saved_seed_refs": replay_refs,
            "script": record(Path(__file__)),
            "source_snapshot": source_manifest,
            "tetrahedron_policy": exclusion,
            "isfixed_exact": True,
            "skin_fields_exact": True,
            "ipc_stiffness_mpa": kappa,
            "original_acceptance_force": 1e-8,
            "internal_force_target": cfg.internal_atol,
            "inversion_policy": policy,
            "source_optimizer_steps": checkpoint["optimizer_steps"],
            "inverse_adopted": False,
            "direction_policy": "Replay the original hash-bound171 saved seed and target controls from its certified bounded joint projection. No new predictor, QP or adjoint. The extra identity-preservation and affine-positivity screen is recorded but bypassed for this diagnostic; it is not part of the original count/volume physical policy.",
            "trust_policy": "Bind the previously certified box and metric-trust checks. Seed admission uses original rigid-arc CCD, inverted-count100 and inverted-volume1e-4 gates. orientation_floor remains None. Final force/contact/Armijo gates remain original.",
            "budget_scope": "600 seconds for fresh original-policy admission checks on the saved seed and one nonlinear corrector. No predictor or projection solve. Physical rebuild excluded; serialization included; fresh failure metrics may finish after deadline but no solve starts after it.",
        },
    )
    runtime = install_mouthopen_hybrid_runtime(
        physics,
        forward_atol=cfg.internal_atol,
        adjoint_rtol=1e-7,
        adjoint_relative_shift=cfg.relative_shift,
        newton_max_steps=3000,
        max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
        fixed_stiffness_mpa=kappa,
    )
    started = time.perf_counter()
    deadline = started + cfg.case_wall_seconds
    runtime.deadline = deadline
    row = {
        "status": "running",
        "inverse_adopted": False,
        "extra_proposal_screen_required": False,
        "diagnostic_bypasses_extra_identity_and_affine_screen": True,
        "original_physical_policy_unchanged": True,
        "original_saved_seed_replayed": replay_refs["seed.npz"],
    }
    stage = "saved_seed_admission"

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            message = "Declared bounded-joint diagnostic budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    def save_state(
        path: Path, u: torch.Tensor, target_q: torch.Tensor, target_pose: torch.Tensor
    ) -> np.ndarray:
        value = u.detach().cpu().numpy()
        j = determinant_ratio(reference, retained_tets, value)
        np.savez_compressed(
            path,
            displacement_m=value,
            activation_inv=target_q.cpu().numpy(),
            pose_normalized=target_pose.cpu().numpy(),
            pose_rad_m=(target_pose * scales).cpu().numpy(),
            original_retained_ids=retained_ids,
            old_det_f=old_all_j,
            det_f=j,
            observed_affine_det_f=j + residual_delta,
            newly_inverted_original_ids=retained_ids[positive & (j <= 0)],
            sign_changed_original_ids=retained_ids[positive != (j > 0)],
        )
        return j

    try:
        check_budget()
        increment = replay["increment"]
        certificate = replay["certificate"]
        delta_q = torch.as_tensor(
            increment[: q.numel()].reshape(q.shape), device=q.device, dtype=q.dtype
        )
        delta_pose = torch.as_tensor(
            increment[q.numel() :], device=pose.device, dtype=pose.dtype
        )
        next_q, next_pose = q + delta_q, pose + delta_pose
        np.testing.assert_array_equal(next_q.cpu().numpy(), replay["activation_inv"])
        np.testing.assert_array_equal(
            next_pose.cpu().numpy(), replay["pose_normalized"]
        )
        np.testing.assert_array_equal(
            (next_pose * scales).cpu().numpy(), replay["pose_rad_m"]
        )
        actual_slope = float(frozen["objective_gradient"] @ increment)
        np.testing.assert_allclose(
            actual_slope, certificate["actual_slope"], rtol=0, atol=1e-14
        )
        assert actual_slope <= frozen["descent_target"] + 1e-12
        row.update(
            projection=certificate,
            actual_slope=actual_slope,
            descent_target=frozen["descent_target"],
        )
        np.savez_compressed(
            output / "joint-increment.npz",
            delta_q=delta_q.cpu().numpy(),
            delta_pose=delta_pose.cpu().numpy(),
            actual_joint_increment=increment,
        )
        next_fixed = physics.boundary(next_pose * scales).detach().clone()
        restore_source()
        seed = torch.as_tensor(
            replay["displacement_m"], device=old_u.device, dtype=old_u.dtype
        ).clone()
        seed_j = save_state(output / "seed.npz", seed, next_q, next_pose)
        np.testing.assert_array_equal(seed_j, replay["det_f"])
        np.testing.assert_array_equal(old_all_j, replay["old_det_f"])
        np.testing.assert_array_equal(retained_ids, replay["original_retained_ids"])
        seed_receipt = replay["predictor"]
        write_json(output / "original-predictor-receipt.json", seed_receipt)
        torch.testing.assert_close(
            seed.flatten()[model.dof_map.fixed_indices], next_fixed, rtol=0, atol=0
        )
        angle = float(
            (
                Rotation.from_rotvec((next_pose * scales).cpu().numpy()[:3])
                * Rotation.from_rotvec((pose * scales).cpu().numpy()[:3]).inv()
            ).magnitude()
        )
        assert 0 <= angle <= math.pi
        pivot = torch.as_tensor(physics.pivot_t, device=q.device, dtype=q.dtype)
        radius, _ = _mandible_arc_radius(physics, collision, pivot, fixed, next_fixed)
        motion = audit_coupled_motion(
            collision, old_u, seed, rotation_margin_m=rotation_sagitta(radius, angle)
        )
        seed_geometry = geometry_metrics(physics, seed)
        observed_affine = seed_j + residual_delta
        violated = retained_ids[
            positive & ((seed_j <= 0) | (observed_affine <= 0))
        ].tolist()
        row.update(
            seed_geometry=seed_geometry,
            motion=motion,
            predictor=seed_receipt,
            minimum_positive_seed_J=float(seed_j[positive].min()),
            minimum_positive_observed_affine_J=float(observed_affine[positive].min()),
            strict_model_margin_minimum_slack=float(
                np.minimum(seed_j, observed_affine)[positive].min()
                - cfg.determinant_margin
            ),
            violated_original_positive_ids=violated,
            extra_proposal_screen_passed=not violated,
            original_seed_physical_gates_passed=bool(
                motion["admitted"]
                and seed_geometry["inverted_tetrahedra"] <= 100
                and seed_geometry["inverted_rest_volume_fraction"] <= 1e-4
            ),
        )
        write_json(output / "summary.json", row)
        assert motion["admitted"]
        assert seed_geometry["inverted_tetrahedra"] <= 100
        assert seed_geometry["inverted_rest_volume_fraction"] <= 1e-4
        stage = "single_nonlinear_corrector"
        check_budget()
        restore_source()
        if hasattr(runtime, "last_failed_displacement"):
            del runtime.last_failed_displacement
        try:
            corrected = runtime.primal(
                materials(next_q), next_fixed, seed.detach().clone()
            )
        except ForwardConvergenceError as error:
            row["corrector_failure"] = {"message": str(error), "receipt": error.receipt}
            assert hasattr(runtime, "last_failed_displacement")
            corrected = runtime.last_failed_displacement.detach().clone()
        corrected_j = save_state(output / "corrected.npz", corrected, next_q, next_pose)
        row["forward"] = copy.deepcopy(runtime.last_forward)
        runtime.deadline = None
        stage = "fresh_corrected_metrics"
        model.set_materials(materials(next_q))
        model.dof_map.fixed_values = next_fixed.detach().clone()
        state = model.State(u=corrected.detach().clone())
        state.collision = collision.state_at(state.u)
        force = float(
            torch.linalg.vector_norm(
                FeasibleExpressionProblem(model=model, collision_step_safety=0.9).grad(
                    state
                )
            )
        )
        contact = runtime._contact_gate(state)
        geometry = geometry_metrics(physics, corrected)
        loss, old_loss = float(objective(corrected)), float(objective(old_u))
        armijo = old_loss + 1e-4 * actual_slope
        gates = (
            force <= 1e-8
            and geometry["inverted_tetrahedra"] <= 100
            and geometry["inverted_rest_volume_fraction"] <= 1e-4
            and contact["receipt"]["contact_numerically_valid"]
            and contact["no_intersections"]
            and contact["minimum_active_gap_at_least_buffer"]
        )
        success = (
            "corrector_failure" not in row
            and gates
            and force <= cfg.internal_atol
            and loss < old_loss
            and loss <= armijo
        )
        row.update(
            status="candidate_passed_diagnostic"
            if success
            else "candidate_rejected_diagnostic",
            success=success,
            raw_force=force,
            contact=contact,
            geometry=geometry,
            loss=loss,
            source_loss=old_loss,
            fit_rms_mm=math.sqrt(loss * scale2) * 1000,
            source_fit_rms_mm=math.sqrt(old_loss * scale2) * 1000,
            armijo_threshold=armijo,
            armijo_passed=loss <= armijo,
            original_acceptance_gates_met=gates,
            internal_force_target_met=force <= cfg.internal_atol,
            newly_inverted_original_ids=retained_ids[
                positive & (corrected_j <= 0)
            ].tolist(),
        )
        check_budget()
    except (ForwardConvergenceError, RuntimeError, AssertionError, ValueError) as error:
        row["status"] = "diagnostic_failed"
        row["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ForwardConvergenceError):
            row["failure"]["receipt"] = error.receipt
        LOG.exception("Frozen bounded-joint diagnostic failed at %s", stage)
    finally:
        runtime.deadline = None
        restore_source()
        for name, item in refs.items():
            assert record(source / name) == item
        for item in (
            *calibration_refs.values(),
            *frozen_refs.values(),
            *replay_refs.values(),
        ):
            bound(item)
        for original, value in zip(checkpoint["moments"], moments, strict=True):
            torch.testing.assert_close(original, value.cpu(), rtol=0, atol=0)
        row["source_hashes_unchanged"] = True
        row["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", row)
        cherries.log_output(output / "protocol.json")
        cherries.log_output(output / "summary.json")
    assert row["status"] != "diagnostic_failed", row


def load_frozen(directory: Path) -> tuple[dict, dict]:
    """Validate completed170 and reconstruct its six original inequality rows."""
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    assert summary["status"] == "diagnostic_failed"
    assert summary["failure"]["stage"] == "joint_projection_0"
    assert summary["source_hashes_unchanged"]
    assert summary["source_forward"]["pncg"]["steps"] == 0
    assert summary["source_forward"]["newton"]["steps"] == 0
    assert protocol["source_optimizer_steps"] == {"q": 268, "pose": 268}
    for key in ("source_refs", "calibration_refs", "source_snapshot"):
        for item in protocol[key].values():
            bound(item)
    for key in ("shifted_relative_residual", "native_shifted_relative_residual"):
        assert summary["objective_sparse_adjoint"][key] <= 1e-7
    refs = {
        name: record(directory / name)
        for name in (
            "summary.json",
            "protocol.json",
            "objective-adjoint.json",
            "objective-and-adam.npz",
            "attempt-00/qp-inputs.npz",
            "attempt-00/input-receipt.json",
            "attempt-00/qp-certificate.json",
            "attempt-00/summary.json",
            "attempt-00/joint-increment.npz",
        )
    }
    input_receipt = json.loads(
        (directory / "attempt-00/input-receipt.json").read_text()
    )
    bound(input_receipt["qp_inputs"])
    bound(input_receipt["objective_and_metric"])
    bound(input_receipt["source_cache"])
    certificate = json.loads((directory / "attempt-00/qp-certificate.json").read_text())
    assert certificate["certified"]
    with np.load(directory / "objective-and-adam.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    gradient = np.r_[values["gq"].ravel(), values["gp"]]
    with np.load(directory / "attempt-00/qp-inputs.npz") as saved:
        inputs = {key: saved[key].copy() for key in saved.files}
    selected = inputs["original_ids"].tolist()
    assert selected == [18514, 155249, 598977, 656201, 688235]
    assert input_receipt["row_order"] == [*selected, "negative_objective_gradient"]
    matrix = []
    for original_id in selected:
        gradient_path = bound(input_receipt["determinant_gradients"][str(original_id)])
        assert gradient_path == directory / f"gradient-{original_id}.npz"
        receipt_path = directory / f"gradient-{original_id}.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["q_comparison_passed"]
        assert receipt["pose_comparison_passed"]
        for key in ("shifted_relative_residual", "native_shifted_relative_residual"):
            assert receipt["sparse_adjoint"][key] <= 1e-7
        refs[f"gradient-{original_id}.npz"] = record(gradient_path)
        refs[f"gradient-{original_id}.json"] = record(receipt_path)
        with np.load(gradient_path) as saved:
            matrix.append(np.r_[saved["q_gradient"].ravel(), saved["pose_gradient"]])
    matrix = np.asarray([*matrix, -gradient])
    metric, x0 = values["diagonal_inverse_metric"], values["x0"]
    np.testing.assert_allclose(
        matrix @ x0, inputs["center_rows"], rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        (matrix * metric) @ matrix.T, inputs["gram"], rtol=1e-10, atol=1e-12
    )
    with np.load(bound(input_receipt["source_cache"])) as saved:
        lookup = {int(value): i for i, value in enumerate(saved["original_ids"])}
        indices = [lookup[value] for value in selected]
        lower = (
            1e-6
            - saved["old_j"][indices]
            - np.minimum(saved["residual_delta_j"][indices], 0)
        )
    target = float(inputs["original_descent_target"])
    np.testing.assert_array_equal(inputs["lower"], np.r_[lower, -target])
    np.testing.assert_array_equal(x0, 0.125 * np.r_[values["dq"].ravel(), values["dp"]])
    np.testing.assert_allclose(
        target, 0.1 * float(gradient @ x0), rtol=1e-12, atol=1e-15
    )
    assert summary["strain_absolute_limit"] == 0.01
    return {
        **values,
        "matrix": matrix,
        "lower": inputs["lower"],
        "protocol": protocol,
        "objective_gradient": gradient,
        "descent_target": target,
        "original_ids": selected,
    }, refs


def load_saved_seed(
    directory: Path, frozen: dict, frozen_refs: dict
) -> tuple[dict, dict]:
    """Bind original171 seed, target and certified projection without regeneration."""
    refs = {
        name: record(directory / name)
        for name in (
            "summary.json",
            "protocol.json",
            "bounded-input-receipt.json",
            "bounded-qp.json",
            "bounded-qp.npz",
            "joint-increment.npz",
            "predictor.json",
            "seed.npz",
        )
    }
    summary = json.loads((directory / "summary.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    assert summary["status"] == "diagnostic_failed"
    assert summary["failure"]["stage"] == "single_actual_seed"
    assert (
        summary["failure"]["message"]
        == "Actual seed violates original-positive or affine positivity; frozen diagnostic stops"
    )
    assert summary["source_hashes_unchanged"]
    assert protocol["source_refs"] == frozen["protocol"]["source_refs"]
    assert protocol["calibration_refs"] == frozen["protocol"]["calibration_refs"]
    assert protocol["frozen_projection_refs"] == frozen_refs
    assert (
        json.loads((directory / "bounded-input-receipt.json").read_text())
        == frozen_refs
    )
    for item in protocol["source_snapshot"].values():
        bound(item)
    certificate = json.loads((directory / "bounded-qp.json").read_text())
    assert certificate == summary["projection"]
    assert certificate["certified"]
    assert certificate["correction_metric_norm"] <= certificate["trust_norm_limit"]
    assert certificate["maximum_absolute_delta_q"] <= 0.01
    with np.load(directory / "bounded-qp.npz") as saved:
        increment = saved["increment"].copy()
    with np.load(directory / "joint-increment.npz") as saved:
        np.testing.assert_array_equal(increment, saved["actual_joint_increment"])
        np.testing.assert_array_equal(
            increment, np.r_[saved["delta_q"].ravel(), saved["delta_pose"]]
        )
    with np.load(directory / "seed.npz") as saved:
        values = {key: saved[key].copy() for key in saved.files}
    np.testing.assert_array_equal(
        values["activation_inv"],
        frozen["source_q"] + increment[:-6].reshape(frozen["source_q"].shape),
    )
    np.testing.assert_array_equal(
        values["pose_normalized"], frozen["source_pose"] + increment[-6:]
    )
    np.testing.assert_allclose(
        frozen["matrix"] @ increment - frozen["lower"],
        certificate["original_unit_slack"],
        rtol=0,
        atol=1e-12,
    )
    assert (
        float(frozen["objective_gradient"] @ increment)
        <= frozen["descent_target"] + 1e-12
    )
    return {
        **values,
        "increment": increment,
        "certificate": certificate,
        "predictor": json.loads((directory / "predictor.json").read_text()),
    }, refs


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
