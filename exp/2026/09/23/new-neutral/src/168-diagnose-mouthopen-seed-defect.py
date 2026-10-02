"""Calibrate frozen seed predictions using bounded absolute-defect corrections."""

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
from mouthopen_coupled_seed import _damped_equilibrium_tangent, _mandible_arc_radius
from mouthopen_pose_projection import (
    determinant_ratio,
    project_pose_descent_direction,
)
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-012"
    output_dir: Path = GROUP / "data/mouthopen-seed-defect-001"
    maximum_seed_attempts: int = 4
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
    assert 4 <= cfg.maximum_seed_attempts <= 12
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
        dq_np = saved["dq"].copy()
    calibration_refs = {
        "model_summary": record(model_dir / "summary.json"),
        "model_source_state": record(model_dir / "source-and-response.npz"),
        "coefficients": record(cache_path),
        "trial_summary": record(trial_dir / "summary.json"),
        "trial_input_receipt": record(trial_dir / "input-receipt.json"),
        "trial_constraints": record(trial_dir / "constraints-00.npz"),
        "trial_predictions": record(trial_dir / "predictions.npz"),
    }
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
    ids = retained_ids[active]
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
    write_json(
        output / "protocol.json",
        {
            "schema": "mouthopen-frozen-seed-defect-v1",
            "config": cfg.model_dump(mode="json"),
            "source_refs": refs,
            "calibration_refs": calibration_refs,
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
            "direction_policy": "Freeze audited012 attempt2 dq, pose gradient, requested pose and slopes. Use the declared maximum_seed_attempts; latest absolute measured seed defect offsets both seed and affine forecasts. No accumulated guard or alpha division. No residual response is added to the physical seed.",
            "budget_scope": "600 seconds for the declared bounded seed-oracle attempts, certified QPs, and at most one nonlinear corrector. Physical rebuild is excluded; serialization is inside the wall budget. Fresh post-failure metrics may finish after exhaustion, but no new solve may start.",
            "inverse_adopted": False,
        },
    )
    np.savez_compressed(
        output / "source-determinants.npz",
        original_retained_ids=retained_ids,
        det_f=old_all_j,
        active_original_ids=ids,
    )
    with np.load(cache_path) as saved:
        np.testing.assert_array_equal(saved["original_ids"], retained_ids)
        np.testing.assert_array_equal(saved["old_j"], old_all_j)
        q_delta = saved["q_delta_j"].copy()
        matrix = saved["pose_jacobian"].copy()
        residual_delta = saved["residual_delta_j"].copy()
        gp = saved["pose_gradient"].copy()
        requested_dp = saved["requested_dp"].copy()
        strain_slope = float(saved["strain_slope"])
        original_target = float(saved["original_target"])
    assert dq_np.shape == tuple(q.shape)
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
        "source_optimizer_steps": checkpoint["optimizer_steps"],
        "attempts": [],
    }
    write_json(output / "summary.json", row)
    stage = "initial_seed_proposal"
    alpha = cfg.candidate_alpha
    dq = torch.as_tensor(dq_np, device=q.device, dtype=q.dtype)
    next_q = q + alpha * dq
    positive = old_all_j > 0
    defect = np.zeros_like(old_all_j)
    admitted = False

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            message = "Declared seed calibration wall budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    def save_state(
        path: Path,
        u: torch.Tensor,
        next_pose: torch.Tensor,
        base_prediction: np.ndarray,
        used_defect: np.ndarray,
    ) -> np.ndarray:
        value = u.detach().cpu().numpy()
        j = determinant_ratio(reference, retained_tets, value)
        np.savez_compressed(
            path,
            displacement_m=value,
            activation_inv=next_q.cpu(),
            pose_normalized=next_pose.cpu(),
            pose_rad_m=(next_pose * scales).cpu(),
            active_cell_ids=active_ids,
            original_retained_ids=retained_ids,
            old_det_f=old_all_j,
            det_f=j,
            original_linear_seed_J=base_prediction,
            calibrated_seed_prediction_J=base_prediction + used_defect,
            calibrated_affine_prediction_J=base_prediction
            + used_defect
            + residual_delta,
            used_absolute_defect=used_defect,
            newly_inverted_original_ids=retained_ids[(old_all_j > 0) & (j <= 0)],
            sign_changed_original_ids=retained_ids[(old_all_j > 0) != (j > 0)],
        )
        return j

    try:
        for attempt in range(cfg.maximum_seed_attempts):
            check_budget()
            restore_source()
            dest = output / f"attempt-{attempt:02d}"
            dest.mkdir()
            item = {
                "attempt": attempt,
                "status": "running",
                "inverse_adopted": False,
                "qp_passes": [],
            }
            row["attempts"].append(item)
            np.savez_compressed(
                dest / "absolute-defect-input.npz",
                original_retained_ids=retained_ids,
                absolute_defect=defect,
            )
            seed_intercept = old_all_j + alpha * q_delta + defect
            affine_intercept = seed_intercept + residual_delta
            lower = (
                cfg.determinant_margin
                - old_all_j
                - alpha * q_delta
                - np.minimum(residual_delta, 0)
                - defect
            )
            active = positive & (
                (old_all_j <= cfg.activation_threshold)
                | (old_all_j + residual_delta + defect <= cfg.activation_threshold)
                | (seed_intercept <= cfg.activation_threshold)
                | (affine_intercept <= cfg.activation_threshold)
            )
            active[np.flatnonzero(positive)[np.argmax(lower[positive])]] = True
            stage = f"attempt{attempt}_certified_qp"
            for qp_pass in range(8):
                check_budget()
                qr = {"pass": qp_pass, "witness": {}}
                item["qp_passes"].append(qr)
                np.savez_compressed(
                    dest / f"qp-inputs-{qp_pass:02d}.npz",
                    original_cell_ids=retained_ids[active],
                    matrix=matrix[active],
                    lower=lower[active],
                    requested_increment=alpha * requested_dp,
                    pose_gradient=gp,
                    strain_increment_slope=np.asarray(alpha * strain_slope),
                    original_increment_target=np.asarray(alpha * original_target),
                )
                try:
                    increment_np, projection = project_pose_descent_direction(
                        alpha * requested_dp,
                        matrix[active],
                        lower[active],
                        pose_gradient=gp,
                        strain_directional=alpha * strain_slope,
                        descent_target=alpha * original_target,
                        feasible_witness=True,
                        witness_policy="attainable_optimum",
                        witness_receipt=qr["witness"],
                    )
                finally:
                    write_json(dest / f"qp-receipt-{qp_pass:02d}.json", qr)
                base_prediction = old_all_j + alpha * q_delta + matrix @ increment_np
                full_slack = (
                    base_prediction
                    + defect
                    + np.minimum(residual_delta, 0)
                    - cfg.determinant_margin
                )
                violating = positive & (full_slack < -1e-10)
                qr.update(
                    projection=projection,
                    minimum_full_positive_slack=float(full_slack[positive].min()),
                    violating_original_ids=retained_ids[violating].tolist(),
                )
                write_json(dest / f"qp-receipt-{qp_pass:02d}.json", qr)
                if not violating.any():
                    break
                assert np.any(violating & ~active)
                active |= violating
            else:
                message = "Full-positive constraint closure exceeded eight passes"
                raise RuntimeError(message)  # noqa: TRY301
            if attempt == 0:
                np.testing.assert_allclose(
                    increment_np,
                    cached_trial["actual_pose_increment"],
                    rtol=0,
                    atol=1e-12,
                )
            actual_slope = float(alpha * strain_slope + gp @ increment_np)
            assert actual_slope < 0
            assert (
                actual_slope
                <= projection["descent"]["joint_directional_upper_bound"] + 1e-11
            )
            increment = torch.as_tensor(
                increment_np, device=pose.device, dtype=pose.dtype
            )
            next_pose = pose + increment
            item.update(
                actual_pose_increment=increment_np.tolist(),
                actual_joint_increment_slope=actual_slope,
                chosen_increment_target=projection["descent"][
                    "joint_directional_upper_bound"
                ],
                full_positive_cells_checked=int(positive.sum()),
            )
            stage = f"attempt{attempt}_production_seed_oracle"
            check_budget()
            # Identical predictor and rigid-arc CCD operations to
            # prepare_coupled_seed, split only to persist raw u before admission.
            seed, predictor = _damped_equilibrium_tangent(
                physics,
                old_materials,
                materials(next_q),
                old_u,
                physics.boundary(next_pose * scales),
                deadline=deadline,
                predictor_relative_shift=cfg.relative_shift,
                predictor_rtol=1e-7,
                material_changed=not torch.equal(q, next_q),
            )
            item["predictor"] = predictor
            seed_j = save_state(
                dest / "seed.npz", seed, next_pose, base_prediction, defect
            )
            measured_defect = seed_j - base_prediction
            np.savez_compressed(
                dest / "measured-absolute-defect.npz",
                original_retained_ids=retained_ids,
                absolute_defect=measured_defect,
            )
            old_physical, new_physical = (
                (pose * scales).cpu().numpy(),
                (next_pose * scales).cpu().numpy(),
            )
            angle = float(
                (
                    Rotation.from_rotvec(new_physical[:3])
                    * Rotation.from_rotvec(old_physical[:3]).inv()
                ).magnitude()
            )
            assert 0 <= angle <= math.pi
            pivot = torch.as_tensor(
                physics.pivot_t, device=old_u.device, dtype=old_u.dtype
            )
            radius, _ = _mandible_arc_radius(
                physics, collision, pivot, fixed, physics.boundary(next_pose * scales)
            )
            motion = audit_coupled_motion(
                collision,
                old_u,
                seed,
                rotation_margin_m=rotation_sagitta(radius, angle),
            )
            seed_geometry = geometry_metrics(physics, seed)
            observed_slack = (
                seed_j + np.minimum(residual_delta, 0) - cfg.determinant_margin
            )
            admitted = bool(
                motion["admitted"]
                and seed_geometry["inverted_tetrahedra"] <= 100
                and seed_geometry["inverted_rest_volume_fraction"] <= 1e-4
                and observed_slack[positive].min() >= -1e-10
            )
            item.update(
                status="seed_admitted" if admitted else "seed_rejected",
                motion_audit=motion,
                geometry=seed_geometry,
                minimum_observed_seed_J=float(seed_j[positive].min()),
                minimum_observed_affine_J=float(
                    (seed_j + residual_delta)[positive].min()
                ),
                minimum_observed_model_slack=float(observed_slack[positive].min()),
                newly_inverted_original_ids=retained_ids[
                    (old_all_j > 0) & (seed_j <= 0)
                ].tolist(),
                model_margin_violating_original_ids=retained_ids[
                    positive & (observed_slack < -1e-10)
                ].tolist(),
                maximum_absolute_seed_defect=float(
                    np.max(np.abs(measured_defect[positive]))
                ),
            )
            write_json(dest / "summary.json", item)
            write_json(output / "summary.json", row)
            if admitted:
                break
            defect = (
                measured_defect.copy()
            )  # Latest absolute defect; never accumulate or divide by alpha.
        if not admitted:
            message = f"No admitted seed after {cfg.maximum_seed_attempts} declared attempts; no nonlinear corrector run"
            raise RuntimeError(message)  # noqa: TRY301
        stage = "single_nonlinear_corrector"
        check_budget()
        if hasattr(runtime, "last_failed_displacement"):
            del runtime.last_failed_displacement
        try:
            corrected = runtime.primal(
                materials(next_q), physics.boundary(next_pose * scales), seed
            )
        except ForwardConvergenceError as error:
            row["corrector_failure"] = {"message": str(error), "receipt": error.receipt}
            assert hasattr(runtime, "last_failed_displacement")
            corrected = runtime.last_failed_displacement.detach().clone()
        corrected_j = save_state(
            output / "corrected.npz", corrected, next_pose, base_prediction, defect
        )
        row["forward"] = copy.deepcopy(runtime.last_forward)
        runtime.deadline = None
        stage = "fresh_saved_candidate_metrics"
        model.set_materials(materials(next_q))
        model.dof_map.fixed_values = (
            physics.boundary(next_pose * scales).detach().clone()
        )
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
        contact_ok = (
            contact["receipt"]["contact_numerically_valid"]
            and contact["no_intersections"]
            and contact["minimum_active_gap_at_least_buffer"]
        )
        gates = (
            force <= 1e-8
            and geometry["inverted_tetrahedra"] <= 100
            and geometry["inverted_rest_volume_fraction"] <= 1e-4
            and contact_ok
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
            true_objective_decrease=loss < old_loss,
            original_acceptance_gates_met=gates,
            internal_force_target_met=force <= cfg.internal_atol,
            newly_inverted_original_ids=retained_ids[
                (old_all_j > 0) & (corrected_j <= 0)
            ].tolist(),
        )
        check_budget()
    except (ForwardConvergenceError, RuntimeError, AssertionError) as error:
        row["status"] = "diagnostic_failed"
        row["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ForwardConvergenceError):
            row["failure"]["receipt"] = error.receipt
        LOG.exception("Seed calibration failed at %s", stage)
    finally:
        runtime.deadline = None
        restore_source()
        for name, reference_item in refs.items():
            assert record(source / name) == reference_item, name
        for reference_item in calibration_refs.values():
            bound(reference_item)
        for original, value in zip(checkpoint["moments"], moments, strict=True):
            torch.testing.assert_close(original, value.cpu(), rtol=0, atol=0)
        row["source_hashes_unchanged"] = True
        row["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", row)
        cherries.log_output(output / "protocol.json")
        cherries.log_output(output / "summary.json")
    assert row["status"] != "diagnostic_failed", row


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
