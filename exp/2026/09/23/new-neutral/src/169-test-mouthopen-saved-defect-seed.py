"""Measure predictor repeatability and physically test the saved defect-calibrated seed."""

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
)
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-012"
    output_dir: Path = GROUP / "data/mouthopen-saved-defect-seed-001"
    defect_run: Path = GROUP / "data/mouthopen-seed-defect-002"
    predictor_repeats: int = 3
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
    assert cfg.predictor_repeats == 3
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
    defect_run = cfg.defect_run.resolve()
    defect_protocol = json.loads((defect_run / "protocol.json").read_text())
    defect_summary = json.loads((defect_run / "summary.json").read_text())
    assert defect_protocol["source_refs"] == refs
    assert defect_protocol["calibration_refs"] == calibration_refs
    assert defect_summary["status"] == "diagnostic_failed"
    assert len(defect_summary["attempts"]) == 12
    assert defect_summary["failure"]["stage"] == "attempt11_production_seed_oracle"
    assert "No admitted seed after 12" in defect_summary["failure"]["message"]
    assert defect_protocol["config"]["relative_shift"] == cfg.relative_shift
    assert defect_protocol["config"]["candidate_alpha"] == cfg.candidate_alpha
    assert defect_protocol["internal_force_target"] == cfg.internal_atol
    for item in defect_protocol["source_snapshot"].values():
        bound(item)
    for item in defect_protocol["calibration_refs"].values():
        bound(item)
    saved_attempt_dir = defect_run / "attempt-11"
    saved_attempt = json.loads((saved_attempt_dir / "summary.json").read_text())
    assert saved_attempt == defect_summary["attempts"][-1]
    assert saved_attempt["attempt"] == 11
    assert saved_attempt["status"] == "seed_rejected"
    with np.load(saved_attempt_dir / "seed.npz") as saved:
        saved_seed_np = saved["displacement_m"].copy()
        saved_seed_j = saved["det_f"].copy()
        saved_target_q = saved["activation_inv"].copy()
        saved_target_pose = saved["pose_normalized"].copy()
        saved_target_pose_physical = saved["pose_rad_m"].copy()
        saved_ids = saved["original_retained_ids"].copy()
    expected_increment = np.asarray(saved_attempt["actual_pose_increment"])
    np.testing.assert_array_equal(
        saved_target_q,
        checkpoint["activation_inv"].numpy() + cfg.candidate_alpha * dq_np,
    )
    np.testing.assert_array_equal(
        saved_target_pose, checkpoint["pose_normalized"].numpy() + expected_increment
    )
    final_qp_path = sorted(saved_attempt_dir.glob("qp-receipt-*.json"))[-1]
    final_qp = json.loads(final_qp_path.read_text())
    np.testing.assert_array_equal(final_qp["witness"]["direction"], expected_increment)
    defect_refs = {
        "protocol": record(defect_run / "protocol.json"),
        "summary": record(defect_run / "summary.json"),
    }
    for path in sorted(saved_attempt_dir.iterdir()):
        if path.is_file():
            defect_refs[path.name] = record(path)
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
            "schema": "mouthopen-frozen-saved-defect-seed-v1",
            "config": cfg.model_dump(mode="json"),
            "source_refs": refs,
            "calibration_refs": calibration_refs,
            "defect_refs": defect_refs,
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
            "direction_policy": "Bind audited run012 controls, optimizer state and cached direction to diagnostic168 output002 attempt11. Measure three independent predictors at its identical target controls, then correct the original hash-bound saved seed once. Repeated predictor states are never adopted; no residual response is added to the physical seed.",
            "budget_scope": "600 seconds for fresh saved-seed admission, three identical-control predictor repeats, and one nonlinear corrector. Physical rebuild is excluded; serialization is inside the wall budget. Fresh post-failure metrics may finish after exhaustion, but no new solve may start.",
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
        residual_delta = saved["residual_delta_j"].copy()
        gp = saved["pose_gradient"].copy()
        strain_slope = float(saved["strain_slope"])
    np.testing.assert_array_equal(saved_ids, retained_ids)
    np.testing.assert_array_equal(
        determinant_ratio(reference, retained_tets, saved_seed_np), saved_seed_j
    )
    next_q = torch.as_tensor(saved_target_q, device=q.device, dtype=q.dtype)
    next_pose = torch.as_tensor(saved_target_pose, device=pose.device, dtype=pose.dtype)
    original_seed = torch.as_tensor(
        saved_seed_np, device=old_u.device, dtype=old_u.dtype
    ).clone()
    expected_fixed = physics.boundary(next_pose * scales).detach().clone()
    np.testing.assert_array_equal(
        (next_pose * scales).cpu().numpy(), saved_target_pose_physical
    )
    torch.testing.assert_close(
        original_seed.flatten()[model.dof_map.fixed_indices],
        expected_fixed,
        rtol=0,
        atol=0,
    )
    actual_slope = float(cfg.candidate_alpha * strain_slope + gp @ expected_increment)
    np.testing.assert_allclose(
        actual_slope, saved_attempt["actual_joint_increment_slope"], rtol=0, atol=1e-14
    )
    assert actual_slope < 0
    positive = old_all_j > 0
    observed_affine = saved_seed_j + residual_delta
    strict_slack = float(
        np.minimum(saved_seed_j, observed_affine)[positive].min()
        - cfg.determinant_margin
    )
    assert strict_slack < -1e-10
    assert np.all(saved_seed_j[positive] > 0)
    assert np.all(observed_affine[positive] > 0)
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
        "strict_model_margin": cfg.determinant_margin,
        "strict_model_margin_passed": False,
        "strict_model_margin_minimum_slack": strict_slack,
        "model_margin_failure_is_not_physical_policy_failure": True,
        "all_original_positive_cells_still_positive": True,
        "all_observed_affine_original_positive_cells_positive": True,
        "actual_joint_increment_slope": actual_slope,
        "repeats": [],
    }
    write_json(output / "summary.json", row)
    stage = "fresh_saved_seed_ccd_and_geometry"

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            message = "Declared saved-seed diagnostic wall budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    def save_state(path: Path, u: torch.Tensor) -> np.ndarray:
        value = u.detach().cpu().numpy()
        j = determinant_ratio(reference, retained_tets, value)
        np.savez_compressed(
            path,
            displacement_m=value,
            activation_inv=saved_target_q,
            pose_normalized=saved_target_pose,
            pose_rad_m=saved_target_pose_physical,
            active_cell_ids=active_ids,
            original_retained_ids=retained_ids,
            old_det_f=old_all_j,
            det_f=j,
            original_saved_seed_det_f=saved_seed_j,
            observed_affine_det_f=j + residual_delta,
            newly_inverted_original_ids=retained_ids[(old_all_j > 0) & (j <= 0)],
            sign_changed_original_ids=retained_ids[(old_all_j > 0) != (j > 0)],
        )
        return j

    try:
        check_budget()
        restore_source()
        angle = float(
            (
                Rotation.from_rotvec(saved_target_pose_physical[:3])
                * Rotation.from_rotvec((pose * scales).cpu().numpy()[:3]).inv()
            ).magnitude()
        )
        assert 0 <= angle <= math.pi
        pivot = torch.as_tensor(physics.pivot_t, device=old_u.device, dtype=old_u.dtype)
        radius, _ = _mandible_arc_radius(
            physics, collision, pivot, fixed, expected_fixed
        )
        motion = audit_coupled_motion(
            collision,
            old_u,
            original_seed,
            rotation_margin_m=rotation_sagitta(radius, angle),
        )
        seed_geometry = geometry_metrics(physics, original_seed)
        row.update(
            saved_seed_motion_audit=motion,
            saved_seed_geometry=seed_geometry,
            minimum_saved_seed_positive_J=float(saved_seed_j[positive].min()),
            minimum_saved_affine_positive_J=float(observed_affine[positive].min()),
        )
        save_state(output / "saved-seed.npz", original_seed)
        assert motion["admitted"]
        assert seed_geometry["inverted_tetrahedra"] <= 100
        assert seed_geometry["inverted_rest_volume_fraction"] <= 1e-4
        repeats = []
        for repeat in range(cfg.predictor_repeats):
            stage = f"independent_predictor_repeat{repeat}"
            check_budget()
            restore_source()
            candidate, receipt = _damped_equilibrium_tangent(
                physics,
                old_materials,
                materials(next_q),
                old_u,
                expected_fixed,
                deadline=deadline,
                predictor_relative_shift=cfg.relative_shift,
                predictor_rtol=1e-7,
                material_changed=not torch.equal(q, next_q),
            )
            torch.testing.assert_close(
                candidate.flatten()[model.dof_map.fixed_indices],
                expected_fixed,
                rtol=0,
                atol=0,
            )
            j = save_state(output / f"repeat-{repeat:02d}.npz", candidate)
            repeats.append(j)
            repeat_row = {
                "repeat": repeat,
                "predictor": receipt,
                "same_old_source_controls_and_displacement": True,
                "used_for_corrector": False,
                "maximum_displacement_difference_from_saved_m": float(
                    torch.linalg.vector_norm(candidate - original_seed, dim=1).max()
                ),
                "minimum_original_positive_J": float(j[positive].min()),
                "newly_inverted_original_ids": retained_ids[
                    positive & (j <= 0)
                ].tolist(),
            }
            row["repeats"].append(repeat_row)
            write_json(output / f"repeat-{repeat:02d}.json", repeat_row)
            write_json(output / "summary.json", row)
        repeated = np.asarray(repeats)
        minimum, maximum = repeated.min(axis=0), repeated.max(axis=0)
        spread = maximum - minimum
        np.savez_compressed(
            output / "repeat-variability.npz",
            original_retained_ids=retained_ids,
            original_saved_seed_J=saved_seed_j,
            repeated_J=repeated,
            minimum_J=minimum,
            maximum_J=maximum,
            range_J=spread,
            std_J=repeated.std(axis=0),
            maximum_absolute_saved_difference=np.max(
                abs(repeated - saved_seed_j), axis=0
            ),
        )
        local = positive & (
            (old_all_j <= cfg.activation_threshold)
            | (saved_seed_j <= cfg.activation_threshold)
            | (observed_affine <= cfg.activation_threshold)
        )
        row["local_repeat_variability"] = [
            {
                "original_id": int(retained_ids[i]),
                "saved_seed_J": float(saved_seed_j[i]),
                "minimum_repeat_J": float(minimum[i]),
                "maximum_repeat_J": float(maximum[i]),
                "repeat_range_J": float(spread[i]),
                "maximum_difference_from_saved_J": float(
                    np.max(abs(repeated[:, i] - saved_seed_j[i]))
                ),
            }
            for i in np.flatnonzero(local)
        ]
        row["maximum_repeat_range_original_positive_J"] = float(spread[positive].max())
        stage = "single_corrector_from_original_saved_seed"
        check_budget()
        restore_source()
        if hasattr(runtime, "last_failed_displacement"):
            del runtime.last_failed_displacement
        try:
            corrected = runtime.primal(
                materials(next_q), expected_fixed, original_seed.detach().clone()
            )
        except ForwardConvergenceError as error:
            row["corrector_failure"] = {"message": str(error), "receipt": error.receipt}
            assert hasattr(runtime, "last_failed_displacement")
            corrected = runtime.last_failed_displacement.detach().clone()
        corrected_j = save_state(output / "corrected.npz", corrected)
        row["forward"] = copy.deepcopy(runtime.last_forward)
        row["corrector_seed"] = defect_refs["seed.npz"]
        runtime.deadline = None
        stage = "fresh_saved_candidate_metrics"
        model.set_materials(materials(next_q))
        model.dof_map.fixed_values = expected_fixed.detach().clone()
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
                positive & (corrected_j <= 0)
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
        LOG.exception("Saved-seed diagnostic failed at %s", stage)
    finally:
        runtime.deadline = None
        restore_source()
        for name, item in refs.items():
            assert record(source / name) == item, name
        for item in calibration_refs.values():
            bound(item)
        for item in defect_refs.values():
            bound(item)
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
