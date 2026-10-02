"""Test one certified affine-residual pose proposal with unchanged physical gates."""

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

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(GROUP / "src"), str(SOLVERS), str(JOINT)]

from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_equilibrium import FeasibleExpressionProblem
from mesh_step_scale import mean_rest_edge_length
from mouthopen_coupled_seed import _damped_equilibrium_tangent, prepare_coupled_seed
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
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-011"
    output_dir: Path = GROUP / "data/mouthopen-affine-corrector-001"
    calibration_dir: Path = GROUP / "data/mouthopen-corrector-determinants-001"
    relative_shift: float = 1e-4
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
    assert cfg.relative_shift in (1e-4, 1e-5)
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
    assert checkpoint["optimizer_steps"] == {"q": 267, "pose": 267}
    assert checkpoint["iteration"] == summary["final"]["iteration"]
    assert checkpoint["optimizer_steps"] == summary["final"]["optimizer_steps"]
    with np.load(source / "endpoint.npz", allow_pickle=False) as saved:
        for key in ("activation_inv", "pose_rad_m", "displacement_m"):
            np.testing.assert_array_equal(saved[key], checkpoint[key].numpy())
        active_ids = saved["active_cell_ids"].copy()
    reference_path = bound(protocol["sources"]["reference_repair"])
    neutral_path = bound(protocol["sources"]["neutral_endpoint"])
    target_path = bound(protocol["sources"]["blendshapes"])
    calibration = cfg.calibration_dir.resolve()
    calibration_protocol = json.loads((calibration / "protocol.json").read_text())
    calibration_summary = json.loads((calibration / "summary.json").read_text())
    assert calibration_summary["status"] == "diagnostic_completed"
    assert calibration_protocol["source_refs"] == refs
    assert calibration_protocol["config"]["epsilon"] == cfg.epsilon == 5e-5
    assert calibration_protocol["internal_force_target"] == cfg.internal_atol
    assert calibration_protocol["config"]["descent_fraction"] == cfg.descent_fraction
    assert (
        calibration_protocol["config"]["activation_threshold"]
        == cfg.activation_threshold
    )
    for item in calibration_protocol["source_snapshot"].values():
        bound(item)
    cpu_path = calibration / "affine-proposal-cpu.json"
    cpu_receipt = json.loads(cpu_path.read_text())
    assert cpu_receipt["margin"] == cfg.determinant_margin
    assert bound(cpu_receipt["protocol"]) == calibration / "protocol.json"
    for item in cpu_receipt["source_helpers"].values():
        bound(item)
    matches = [
        row
        for row in cpu_receipt["cases"]
        if row["case"] == f"shift-{cfg.relative_shift:g}"
        and row["alpha"] == cfg.candidate_alpha
    ]
    assert len(matches) == 1
    cpu = matches[0]
    assert cpu["success"]
    assert cpu["all_positive_cells_checked"]
    for item in cpu["inputs"].values():
        bound(item)
    calibration_refs = {
        "protocol": record(calibration / "protocol.json"),
        "summary": record(calibration / "summary.json"),
        "cpu_affine_receipt": record(cpu_path),
        "common_direction": record(calibration / "common-direction.npz"),
        **cpu["inputs"],
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
            "schema": "mouthopen-frozen-affine-corrector-v1",
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
            "direction_policy": "Reuse completed164 selected-shift source-Adam dq and gradients; independently solve actual-pose-increment QP with seed and residual-affine constraints. No residual response is added to the production seed or adopted.",
            "budget_scope": "600 seconds for CPU affine QP, recorded predictor, production CCD seed, and one candidate corrector. Physical rebuild is excluded; serialization is inside the wall budget. Fresh post-failure metrics may finish after exhaustion, but no new solve may start.",
            "inverse_adopted": False,
        },
    )
    np.savez_compressed(
        output / "source-determinants.npz",
        original_retained_ids=retained_ids,
        det_f=old_all_j,
        active_original_ids=ids,
    )
    baseline_case = calibration / f"shift-{cfg.relative_shift:g}"
    selected_summary = json.loads((baseline_case / "summary.json").read_text())
    assert selected_summary["status"] == "diagnostic_completed"
    assert selected_summary["source_optimizer_steps"] == checkpoint["optimizer_steps"]
    assert selected_summary["relative_shift"] == cfg.relative_shift
    with np.load(baseline_case / "full-tangents.npz") as saved:
        np.testing.assert_array_equal(saved["original_retained_ids"], retained_ids)
        np.testing.assert_array_equal(saved["old_det_f"], old_all_j)
        q_delta = saved["q_delta_j"].copy()
        matrix = saved["pose_jacobian"].copy()
    with np.load(baseline_case / "residual-response.npz") as saved:
        np.testing.assert_array_equal(saved["original_retained_ids"], retained_ids)
        np.testing.assert_array_equal(saved["old_det_f"], old_all_j)
        residual_delta = saved["residual_delta_j"].copy()
    with np.load(baseline_case / "qp-inputs.npz") as saved:
        dq_np = saved["dq"].copy()
        requested_dp = saved["requested_dp"].copy()
        gp = saved["pose_gradient"].copy()
        gq = saved["strain_gradient"].copy()
        strain_slope = float(saved["strain_directional"])
        original_target = float(saved["descent_target"])
    if cfg.relative_shift == 1e-4:
        with np.load(calibration / "common-direction.npz") as saved:
            np.testing.assert_array_equal(saved["baseline_dq"], dq_np)
    assert dq_np.shape == gq.shape == tuple(q.shape)
    np.testing.assert_allclose(np.sum(gq * dq_np), strain_slope, rtol=1e-12, atol=1e-14)
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
        "calibration_refs": calibration_refs,
    }
    write_json(output / "summary.json", row)
    stage = "certified_affine_increment_qp"

    def check_budget() -> None:
        if time.perf_counter() >= deadline:
            message = "Declared diagnostic wall budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    try:
        check_budget()
        alpha = cfg.candidate_alpha
        positive = old_all_j > 0
        active = positive & (
            (old_all_j <= cfg.activation_threshold)
            | (old_all_j + residual_delta <= cfg.activation_threshold)
            | (old_all_j + alpha * q_delta <= cfg.activation_threshold)
            | (old_all_j + residual_delta + alpha * q_delta <= cfg.activation_threshold)
        )
        lower = (
            cfg.determinant_margin
            - old_all_j
            - np.minimum(residual_delta, 0)
            - alpha * q_delta
        )
        np.savez_compressed(
            output / "qp-inputs.npz",
            original_cell_ids=retained_ids[active],
            matrix=matrix[active],
            lower=lower[active],
            requested_increment=alpha * requested_dp,
            pose_gradient=gp,
            strain_increment_slope=np.asarray(alpha * strain_slope),
            original_increment_target=np.asarray(alpha * original_target),
        )
        row["witness"] = {}
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
                witness_receipt=row["witness"],
            )
        finally:
            write_json(output / "witness-receipt.json", row["witness"])
        predicted_seed = old_all_j + alpha * q_delta + matrix @ increment_np
        predicted_affine = predicted_seed + residual_delta
        minimum_slack = float(
            np.minimum(predicted_seed, predicted_affine)[positive].min()
            - cfg.determinant_margin
        )
        assert minimum_slack >= -1e-10
        np.testing.assert_allclose(
            increment_np, cpu["actual_pose_increment_normalized"], rtol=0, atol=1e-12
        )
        actual_slope = float(alpha * strain_slope + gp @ increment_np)
        assert actual_slope < 0
        assert (
            actual_slope
            <= projection["descent"]["joint_directional_upper_bound"] + 1e-11
        )
        row.update(
            projection=projection,
            all_positive_retained_cells_checked=int(positive.sum()),
            minimum_linear_prediction_slack=minimum_slack,
            actual_pose_increment_normalized=increment_np.tolist(),
            actual_joint_increment_slope=actual_slope,
        )
        write_json(output / "qp-result.json", row)
        dq = torch.as_tensor(dq_np, device=q.device, dtype=q.dtype)
        increment = torch.as_tensor(increment_np, device=pose.device, dtype=pose.dtype)
        next_q, next_pose = q + alpha * dq, pose + increment
        row["candidate_controls"] = {
            "alpha_q": alpha,
            "pose_increment_already_scaled": True,
        }

        def save_state(
            path: Path, u: torch.Tensor, *, seed_j: np.ndarray | None = None
        ) -> np.ndarray:
            u_np = u.detach().cpu().numpy()
            j = determinant_ratio(reference, retained_tets, u_np)
            arrays = {
                "displacement_m": u_np,
                "activation_inv": next_q.cpu().numpy(),
                "pose_normalized": next_pose.cpu().numpy(),
                "pose_rad_m": (next_pose * scales).cpu().numpy(),
                "active_cell_ids": active_ids,
                "original_retained_ids": retained_ids,
                "old_det_f": old_all_j,
                "predicted_seed_det_f": predicted_seed,
                "predicted_affine_det_f": predicted_affine,
                "det_f": j,
                "newly_inverted_original_ids": retained_ids[(old_all_j > 0) & (j <= 0)],
                "sign_changed_original_ids": retained_ids[(old_all_j > 0) != (j > 0)],
            }
            if seed_j is not None:
                arrays["seed_det_f"] = seed_j
            np.savez_compressed(path, **arrays)
            return j

        stage = "raw_seed_capture"
        check_budget()
        restore_source()
        # Record the same predictor before production CCD can raise. The
        # production helper runs its unchanged predictor/CCD path below.
        raw_seed, raw_predictor = _damped_equilibrium_tangent(
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
        save_state(output / "raw-predictor.npz", raw_seed)
        row["raw_predictor"] = raw_predictor
        stage = "production_ccd_seed"
        check_budget()
        restore_source()
        seed, seed_receipt = prepare_coupled_seed(
            physics,
            materials,
            q,
            next_q,
            pose * scales,
            next_pose * scales,
            old_u,
            output / "production-seed",
            deadline=deadline,
            predictor_relative_shift=cfg.relative_shift,
            predictor_rtol=1e-7,
        )
        seed_j = save_state(output / "seed.npz", seed)
        row["seed_receipt"] = seed_receipt
        row["raw_vs_production_seed_max_abs_error"] = float(
            (raw_seed - seed).abs().max()
        )
        row["raw_vs_production_seed_l2_error"] = float(
            torch.linalg.vector_norm(raw_seed - seed)
        )
        # Independent iterative solves need not return identical displacement.
        # Their native and sparse residual contracts certify the predictor;
        # physical admission uses the actual production CCD seed below.
        for predictor in (raw_predictor, seed_receipt["predictor"]):
            solver = predictor["sparse_solver"]
            assert solver["native_shifted_relative_residual"] <= 1e-7
            assert solver["shifted_relative_residual"] <= 1e-7
            assert predictor["predictor_relative_shift"] == cfg.relative_shift
        expected_fixed = physics.boundary(next_pose * scales)
        for value in (raw_seed, seed):
            torch.testing.assert_close(
                value.flatten()[model.dof_map.fixed_indices],
                expected_fixed,
                rtol=0,
                atol=0,
            )
        seed_geometry = geometry_metrics(physics, seed)
        row["seed_geometry"] = seed_geometry
        if (
            seed_geometry["inverted_tetrahedra"] > 100
            or seed_geometry["inverted_rest_volume_fraction"] > 1e-4
        ):
            row["status"] = "seed_rejected_inversion_allowance"
        else:
            stage = "production_corrector"
            check_budget()
            if hasattr(runtime, "last_failed_displacement"):
                del runtime.last_failed_displacement
            try:
                corrected = runtime.primal(
                    materials(next_q), physics.boundary(next_pose * scales), seed
                )
            except ForwardConvergenceError as error:
                row["corrector_failure"] = {
                    "message": str(error),
                    "receipt": error.receipt,
                }
                assert hasattr(runtime, "last_failed_displacement")
                corrected = runtime.last_failed_displacement.detach().clone()
            corrected_j = save_state(output / "corrected.npz", corrected, seed_j=seed_j)
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
                    FeasibleExpressionProblem(
                        model=model, collision_step_safety=0.9
                    ).grad(state)
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
                maximum_corrector_motion_m=float(
                    torch.linalg.vector_norm(corrected - seed, dim=1).max()
                ),
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
        LOG.exception("Affine corrector diagnostic failed at %s", stage)
    finally:
        runtime.deadline = None
        restore_source()
        for name, item in refs.items():
            assert record(source / name) == item, name
        for item in calibration_refs.values():
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
