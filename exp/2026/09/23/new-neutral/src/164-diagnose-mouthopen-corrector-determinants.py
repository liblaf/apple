"""Measure determinant prediction, residual response, and nonlinear correction at a frozen state."""

# ruff: noqa: B023, C901, E402, PLR0912, PLR0915, SLF001
# Loop-local diagnostic functions are invoked synchronously and never escape a case.
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
from liblaf.apple.inverse._diff_forward import _AdjointProblem

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
from mouthopen_block_optimizer import block_adam_update
from mouthopen_coupled_seed import _damped_equilibrium_tangent, _mandible_arc_radius
from mouthopen_pose_projection import (
    determinant_directional_derivative,
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
    output_dir: Path = GROUP / "data/mouthopen-corrector-determinants-001"
    relative_shifts: tuple[float, float] = (1e-4, 1e-5)
    epsilon: float = 0.00005
    determinant_margin: float = 1e-6
    activation_threshold: float = 0.05
    descent_fraction: float = 0.1
    candidate_alpha: float = 0.125
    baseline_tiny_alpha: float = 1.9073486328125e-6
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
    assert cfg.relative_shifts == (1e-4, 1e-5)
    assert cfg.baseline_tiny_alpha == 1.9073486328125e-6
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
    ids, j0 = retained_ids[active], old_all_j[active]
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
            "schema": "mouthopen-frozen-corrector-determinants-v1",
            "config": cfg.model_dump(mode="json"),
            "source_refs": refs,
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
            "direction_policy": "Fresh source-Adam proposal at each shift from identical saved moments; attainable-optimum projection. Baseline dq is also probed unchanged at both shifts. Residual-only response is never included in derivative probes or adopted.",
            "budget_scope": "600 seconds per shift for adjoint, derivative probes, residual response, LP/QP, and candidate corrections. Physical rebuild is excluded; serialization is inside the wall budget. Fresh post-failure metrics may finish after exhaustion, but no new solve may start.",
            "inverse_adopted": False,
        },
    )
    np.savez_compressed(
        output / "source-determinants.npz",
        original_retained_ids=retained_ids,
        det_f=old_all_j,
        active_original_ids=ids,
    )
    cases = []
    overall = {"status": "running", "inverse_adopted": False, "cases": cases}
    write_json(output / "summary.json", overall)
    common_dq = None
    for shift in cfg.relative_shifts:
        case = output / f"shift-{shift:g}"
        case.mkdir()
        runtime = install_mouthopen_hybrid_runtime(
            physics,
            forward_atol=cfg.internal_atol,
            adjoint_rtol=1e-7,
            adjoint_relative_shift=shift,
            newton_max_steps=3000,
            max_step_norm_m=0.5 * mean_rest_edge_length(model, physics.points),
            fixed_stiffness_mpa=kappa,
        )
        started = time.perf_counter()
        deadline = started + cfg.case_wall_seconds
        runtime.deadline = deadline
        row = {
            "status": "running",
            "relative_shift": shift,
            "inverse_adopted": False,
            "source_optimizer_steps": checkpoint["optimizer_steps"],
        }
        cases.append(row)
        write_json(case / "summary.json", row)
        stage = "fresh_adjoint"

        def check_budget() -> None:
            if time.perf_counter() >= deadline:
                message = "Declared diagnostic case budget exhausted"
                raise ForwardConvergenceError(message)

        def restore_source() -> None:
            model.set_materials(old_materials)
            model.dof_map.fixed_values = fixed.detach().clone()

        def full_derivative(delta: np.ndarray) -> np.ndarray:
            return determinant_directional_derivative(
                reference, retained_tets, old_np, delta
            )

        try:
            restore_source()
            q_grad = q.detach().clone().requires_grad_()
            pose_grad = pose.detach().clone().requires_grad_()
            equilibrium = runtime.solve(
                materials(q_grad),
                physics.boundary(pose_grad * scales),
                old_u.detach().clone(),
                key=f"corrector_determinants_{shift:g}",
            )
            torch.testing.assert_close(equilibrium, old_u, rtol=0, atol=5e-16)
            gq, gp = torch.autograd.grad(objective(equilibrium), (q_grad, pose_grad))
            row["adjoint"] = copy.deepcopy(runtime.last_sparse_adjoint)
            _, proposed_steps, dq, requested_dp = block_adam_update(
                moments,
                (gq, gp),
                q_optimizer_step=checkpoint["optimizer_steps"]["q"],
                pose_optimizer_step=checkpoint["optimizer_steps"]["pose"],
                update_q=True,
                update_pose=True,
                learning_rate=protocol["config"]["learning_rate"],
                pose_learning_rate=protocol["config"]["pose_learning_rate"],
            )
            row["uncommitted_proposal_counters"] = proposed_steps
            if common_dq is None:
                common_dq = dq.detach().clone()
                np.savez_compressed(
                    output / "common-direction.npz", baseline_dq=common_dq.cpu()
                )
            strain_slope = float((gq * dq).sum())
            requested_slope = strain_slope + float((gp * requested_dp).sum())
            assert requested_slope < 0
            target_slope = cfg.descent_fraction * requested_slope
            receipts = []
            row["tangent_receipts"] = receipts
            stage = "analytic_direction_tangents"

            def tangent(
                value: torch.Tensor,
                next_pose: torch.Tensor,
                *,
                changed: bool,
                label: str,
            ):
                check_budget()
                seed, receipt = _damped_equilibrium_tangent(
                    physics,
                    old_materials,
                    materials(value),
                    old_u,
                    physics.boundary(next_pose * scales),
                    deadline=deadline,
                    predictor_relative_shift=shift,
                    predictor_rtol=1e-7,
                    material_changed=changed,
                )
                receipts.append({"label": label, **copy.deepcopy(receipt)})
                delta = (seed.cpu().numpy() - old_np) / cfg.epsilon
                return full_derivative(delta)

            q_delta_all = tangent(
                q + cfg.epsilon * dq, pose, changed=True, label="fresh_q"
            )
            common_q_delta_all = (
                q_delta_all
                if shift == cfg.relative_shifts[0]
                else tangent(
                    q + cfg.epsilon * common_dq,
                    pose,
                    changed=True,
                    label="common_baseline_q",
                )
            )
            matrix_all = np.empty((len(retained_ids), 6))
            for coordinate in range(6):
                probe = pose.clone()
                probe[coordinate] += cfg.epsilon
                matrix_all[:, coordinate] = tangent(
                    q, probe, changed=False, label=f"pose{coordinate}"
                )
            matrix, q_delta = matrix_all[active], q_delta_all[active]
            stage = "separate_residual_response"
            restore_source()
            check_budget()
            state = model.State(u=old_u.detach().clone())
            state.collision = collision.state_at(state.u)
            residual = model.dof_map.to_free_grad(model.grad(state))
            problem = _AdjointProblem(b=-residual, model=model, model_state=state)
            solution = runtime.solver.solve(problem, torch.zeros_like(residual))
            residual_receipt = copy.deepcopy(runtime.last_sparse_adjoint)
            assert residual_receipt["shifted_relative_residual"] <= 1e-7
            assert residual_receipt["native_shifted_relative_residual"] <= 1e-7
            du0 = torch.zeros_like(old_u)
            du0.flatten()[model.dof_map.free_indices] = solution.params.detach()
            assert torch.isfinite(du0).all()
            assert not torch.count_nonzero(du0.flatten()[model.dof_map.fixed_indices])
            du0_np = du0.cpu().numpy()
            residual_delta_j = full_derivative(du0_np)
            row["residual_response"] = {
                "equation": "(H_old + shift I) du0_free = -old_free_force; du0_fixed = 0",
                "linear_model_only": True,
                "controls_unchanged": True,
                "accepted_equilibrium": False,
                "used_in_direction_probes": False,
                "used_in_candidate_seeds": False,
                "old_force_norm": float(torch.linalg.vector_norm(residual)),
                "maximum_displacement_m": float(np.max(np.linalg.norm(du0_np, axis=1))),
                "solver": residual_receipt,
            }
            np.savez_compressed(
                case / "residual-response.npz",
                displacement_response_m=du0_np,
                original_retained_ids=retained_ids,
                old_det_f=old_all_j,
                residual_delta_j=residual_delta_j,
                affine_intercept_j=old_all_j + residual_delta_j,
                displaced_geometry_j=determinant_ratio(
                    reference, retained_tets, old_np + du0_np
                ),
            )
            lower = cfg.determinant_margin - j0 - q_delta
            np.savez_compressed(
                case / "qp-inputs.npz",
                original_cell_ids=ids,
                j_old=j0,
                matrix=matrix,
                lower=lower,
                q_analytic_delta_j=q_delta,
                pose_gradient=gp.cpu(),
                strain_gradient=gq.cpu(),
                dq=dq.cpu(),
                requested_dp=requested_dp.cpu(),
                strain_directional=np.asarray(strain_slope),
                descent_target=np.asarray(target_slope),
            )
            np.savez_compressed(
                case / "full-tangents.npz",
                original_retained_ids=retained_ids,
                old_det_f=old_all_j,
                q_delta_j=q_delta_all,
                common_baseline_q_delta_j=common_q_delta_all,
                pose_jacobian=matrix_all,
            )
            stage = "attainable_optimum_qp"
            check_budget()
            row["witness"] = {}
            try:
                dp_np, qp = project_pose_descent_direction(
                    requested_dp.cpu().numpy(),
                    matrix,
                    lower,
                    pose_gradient=gp.cpu().numpy(),
                    strain_directional=strain_slope,
                    descent_target=target_slope,
                    feasible_witness=True,
                    witness_policy="attainable_optimum",
                    witness_receipt=row["witness"],
                )
            finally:
                write_json(case / "witness-receipt.json", row["witness"])
            row["projection"] = qp
            write_json(case / "qp-result.json", qp)
            projected = torch.as_tensor(dp_np, device=pose.device, dtype=pose.dtype)
            actual_slope = float((gq * dq).sum() + (gp * projected).sum())
            assert actual_slope < 0
            row["candidates"] = []

            def candidate_probe(alpha: float, label: str) -> None:
                check_budget()
                runtime.deadline = deadline
                restore_source()
                dest = case / label
                dest.mkdir()
                next_q, next_pose = q + alpha * dq, pose + alpha * projected
                predicted_j = old_all_j + alpha * (q_delta_all + matrix_all @ dp_np)
                candidate = {
                    "alpha": alpha,
                    "adopted": False,
                    "success": False,
                    "status": "running",
                    "label": label,
                }
                row["candidates"].append(candidate)
                corrected = None
                seed = None
                try:
                    seed, predictor = _damped_equilibrium_tangent(
                        physics,
                        old_materials,
                        materials(next_q),
                        old_u,
                        physics.boundary(next_pose * scales),
                        deadline=deadline,
                        predictor_relative_shift=shift,
                        predictor_rtol=1e-7,
                        material_changed=not torch.equal(q, next_q),
                    )
                    candidate["predictor"] = predictor
                    seed_np = seed.cpu().numpy()
                    seed_j = determinant_ratio(reference, retained_tets, seed_np)
                    np.savez_compressed(
                        dest / "seed.npz",
                        displacement_m=seed_np,
                        activation_inv=next_q.cpu(),
                        pose_normalized=next_pose.cpu(),
                        pose_rad_m=(next_pose * scales).cpu(),
                        original_retained_ids=retained_ids,
                        old_det_f=old_all_j,
                        predicted_det_f=predicted_j,
                        affine_predicted_det_f=predicted_j + residual_delta_j,
                        seed_det_f=seed_j,
                        newly_inverted_original_ids=retained_ids[
                            (old_all_j > 0) & (seed_j <= 0)
                        ],
                        sign_changed_original_ids=retained_ids[
                            (old_all_j > 0) != (seed_j > 0)
                        ],
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
                    radius, _ = _mandible_arc_radius(
                        physics,
                        collision,
                        torch.as_tensor(
                            physics.pivot_t, device=old_u.device, dtype=old_u.dtype
                        ),
                        fixed,
                        physics.boundary(next_pose * scales),
                    )
                    motion = audit_coupled_motion(
                        collision,
                        old_u,
                        seed,
                        rotation_margin_m=rotation_sagitta(radius, angle),
                    )
                    candidate["motion_audit"] = motion
                    seed_geometry = geometry_metrics(physics, seed)
                    candidate["seed_geometry"] = seed_geometry
                    if not motion["admitted"]:
                        candidate["status"] = "seed_rejected_ccd"
                        return
                    if (
                        seed_geometry["inverted_tetrahedra"] > 100
                        or seed_geometry["inverted_rest_volume_fraction"] > 1e-4
                    ):
                        candidate["status"] = "seed_rejected_inversion_allowance"
                        return
                    check_budget()
                    if hasattr(runtime, "last_failed_displacement"):
                        del runtime.last_failed_displacement
                    try:
                        corrected = runtime.primal(
                            materials(next_q),
                            physics.boundary(next_pose * scales),
                            seed,
                        )
                    except ForwardConvergenceError as error:
                        candidate["failure"] = {
                            "message": str(error),
                            "receipt": error.receipt,
                        }
                        if hasattr(runtime, "last_failed_displacement"):
                            corrected = (
                                runtime.last_failed_displacement.detach().clone()
                            )
                        else:
                            raise
                    candidate["forward"] = copy.deepcopy(runtime.last_forward)
                    corrected_np = corrected.cpu().numpy()
                    corrected_j = determinant_ratio(
                        reference, retained_tets, corrected_np
                    )
                    np.savez_compressed(
                        dest / "corrected.npz",
                        displacement_m=corrected_np,
                        activation_inv=next_q.cpu(),
                        pose_normalized=next_pose.cpu(),
                        pose_rad_m=(next_pose * scales).cpu(),
                        active_cell_ids=active_ids,
                        original_retained_ids=retained_ids,
                        old_det_f=old_all_j,
                        predicted_det_f=predicted_j,
                        affine_predicted_det_f=predicted_j + residual_delta_j,
                        seed_det_f=seed_j,
                        corrected_det_f=corrected_j,
                        newly_inverted_original_ids=retained_ids[
                            (old_all_j > 0) & (corrected_j <= 0)
                        ],
                        sign_changed_original_ids=retained_ids[
                            (old_all_j > 0) != (corrected_j > 0)
                        ],
                    )
                    runtime.deadline = None  # Fresh metrics on the saved raw candidate are not another solve.
                    model.set_materials(materials(next_q))
                    model.dof_map.fixed_values = (
                        physics.boundary(next_pose * scales).detach().clone()
                    )
                    measured = model.State(u=corrected.detach().clone())
                    measured.collision = collision.state_at(measured.u)
                    force = float(
                        torch.linalg.vector_norm(
                            FeasibleExpressionProblem(
                                model=model, collision_step_safety=0.9
                            ).grad(measured)
                        )
                    )
                    contact = runtime._contact_gate(measured)
                    geometry = geometry_metrics(physics, corrected)
                    loss, old_loss = (
                        float(objective(corrected)),
                        float(objective(old_u)),
                    )
                    armijo = old_loss + 1e-4 * alpha * actual_slope
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
                    candidate.update(
                        raw_force=force,
                        contact=contact,
                        geometry=geometry,
                        loss=loss,
                        source_loss=old_loss,
                        armijo_threshold=armijo,
                        actual_joint_slope=actual_slope,
                        armijo_passed=loss <= armijo,
                        true_objective_decrease=loss < old_loss,
                        original_acceptance_gates_met=gates,
                        internal_force_target_met=force <= cfg.internal_atol,
                        newly_inverted_original_ids=retained_ids[
                            (old_all_j > 0) & (corrected_j <= 0)
                        ].tolist(),
                        maximum_corrector_motion_m=float(
                            np.max(np.linalg.norm(corrected_np - seed_np, axis=1))
                        ),
                        success="failure" not in candidate
                        and gates
                        and force <= cfg.internal_atol
                        and loss < old_loss
                        and loss <= armijo,
                    )
                    candidate["status"] = (
                        "candidate_passed_diagnostic"
                        if candidate["success"]
                        else "candidate_rejected_diagnostic"
                    )
                except ForwardConvergenceError as error:
                    candidate.update(
                        status="candidate_failed",
                        failure={"message": str(error), "receipt": error.receipt},
                    )
                    raise
                finally:
                    runtime.deadline = deadline
                    restore_source()
                    write_json(dest / "summary.json", candidate)
                    write_json(case / "summary.json", row)

            # First capture the known seed-admissible tiny correction. It cannot
            # be skipped because the larger candidate was rejected at admission.
            if shift == cfg.relative_shifts[0]:
                stage = "baseline_tiny_candidate"
                candidate_probe(cfg.baseline_tiny_alpha, "tiny-replay")
            stage = "changed_candidate"
            candidate_probe(cfg.candidate_alpha, "candidate")
            check_budget()
            row["status"] = "diagnostic_completed"
        except (ForwardConvergenceError, RuntimeError, AssertionError) as error:
            row["status"] = "diagnostic_failed"
            row["failure"] = {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, ForwardConvergenceError):
                row["failure"]["receipt"] = error.receipt
            LOG.exception("Corrector determinant diagnostic failed at %s", stage)
        finally:
            runtime.deadline = None
            restore_source()
            row["elapsed_seconds"] = time.perf_counter() - started
            for name, item in refs.items():
                assert record(source / name) == item, name
            for original, value in zip(checkpoint["moments"], moments, strict=True):
                torch.testing.assert_close(original, value.cpu(), rtol=0, atol=0)
            row["source_hashes_unchanged"] = True
            write_json(case / "summary.json", row)
            write_json(output / "summary.json", overall)
        if row["status"] != "diagnostic_completed":
            break
    overall["status"] = (
        "diagnostic_completed"
        if len(cases) == len(cfg.relative_shifts)
        and all(x["status"] == "diagnostic_completed" for x in cases)
        else "diagnostic_failed"
    )
    write_json(output / "summary.json", overall)
    cherries.log_output(output / "protocol.json")
    cherries.log_output(output / "summary.json")
    assert overall["status"] == "diagnostic_completed", overall


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
