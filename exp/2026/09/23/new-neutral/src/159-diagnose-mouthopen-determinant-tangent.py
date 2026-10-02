"""Calibrate frozen-state determinant tangents and probe one changed direction."""

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
from typing import Any

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
from mouthopen_block_optimizer import block_adam_update
from mouthopen_coupled_seed import _damped_equilibrium_tangent, prepare_coupled_seed
from mouthopen_pose_projection import determinant_ratio, project_pose_descent_direction
from mouthopen_runtime import install_mouthopen_hybrid_runtime
from mouthopen_tet_policy import exclude_fully_fixed_tetrahedra, geometry_metrics
from neutral_active_strain import install_active_strain
from reference_rebase import build_rebased_physics

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    source_run: Path = GROUP / "data/inverse-mouthopen-coupled-008"
    output_dir: Path = GROUP / "data/mouthopen-determinant-calibration-001"
    shifts: tuple[float, ...] = (0.001, 0.0001)
    epsilon: float = 0.0001
    determinant_margin: float = 1e-6
    activation_threshold: float = 0.05
    descent_fraction: float = 0.1
    candidate_alpha: float = 0.25
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


def determinant_derivative(
    reference: np.ndarray, tets: np.ndarray, u: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Differentiate the determinant using cofactors, including singular cells."""
    rest = reference[tets[:, 1:]] - reference[tets[:, :1]]
    x = reference + u[: len(reference)]
    edges = x[tets[:, 1:]] - x[tets[:, :1]]
    delta = direction[tets[:, 1:]] - direction[tets[:, :1]]
    rest_det = np.linalg.det(rest)
    assert np.all(rest_det > 0)
    return (
        np.einsum("ij,ij->i", delta[:, 0], np.cross(edges[:, 1], edges[:, 2]))
        + np.einsum("ij,ij->i", edges[:, 0], np.cross(delta[:, 1], edges[:, 2]))
        + np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], delta[:, 2]))
    ) / rest_det


def compact_tangent(receipt: dict) -> dict:
    sparse = receipt["sparse_solver"]
    return {
        key: receipt[key]
        for key in (
            "old_free_force_norm",
            "rhs_norm",
            "parameter_force_change_norm",
            "boundary_force_change_norm",
            "maximum_free_displacement_m",
            "maximum_boundary_displacement_m",
        )
    } | {
        "linear": {
            key: sparse[key]
            for key in (
                "shifted_relative_residual",
                "original_unshifted_relative_residual",
            )
        }
    }


def main(cfg: Config) -> None:
    assert cfg.shifts == (0.001, 0.0001)
    assert 0 < cfg.internal_atol <= 1e-9
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
    ids, tets, j0 = retained_ids[active], retained_tets[active], old_all_j[active]
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
            "schema": "mouthopen-frozen-determinant-calibration-v1",
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
            "direction_policy": "Fixed fresh-Adam proposal computed at baseline shift0.001, reused at both shifts to isolate tangent changes. Each shift uses its own fresh adjoint in the descent constraint.",
            "budget_scope": "600 seconds maximum per shift case, including adjoint, tangent probes, QP and optional candidate corrector; build and fresh post-failure output measurement excluded.",
            "inverse_adopted": False,
        },
    )
    np.savez_compressed(
        output / "source-determinants.npz",
        original_retained_ids=retained_ids,
        det_f=old_all_j,
        active_original_ids=ids,
    )
    results = []
    fixed_dq = fixed_dp = None
    for case_index, shift in enumerate(cfg.shifts):
        cherries.set_step(case_index)
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
        runtime.deadline = started + cfg.case_wall_seconds
        row = {
            "shift": shift,
            "status": "running",
            "inverse_adopted": False,
            "source_optimizer_steps": checkpoint["optimizer_steps"],
        }
        results.append(row)
        write_json(output / "summary.json", {"status": "running", "results": results})
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        stage = "fresh_adjoint"
        try:
            q_grad = q.detach().clone().requires_grad_()
            pose_grad = pose.detach().clone().requires_grad_()
            equilibrium = runtime.solve(
                materials(q_grad),
                physics.boundary(pose_grad * scales),
                old_u.detach().clone(),
                key="calibration",
            )
            torch.testing.assert_close(equilibrium, old_u, rtol=0, atol=5e-16)
            gq, gp = torch.autograd.grad(objective(equilibrium), (q_grad, pose_grad))
            row["adjoint"] = copy.deepcopy(runtime.last_sparse_adjoint)
            if fixed_dq is None:
                _, proposed_steps, fixed_dq, fixed_dp = block_adam_update(
                    moments,
                    (gq, gp),
                    q_optimizer_step=checkpoint["optimizer_steps"]["q"],
                    pose_optimizer_step=checkpoint["optimizer_steps"]["pose"],
                    update_q=True,
                    update_pose=True,
                    learning_rate=protocol["config"]["learning_rate"],
                    pose_learning_rate=protocol["config"]["pose_learning_rate"],
                )
                np.savez_compressed(
                    output / "fixed-adam-proposal.npz",
                    dq=fixed_dq.cpu(),
                    dp=fixed_dp.cpu(),
                )
                row["uncommitted_proposal_counters"] = proposed_steps
            assert fixed_dp is not None
            stage = "tangent_calibration"
            receipts = []
            row["tangent_receipts"] = receipts

            def tangent(
                value: torch.Tensor,
                next_pose: torch.Tensor,
                *,
                changed: bool,
                label: str,
                _runtime: Any = runtime,
                _shift: float = shift,
                _receipts: list = receipts,
            ) -> torch.Tensor:
                seed, receipt = _damped_equilibrium_tangent(
                    physics,
                    old_materials,
                    materials(value),
                    old_u,
                    physics.boundary(next_pose * scales),
                    deadline=_runtime.deadline,
                    predictor_relative_shift=_shift,
                    predictor_rtol=1e-7,
                    material_changed=changed,
                )
                _receipts.append({"label": label, **compact_tangent(receipt)})
                return seed

            q_full = tangent(q + fixed_dq, pose, changed=True, label="q_full")
            q_full_np = q_full.cpu().numpy()
            q_full_j = determinant_ratio(reference, tets, q_full_np)
            q_full_analytic = determinant_derivative(
                reference, tets, old_np, q_full_np - old_np
            )
            q_analytic, q_finite, pose_analytic, pose_finite = [], [], [], []
            for epsilon in (cfg.epsilon, 0.5 * cfg.epsilon):
                seed = tangent(
                    q + epsilon * fixed_dq,
                    pose,
                    changed=True,
                    label=f"q_epsilon_{epsilon:g}",
                )
                seed_np = seed.cpu().numpy()
                q_analytic.append(
                    determinant_derivative(
                        reference, tets, old_np, (seed_np - old_np) / epsilon
                    )
                )
                q_finite.append(
                    (determinant_ratio(reference, tets, seed_np) - j0) / epsilon
                )
                analytic_columns, finite_columns = [], []
                for coordinate in range(6):
                    probe = pose.clone()
                    probe[coordinate] += epsilon
                    seed = tangent(
                        q,
                        probe,
                        changed=False,
                        label=f"pose{coordinate}_epsilon_{epsilon:g}",
                    )
                    seed_np = seed.cpu().numpy()
                    analytic_columns.append(
                        determinant_derivative(
                            reference, tets, old_np, (seed_np - old_np) / epsilon
                        )
                    )
                    finite_columns.append(
                        (determinant_ratio(reference, tets, seed_np) - j0) / epsilon
                    )
                pose_analytic.append(np.column_stack(analytic_columns))
                pose_finite.append(np.column_stack(finite_columns))
            row["tangent_receipts"] = receipts
            np.savez_compressed(
                case / "calibration.npz",
                original_cell_ids=ids,
                j_old=j0,
                q_full_finite_j=q_full_j,
                q_full_analytic_dj=q_full_analytic,
                q_small_analytic_dj=np.asarray(q_analytic),
                q_small_finite_dj=np.asarray(q_finite),
                pose_analytic_jacobian=np.asarray(pose_analytic),
                pose_finite_jacobian=np.asarray(pose_finite),
                epsilon=np.array([cfg.epsilon, 0.5 * cfg.epsilon]),
            )
            row["critical_cells"] = [
                {
                    "original_id": int(ids[i]),
                    "old_j": float(j0[i]),
                    "q_full_finite_delta_j": float(q_full_j[i] - j0[i]),
                    "q_full_analytic_delta_j": float(q_full_analytic[i]),
                    "q_small_analytic_delta_j": float(q_analytic[-1][i]),
                    "pose_fd_half_spacing_max_change": float(
                        np.max(np.abs(pose_finite[1][i] - pose_finite[0][i]))
                    ),
                    "pose_half_fd_vs_analytic_max_error": float(
                        np.max(np.abs(pose_finite[1][i] - pose_analytic[1][i]))
                    ),
                }
                for i in range(len(ids))
            ]
            stage = "objective_aware_projection"
            strain_slope = float((gq * fixed_dq).sum())
            requested_slope = strain_slope + float((gp * fixed_dp).sum())
            assert requested_slope < 0, requested_slope
            matrix = pose_analytic[-1]
            lower = cfg.determinant_margin - j0 - q_analytic[-1]
            dp_np, qp = project_pose_descent_direction(
                fixed_dp.cpu().numpy(),
                matrix,
                lower,
                pose_gradient=gp.cpu().numpy(),
                strain_directional=strain_slope,
                descent_target=cfg.descent_fraction * requested_slope,
            )
            row["projection"] = qp
            projected = torch.as_tensor(dp_np, device=pose.device, dtype=pose.dtype)
            np.savez_compressed(
                case / "projected-direction.npz",
                original_cell_ids=ids,
                projected_dp=dp_np,
                matrix=matrix,
                lower=lower,
                pose_gradient=gp.cpu(),
                strain_directional=np.asarray(strain_slope),
                analytic_full_predicted_j=j0 + q_analytic[-1] + matrix @ dp_np,
                existing_geometry_full_predicted_j=q_full_j + pose_finite[0] @ dp_np,
            )
            row["status"] = "calibrated_without_candidate"
            if case_index == len(cfg.shifts) - 1:
                stage = "changed_direction_candidate"
                alpha = cfg.candidate_alpha
                next_q, next_pose = q + alpha * fixed_dq, pose + alpha * projected
                candidate = {"alpha": alpha, "adopted": False, "success": False}
                row["candidate"] = candidate
                seed, seed_receipt = prepare_coupled_seed(
                    physics,
                    materials,
                    q,
                    next_q,
                    pose * scales,
                    next_pose * scales,
                    old_u,
                    case / "candidate-seed",
                    deadline=runtime.deadline,
                    predictor_relative_shift=shift,
                    predictor_rtol=1e-7,
                )
                candidate["seed_receipt"] = seed_receipt
                seed_geometry = geometry_metrics(physics, seed)
                candidate["seed_geometry"] = seed_geometry
                seed_ok = (
                    seed_geometry["inverted_tetrahedra"] <= 100
                    and seed_geometry["inverted_rest_volume_fraction"] <= 0.0001
                )
                predicted_j = j0 + alpha * (q_analytic[-1] + matrix @ dp_np)
                corrected = seed
                solve_failed = None
                if seed_ok:
                    try:
                        corrected = runtime.primal(
                            materials(next_q),
                            physics.boundary(next_pose * scales),
                            seed,
                        )
                    except ForwardConvergenceError as error:
                        solve_failed = {"message": str(error), "receipt": error.receipt}
                        assert hasattr(runtime, "last_failed_displacement")
                        corrected = runtime.last_failed_displacement.detach().clone()
                    candidate["forward"] = copy.deepcopy(runtime.last_forward)
                else:
                    solve_failed = {
                        "message": "Candidate seed exceeds unchanged inversion allowance; no corrector attempted"
                    }
                runtime.deadline = None
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
                loss = float(objective(corrected))
                contact_ok = (
                    contact["receipt"]["contact_numerically_valid"]
                    and contact["no_intersections"]
                    and contact["minimum_active_gap_at_least_buffer"]
                )
                gates = (
                    force <= 1e-8
                    and geometry["inverted_tetrahedra"] <= 100
                    and geometry["inverted_rest_volume_fraction"] <= 0.0001
                    and contact_ok
                )
                candidate.update(
                    failure=solve_failed,
                    raw_force=force,
                    contact=contact,
                    geometry=geometry,
                    loss=loss,
                    source_loss=float(objective(old_u)),
                    true_objective_decrease=loss < float(objective(old_u)),
                    original_acceptance_gates_met=gates,
                    internal_force_target_met=force <= cfg.internal_atol,
                    success=solve_failed is None
                    and gates
                    and force <= cfg.internal_atol
                    and loss < float(objective(old_u)),
                )
                corrected_np = corrected.detach().cpu().numpy()
                corrected_all_j = determinant_ratio(
                    reference, retained_tets, corrected_np
                )
                np.savez_compressed(
                    case / "candidate.npz",
                    displacement_m=corrected_np,
                    activation_inv=next_q.cpu(),
                    pose_normalized=next_pose.cpu(),
                    pose_rad_m=(next_pose * scales).cpu(),
                    active_cell_ids=active_ids,
                    original_retained_ids=retained_ids,
                    corrected_det_f=corrected_all_j,
                    active_original_ids=ids,
                    predicted_active_j=predicted_j,
                    seed_active_j=determinant_ratio(
                        reference, tets, seed.cpu().numpy()
                    ),
                    corrected_active_j=corrected_all_j[active],
                    newly_inverted_original_ids=retained_ids[
                        (old_all_j > 0) & (corrected_all_j <= 0)
                    ],
                )
                candidate["newly_inverted_original_ids"] = retained_ids[
                    (old_all_j > 0) & (corrected_all_j <= 0)
                ].tolist()
                row["status"] = (
                    "candidate_passed_diagnostic"
                    if candidate["success"]
                    else "candidate_rejected_diagnostic"
                )
        except (ForwardConvergenceError, RuntimeError, AssertionError) as error:
            row["status"] = "case_failed"
            row["failure"] = {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
            if isinstance(error, ForwardConvergenceError):
                row["failure"]["receipt"] = error.receipt
            LOG.exception("Calibration case failed at %s", stage)
        finally:
            runtime.deadline = None
            model.set_materials(old_materials)
            model.dof_map.fixed_values = fixed.detach().clone()
            row["elapsed_seconds"] = time.perf_counter() - started
            for name, item in refs.items():
                assert record(source / name) == item, name
            for original, value in zip(checkpoint["moments"], moments, strict=True):
                torch.testing.assert_close(original, value.cpu(), rtol=0, atol=0)
            write_json(case / "result.json", row)
            write_json(
                output / "summary.json", {"status": "running", "results": results}
            )
        if row["status"] == "case_failed":
            break
    write_json(
        output / "summary.json",
        {
            "status": "finished_diagnostic",
            "results": results,
            "inverse_adopted": False,
            "source_hashes_unchanged": True,
            "diagnostic_complete": len(results) == len(cfg.shifts)
            and all(r["status"] != "case_failed" for r in results),
        },
    )
    cherries.log_output(output / "protocol.json")
    cherries.log_output(output / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
