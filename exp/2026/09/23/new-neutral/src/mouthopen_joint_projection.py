"""Build one frozen joint determinant-adjoint cache for bounded trial increments."""

# ruff: noqa: PLR0915, SLF001
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import write_json
from joint_coupled_predictor import clone_materials
from joint_equilibrium import ForwardConvergenceError
from mouthopen_coupled_seed import _assert_materials_equal, _damped_equilibrium_tangent
from mouthopen_joint_direction import (
    JointProjectionCache,
    project_joint_increment,
    record,
    save_joint_projection_cache,
)
from mouthopen_pose_projection import (
    determinant_directional_derivative,
    determinant_ratio,
)

from liblaf.apple.inverse._diff_forward import _AdjointProblem

__all__ = [
    "JointProjectionCache",
    "build_joint_projection_cache",
    "project_joint_increment",
]


def build_joint_projection_cache(
    physics: Any,
    materials: Any,
    physical_pose: Any,
    q: torch.Tensor,
    pose: torch.Tensor,
    u: torch.Tensor,
    dq: torch.Tensor,
    dp: torch.Tensor,
    gq: torch.Tensor,
    gp: torch.Tensor,
    proposed_moments: list[torch.Tensor],
    next_optimizer_steps: dict[str, int],
    output: Path,
    *,
    learning_rate: float,
    pose_learning_rate: float,
    deadline: float | None,
) -> JointProjectionCache:
    """Reuse runner Adam history and freeze one exact accepted source graph.

    The separate residual response changes only the determinant forecast. Actual
    seeds and corrected equilibria retain the runner's original physical gates.
    """
    epsilon, comparison_atol, comparison_rtol = 5e-5, 1e-7, 0.005
    output.mkdir(parents=True, exist_ok=False)
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    assert runtime.adjoint_relative_shift > 0
    assert 0 < runtime.tolerances["adjoint_rtol"] <= 1e-7
    assert learning_rate > 0
    assert pose_learning_rate > 0
    assert len(proposed_moments) == 4
    assert next_optimizer_steps["q"] >= 1
    assert next_optimizer_steps["pose"] >= 1
    old_materials = clone_materials(materials(q))
    original_materials = clone_materials(model.get_materials())
    original_fixed = model.dof_map.fixed_values.detach().clone()
    original_runtime_u = runtime.forward.state.u.detach().clone()
    original_runtime_collision = runtime.forward.state.collision
    original_deadline = runtime.deadline
    runtime.deadline = deadline
    fixed = physics.boundary(physical_pose(pose)).detach().clone()
    original_inputs = [
        value.detach().clone()
        for value in (q, pose, u, dq, dp, gq, gp, *proposed_moments)
    ]
    reference = np.asarray(physics.points)
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)
    tets = np.asarray(physics.base.tets)[retained_ids]
    old_np = u.detach().cpu().numpy()
    old_j = determinant_ratio(reference, tets, old_np)
    select = (old_j > 0) & ((old_j < 1e-4) | np.isin(retained_ids, [18514, 155249]))
    selected_indices = np.flatnonzero(select)
    selected_ids = retained_ids[select]
    receipt = {
        "status": "running",
        "selected_original_ids": selected_ids.tolist(),
        "selection": "original-positive J<1e-4 plus original-positive18514/155249; maximum16",
        "relative_shift": runtime.adjoint_relative_shift,
        "epsilon": epsilon,
        "comparison_absolute_tolerance": comparison_atol,
        "comparison_relative_tolerance": comparison_rtol,
        "comparison_scope": "Same shifted derivative model compared to one finite combined q/pose tangent; no unshifted-accuracy claim",
        "next_optimizer_steps": dict(next_optimizer_steps),
        "learning_rate": learning_rate,
        "pose_learning_rate": pose_learning_rate,
        "inverse_metric": "block_rate/(sqrt(proposed_v/(1-.999**next_block_step))+1e-12)",
        "optimizer_state_changed": False,
        "residual_response_used_in_seed": False,
        "residual_response_used_in_control_probe": False,
        "extra_positive_screen_is_physical_gate": False,
        "determinant_adjoint_receipts": [],
    }
    stage = "source_binding"
    started = time.perf_counter()
    graph_key = "MouthOpenJointDeterminants"

    def budget() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            message = "Joint determinant cache wall budget exhausted"
            raise ForwardConvergenceError(message)

    def restore_source() -> None:
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()

    def check_linear(receipt: dict) -> None:
        assert receipt["shifted_relative_residual"] <= 1e-7
        assert receipt["native_shifted_relative_residual"] <= 1e-7

    try:
        assert len(selected_ids) <= 16
        torch.testing.assert_close(
            u.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=0
        )
        assert q.shape == dq.shape == gq.shape
        assert pose.shape == dp.shape == gp.shape == (6,)
        mq, vq, mp, vp = proposed_moments
        qstep, pstep = next_optimizer_steps["q"], next_optimizer_steps["pose"]
        d_q = learning_rate / ((vq.detach() / (1 - 0.999**qstep)).sqrt() + 1e-12)
        d_p = pose_learning_rate / ((vp.detach() / (1 - 0.999**pstep)).sqrt() + 1e-12)
        # Use the runner's arithmetic order to validate its untouched Adam proposal.
        expected_dq = (
            -learning_rate
            * (mq / (1 - 0.9**qstep))
            / ((vq / (1 - 0.999**qstep)).sqrt() + 1e-12)
        )
        expected_dp = (
            -pose_learning_rate
            * (mp / (1 - 0.9**pstep))
            / ((vp / (1 - 0.999**pstep)).sqrt() + 1e-12)
        )
        torch.testing.assert_close(dq, expected_dq, rtol=0, atol=0)
        torch.testing.assert_close(dp, expected_dp, rtol=0, atol=0)
        inverse_metric = torch.cat((d_q.flatten(), d_p.flatten())).cpu().numpy()
        proposal = (
            torch.cat((dq.detach().flatten(), dp.detach().flatten())).cpu().numpy()
        )
        objective_gradient = (
            torch.cat((gq.detach().flatten(), gp.detach().flatten())).cpu().numpy()
        )
        np.savez_compressed(
            output / "source-and-adam.npz",
            activation_inv=q.detach().cpu().numpy(),
            pose_normalized=pose.detach().cpu().numpy(),
            displacement_m=old_np,
            dq=dq.detach().cpu().numpy(),
            dp=dp.detach().cpu().numpy(),
            gq=gq.detach().cpu().numpy(),
            gp=gp.detach().cpu().numpy(),
            proposed_mq=mq.detach().cpu().numpy(),
            proposed_vq=vq.detach().cpu().numpy(),
            proposed_mp=mp.detach().cpu().numpy(),
            proposed_vp=vp.detach().cpu().numpy(),
            q_optimizer_step=qstep,
            pose_optimizer_step=pstep,
        )
        receipt["source_and_adam"] = record(output / "source-and-adam.npz")
        stage = "exact_source_graph"
        budget()
        restore_source()
        with torch.enable_grad():
            source_q = q.detach().clone().requires_grad_()
            source_pose = pose.detach().clone().requires_grad_()
            source_u = runtime.solve(
                materials(source_q),
                physics.boundary(physical_pose(source_pose)),
                u.detach(),
                key=graph_key,
            )
            torch.testing.assert_close(source_u.detach(), u, rtol=0, atol=0)
            receipt["source_forward"] = copy.deepcopy(runtime.last_forward)
            assert runtime.last_forward["pncg"]["steps"] == 0
            assert runtime.last_forward["newton"]["steps"] == 0
            points_t = torch.as_tensor(reference, device=q.device, dtype=q.dtype)
            rows = []
            for index, original_id in zip(selected_indices, selected_ids, strict=True):
                stage = f"determinant_adjoint_{original_id}"
                budget()
                restore_source()
                tet = torch.as_tensor(tets[index], device=q.device, dtype=torch.long)
                rest = points_t[tet]
                deformed = rest + source_u[tet]
                j = torch.linalg.det((deformed[1:] - deformed[0]).T) / torch.linalg.det(
                    (rest[1:] - rest[0]).T
                )
                np.testing.assert_allclose(
                    float(j.detach()), old_j[index], rtol=1e-9, atol=1e-12
                )
                runtime.drop_warm_adjoint(graph_key)
                jq, jp = torch.autograd.grad(
                    j, (source_q, source_pose), retain_graph=True
                )
                full = (
                    torch.cat((jq.detach().flatten(), jp.detach().flatten()))
                    .cpu()
                    .numpy()
                )
                rows.append(full)
                adjoint = {
                    "original_id": int(original_id),
                    "source_J": float(old_j[index]),
                    "adjoint": copy.deepcopy(runtime.last_adjoint),
                    "sparse_adjoint": copy.deepcopy(runtime.last_sparse_adjoint),
                }
                np.savez_compressed(
                    output / f"gradient-{original_id}.npz",
                    q_gradient=jq.detach().cpu().numpy(),
                    pose_gradient=jp.detach().cpu().numpy(),
                )
                adjoint["gradient"] = record(output / f"gradient-{original_id}.npz")
                receipt["determinant_adjoint_receipts"].append(adjoint)
                write_json(output / f"gradient-{original_id}.json", adjoint)
                check_linear(adjoint["sparse_adjoint"])
            determinant_gradients = np.asarray(rows).reshape(
                (len(selected_ids), len(proposal))
            )
        stage = "combined_tangent_validation"
        budget()
        restore_source()
        with torch.no_grad():
            seed, probe_receipt = _damped_equilibrium_tangent(
                physics,
                old_materials,
                materials(q + epsilon * dq),
                u,
                physics.boundary(physical_pose(pose + epsilon * dp)),
                deadline=deadline,
                predictor_relative_shift=runtime.adjoint_relative_shift,
                predictor_rtol=runtime.tolerances["adjoint_rtol"],
                material_changed=True,
            )
        receipt["combined_probe"] = copy.deepcopy(probe_receipt)
        if probe_receipt["rhs_norm"] > 0:
            check_linear(probe_receipt["sparse_solver"])
        else:
            assert probe_receipt["maximum_free_displacement_m"] == 0
        probe_delta = (seed.detach().cpu().numpy() - old_np) / epsilon
        probe_dj = determinant_directional_derivative(
            reference, tets, old_np, probe_delta
        )
        adjoint_dj = determinant_gradients @ proposal
        finite_dj = probe_dj[selected_indices]
        comparison_scale = np.maximum(abs(adjoint_dj), abs(finite_dj))
        comparison_limit = comparison_atol + comparison_rtol * comparison_scale
        difference = adjoint_dj - finite_dj
        np.savez_compressed(
            output / "combined-probe.npz",
            selected_original_ids=selected_ids,
            adjoint_directional=adjoint_dj,
            probe_directional=finite_dj,
            difference=difference,
            comparison_scale=comparison_scale,
            comparison_limit=comparison_limit,
            full_probe_directional=probe_dj,
        )
        receipt["combined_probe_comparison"] = {
            "passed": bool(np.all(abs(difference) <= comparison_limit)),
            "maximum_absolute_difference": float(np.max(abs(difference), initial=0)),
            "receipt": record(output / "combined-probe.npz"),
        }
        write_json(output / "summary.json", receipt)
        assert receipt["combined_probe_comparison"]["passed"], (
            "Combined joint determinant directional comparison failed"
        )
        stage = "separate_residual_response"
        budget()
        restore_source()
        with torch.no_grad():
            state = model.State(u=u.detach().clone())
            state.collision = collision.state_at(state.u)
            residual = model.dof_map.to_free_grad(model.grad(state))
            residual_norm = float(torch.linalg.vector_norm(residual))
            assert residual_norm <= runtime.tolerances["atol"]
            problem = _AdjointProblem(b=-residual, model=model, model_state=state)
            solution = runtime.solver.solve(problem, torch.zeros_like(residual))
            solver = copy.deepcopy(runtime.last_sparse_adjoint)
            receipt["residual_response"] = {
                "equation": "(H_old+shift I)w_free=-old_free_force; w_fixed=0",
                "old_force_norm": residual_norm,
                "solver": solver,
                "linear_prediction_only": True,
            }
            write_json(output / "summary.json", receipt)
            check_linear(solver)
            assert solution.success, solver
            response = torch.zeros_like(u)
            response.flatten()[model.dof_map.free_indices] = solution.params.detach()
            assert torch.isfinite(response).all()
            assert not torch.count_nonzero(
                response.flatten()[model.dof_map.fixed_indices]
            )
            residual_delta = determinant_directional_derivative(
                reference, tets, old_np, response.cpu().numpy()
            )
        np.savez_compressed(
            output / "residual-response.npz",
            displacement_response_m=response.cpu().numpy(),
            original_retained_ids=retained_ids,
            old_det_f=old_j,
            residual_delta_j=residual_delta,
        )
        receipt["residual_response"]["arrays"] = record(
            output / "residual-response.npz"
        )
        stage = "save_cache"
        cache = save_joint_projection_cache(
            output / "cache",
            original_ids=retained_ids,
            old_j=old_j,
            residual_delta_j=residual_delta,
            selected_ids=selected_ids,
            determinant_gradients=determinant_gradients,
            objective_gradient=objective_gradient,
            inverse_metric=inverse_metric,
            proposal=proposal,
            q_shape=tuple(q.shape),
        )
        receipt.update(status="joint_cache_ready", cache=record(cache.source_path))
    except Exception as error:
        receipt.update(
            status="joint_cache_failed",
            failure={
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    else:
        return cache
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed
        runtime.forward.state.u = original_runtime_u
        runtime.forward.state.collision = original_runtime_collision
        runtime.deadline = original_deadline
        runtime.drop_warm_adjoint(graph_key)
        _assert_materials_equal(model.get_materials(), original_materials)
        torch.testing.assert_close(
            model.dof_map.fixed_values, original_fixed, rtol=0, atol=0
        )
        for value, original in zip(
            (q, pose, u, dq, dp, gq, gp, *proposed_moments),
            original_inputs,
            strict=True,
        ):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        assert next_optimizer_steps == receipt["next_optimizer_steps"]
        receipt.update(
            model_state_restored_exactly=True,
            optimizer_inputs_unchanged=True,
            seconds=time.perf_counter() - started,
        )
        write_json(output / "summary.json", receipt)
