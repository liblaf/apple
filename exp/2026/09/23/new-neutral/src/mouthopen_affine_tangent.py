"""Cache coupled determinant tangents and a separate force-residual response."""

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
from mouthopen_affine_direction import AffineDirectionCache, save_affine_cache
from mouthopen_coupled_seed import (
    _assert_materials_equal,
    _damped_equilibrium_tangent,
)
from mouthopen_pose_projection import (
    determinant_directional_derivative,
    determinant_ratio,
)

from liblaf.apple.inverse._diff_forward import _AdjointProblem


@torch.no_grad()
def build_affine_tangent_cache(  # noqa: PLR0915
    physics: Any,
    materials: Any,
    physical_pose: Any,
    q: torch.Tensor,
    dq: torch.Tensor,
    pose: torch.Tensor,
    requested_dp: torch.Tensor,
    displacement: torch.Tensor,
    output: Path,
    *,
    pose_gradient: torch.Tensor,
    strain_slope: float,
    original_target: float,
    epsilon: float,
    deadline: float | None,
) -> AffineDirectionCache:
    """Build one immutable linear model; never change the accepted state.

    The residual response is an unscaled determinant intercept. It is not
    included in control derivative probes or the actual production seed.
    Each proposed trial still needs CCD and nonlinear physical admission.
    """
    assert epsilon > 0
    assert original_target < 0
    output.mkdir(parents=True, exist_ok=False)
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    old_materials = clone_materials(materials(q))
    original_materials = clone_materials(model.get_materials())
    original_fixed = model.dof_map.fixed_values.detach().clone()
    fixed = physics.boundary(physical_pose(pose)).detach().clone()
    torch.testing.assert_close(
        displacement.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=5e-16
    )
    reference = np.asarray(physics.points)
    retained_ids = np.asarray(physics.base._mouthopen_retained_tetrahedron_ids)  # noqa: SLF001
    tets = np.asarray(physics.base.tets)[retained_ids]
    old_np = displacement.cpu().numpy()
    old_j = determinant_ratio(reference, tets, old_np)
    receipts: list[dict] = []
    receipt: dict = {
        "status": "running",
        "equilibrium_claimed": False,
        "residual_response_used_in_seed": False,
        "residual_response_used_in_control_probes": False,
        "relative_shift": runtime.adjoint_relative_shift,
        "internal_force_atol": runtime.tolerances["atol"],
        "epsilon": epsilon,
        "tangent_receipts": receipts,
    }
    stage = "control_tangents"
    started = time.perf_counter()

    def budget() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            message = "affine tangent wall budget exhausted"
            raise ForwardConvergenceError(message)

    def derivative(delta: np.ndarray) -> np.ndarray:
        return determinant_directional_derivative(reference, tets, old_np, delta)

    try:

        def tangent(value: torch.Tensor, next_pose: torch.Tensor, *, changed: bool):
            budget()
            seed, probe_receipt = _damped_equilibrium_tangent(
                physics,
                old_materials,
                materials(value),
                displacement,
                physics.boundary(physical_pose(next_pose)),
                deadline=deadline,
                predictor_relative_shift=runtime.adjoint_relative_shift,
                predictor_rtol=runtime.tolerances["adjoint_rtol"],
                material_changed=changed,
            )
            receipts.append(copy.deepcopy(probe_receipt))
            return derivative((seed.cpu().numpy() - old_np) / epsilon)

        q_delta = tangent(q + epsilon * dq, pose, changed=True)
        matrix = np.empty((len(old_j), 6))
        for coordinate in range(6):
            probe = pose.clone()
            probe[coordinate] += epsilon
            matrix[:, coordinate] = tangent(q, probe, changed=False)
        stage = "separate_residual_response"
        budget()
        model.set_materials(old_materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        state = model.State(u=displacement.detach().clone())
        state.collision = collision.state_at(state.u)
        residual = model.dof_map.to_free_grad(model.grad(state))
        assert torch.linalg.vector_norm(residual) <= runtime.tolerances["atol"]
        problem = _AdjointProblem(b=-residual, model=model, model_state=state)
        solution = runtime.solver.solve(problem, torch.zeros_like(residual))
        solver = copy.deepcopy(runtime.last_sparse_adjoint)
        assert solver["shifted_relative_residual"] <= runtime.tolerances["adjoint_rtol"]
        assert (
            solver["native_shifted_relative_residual"]
            <= runtime.tolerances["adjoint_rtol"]
        )
        response = torch.zeros_like(displacement)
        response.flatten()[model.dof_map.free_indices] = solution.params.detach()
        assert torch.isfinite(response).all()
        assert not torch.count_nonzero(response.flatten()[model.dof_map.fixed_indices])
        residual_delta = derivative(response.cpu().numpy())
        receipt["residual_response"] = {
            "equation": "(H_old + shift I) w_free = -old_free_force; w_fixed = 0",
            "linear_prediction_only": True,
            "old_force_norm": float(torch.linalg.vector_norm(residual)),
            "maximum_displacement_m": float(
                torch.linalg.vector_norm(response, dim=1).max()
            ),
            "solver": solver,
        }
        np.savez_compressed(
            output / "source-and-response.npz",
            displacement_m=old_np,
            activation_inv=q.cpu().numpy(),
            pose_normalized=pose.cpu().numpy(),
            dq=dq.cpu().numpy(),
            residual_displacement_m=response.cpu().numpy(),
        )
        stage = "save_coefficients"
        cache = save_affine_cache(
            output / "cache",
            original_ids=retained_ids,
            old_j=old_j,
            q_delta_j=q_delta,
            pose_jacobian=matrix,
            residual_delta_j=residual_delta,
            pose_gradient=pose_gradient.cpu().numpy(),
            requested_dp=requested_dp.cpu().numpy(),
            strain_slope=strain_slope,
            original_target=original_target,
        )
        receipt["status"] = "model_ready_for_trial_projection"
        receipt["cache"] = {
            "path": str(cache.source_path),
            "sha256": cache.source_sha256,
        }
    except Exception as error:
        receipt["status"] = "failed"
        receipt["failure"] = {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    else:
        return cache
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed
        _assert_materials_equal(model.get_materials(), original_materials)
        torch.testing.assert_close(
            model.dof_map.fixed_values, original_fixed, rtol=0, atol=0
        )
        receipt["model_state_restored_exactly"] = True
        receipt["seconds"] = time.perf_counter() - started
        write_json(output / "summary.json", receipt)
