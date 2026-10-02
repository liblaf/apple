# ruff: noqa: EM101, PLR0915, TRY003, TRY300, TRY301
"""Damped equilibrium-tangent CCD seeds for coupled MouthOpen updates.

This helper changes only the starting displacement supplied to the production
collision-on corrector.  It never disables or replaces the IPC potential.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch
from joint_common import write_json
from joint_coupled_predictor import (
    audit_coupled_motion,
    clone_materials,
    rotation_sagitta,
)
from joint_equilibrium import ForwardConvergenceError
from scipy.spatial.transform import Rotation

from liblaf.apple.inverse._diff_forward import _AdjointProblem


def _mandible_arc_radius(
    physics: Any,
    collision: Any,
    pivot: torch.Tensor,
    old_fixed: torch.Tensor,
    new_fixed: torch.Tensor,
) -> tuple[float, int]:
    """Bound only vertices with rigid mandible arcs, never linear soft motion."""
    assert pivot.shape == (3,)
    geometry = physics.full_skull.geometry
    source_ids = torch.as_tensor(
        geometry.mandible_global_ids, device=collision.indices.device, dtype=torch.long
    )
    is_source_mandible = torch.isin(collision.indices, source_ids)
    assert int(is_source_mandible.sum()) == geometry.mandible_node_count
    torch.testing.assert_close(
        collision.indices[is_source_mandible], source_ids, rtol=0, atol=0
    )
    source_points = torch.as_tensor(
        geometry.mandible_points_m,
        device=collision.vertices.device,
        dtype=collision.vertices.dtype,
    )
    torch.testing.assert_close(
        collision.vertices[is_source_mandible], source_points, rtol=0, atol=0
    )

    dofs = physics.runtime.forward.model.dof_map
    fixed_mask = torch.zeros(dofs.n_full, device=old_fixed.device, dtype=torch.bool)
    fixed_mask[dofs.fixed_indices] = True
    fixed_mask = fixed_mask.reshape(-1, 3)
    assert bool(torch.all(fixed_mask == fixed_mask[:, :1]))
    old_full = torch.zeros(
        (dofs.n_full,), device=old_fixed.device, dtype=old_fixed.dtype
    )
    new_full = torch.zeros_like(old_full)
    old_full[dofs.fixed_indices] = old_fixed
    new_full[dofs.fixed_indices] = new_fixed
    moving_fixed = (
        torch.linalg.vector_norm(
            (new_full - old_full).reshape(-1, 3)[collision.indices], dim=1
        )
        > 0
    )
    # A non-source collision vertex follows the predictor's endpoint chord.  A
    # fixed vertex with a rigid arc would need its own sagitta contribution.
    assert not bool(torch.any(moving_fixed & ~is_source_mandible))
    radius = float(
        torch.linalg.vector_norm(
            collision.vertices[is_source_mandible] - pivot, dim=1
        ).max()
    )
    return radius, int(is_source_mandible.sum())


def _assert_materials_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for name, left_fields in left.items():
        right_fields = right[name]
        assert left_fields.keys() == right_fields.keys()
        for field, left_value in left_fields.items():
            torch.testing.assert_close(left_value, right_fields[field], rtol=0, atol=0)


@torch.no_grad()
def _damped_equilibrium_tangent(
    physics: Any,
    old_materials: dict,
    new_materials: dict,
    old_full: torch.Tensor,
    fixed_target: torch.Tensor,
    *,
    deadline: float | None,
    predictor_relative_shift: float,
    predictor_rtol: float,
    material_changed: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve the current runtime's damped old-contact tangent equation.

    ``MouthOpenHybridEquilibrium.solver`` owns the sparse HybridHessian and its
    optional relative diagonal shift.  The right hand side is assembled at the
    old material/contact state:

    ``(H_ff(old) + lambda I) du_f = -[df_material + H_old df_fixed]``.

    The accepted old equilibrium has met its force tolerance. Its small residual
    is left to the nonlinear corrector. An unscaled residual correction here
    would create nonzero motion even as the parameter step approaches zero,
    defeating CCD backtracking.

    The material-force difference excludes IPC because IPC has no Raw6
    parameter.  Both the free and boundary Hessian products retain old IPC.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        raise ForwardConvergenceError("declared seed wall budget exhausted")
    assert predictor_relative_shift >= 0
    assert 0 < predictor_rtol <= 1e-7
    runtime = physics.runtime
    assert predictor_relative_shift == runtime.adjoint_relative_shift
    assert predictor_rtol == runtime.tolerances["adjoint_rtol"]
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    dofs = model.dof_map
    u = old_full.detach().clone()
    fixed_old = u.flatten()[dofs.fixed_indices].clone()
    assert fixed_target.shape == fixed_old.shape

    model.set_materials(old_materials)
    dofs.fixed_values = fixed_old
    state = model.State(u=u)
    state.collision = collision.state_at(u)
    residual = model.grad(state)

    old_force_norm = float(torch.linalg.vector_norm(dofs.to_free_grad(residual)))
    assert old_force_norm <= runtime.tolerances["atol"], old_force_norm
    delta_material_force = torch.zeros_like(u)
    if material_changed:
        old_bulk = torch.zeros_like(u)
        model.warp_model.grad(u, old_bulk)
        model.set_materials(new_materials)
        new_bulk = torch.zeros_like(u)
        model.warp_model.grad(u, new_bulk)
        model.set_materials(old_materials)
        delta_material_force = new_bulk - old_bulk

    boundary_delta = torch.zeros_like(u)
    boundary_delta.flatten()[dofs.fixed_indices] = fixed_target - fixed_old
    boundary_force = model.hess_prod(state, boundary_delta)
    rhs = -dofs.to_free_grad(delta_material_force + boundary_force)
    assert bool(torch.isfinite(rhs).all())
    rhs_norm = float(torch.linalg.vector_norm(rhs))

    if rhs_norm == 0:
        delta_free = torch.zeros_like(rhs)
        solver_receipt = {
            "method": "zero-right-hand-side",
            "relative_shift": predictor_relative_shift,
            "shift": 0.0,
            "shifted_relative_residual": 0.0,
            "original_unshifted_relative_residual": 0.0,
        }
        result = "zero right-hand side"
    else:
        system = _AdjointProblem(b=rhs, model=model, model_state=state)
        solution = runtime.solver.solve(system, torch.zeros_like(rhs))
        delta_free = solution.params.detach().clone()
        # CuPy may label an iteration limit as unsuccessful even when the
        # audited sparse and native shifted residuals below satisfy the stated
        # tolerance.  Residuals, rather than this status flag, are the gate.
        assert bool(torch.isfinite(delta_free).all())
        solver_receipt = copy.deepcopy(runtime.last_sparse_adjoint)
        assert solver_receipt["shifted_relative_residual"] <= predictor_rtol, (
            solver_receipt
        )
        assert solver_receipt["native_shifted_relative_residual"] <= predictor_rtol, (
            solver_receipt
        )
        result = str(solution.result)

    dofs.fixed_values = fixed_target.detach().clone()
    candidate = dofs.to_full(dofs.to_free(u) + delta_free).detach().clone()
    assert bool(torch.isfinite(candidate).all())
    torch.testing.assert_close(
        candidate.flatten()[dofs.fixed_indices], fixed_target, rtol=0, atol=0
    )
    return candidate, {
        "method": "damped-old-contact-equilibrium-tangent",
        "equation": "(H_ff(old_contact,old_material)+lambda*I) du_f = -(delta_f_material + H_old delta_u_fixed)_f",
        "residual_term_omitted_for_zero_update_consistency": True,
        "linear_result": result,
        "solver_reported_success": solver_receipt.get("solver_success", True),
        "residual_acceptance": "native and sparse shifted relative residual <= predictor_rtol; solver status is recorded only",
        "predictor_relative_shift": predictor_relative_shift,
        "predictor_rtol": predictor_rtol,
        "rhs_norm": rhs_norm,
        "old_free_force_norm": float(
            torch.linalg.vector_norm(dofs.to_free_grad(residual))
        ),
        "parameter_force_change_norm": float(
            torch.linalg.vector_norm(dofs.to_free_grad(delta_material_force))
        ),
        "boundary_force_change_norm": float(
            torch.linalg.vector_norm(dofs.to_free_grad(boundary_force))
        ),
        "maximum_free_displacement_m": float(delta_free.abs().max()),
        "maximum_boundary_displacement_m": float(boundary_delta.abs().max()),
        "sparse_solver": solver_receipt,
    }


@torch.no_grad()
def prepare_coupled_seed(
    physics: Any,
    materials: Any,
    old_q: torch.Tensor,
    new_q: torch.Tensor,
    old_pose_rad_m: torch.Tensor,
    new_pose_rad_m: torch.Tensor,
    seed: torch.Tensor,
    output_dir: Path,
    *,
    deadline: float | None = None,
    predictor_relative_shift: float | None = None,
    predictor_rtol: float | None = None,
    **_unused: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return one full-contact tangent seed, or fail for adaptive backtracking.

    The caller owns pose-quality policy and terminal tetrahedron acceptance.
    This function deliberately has no detF or IsFixed quality threshold; it
    screens the proposed old-to-candidate motion using IPC CCD only.
    """
    assert old_q.shape == new_q.shape
    assert old_pose_rad_m.shape == new_pose_rad_m.shape == (6,)
    assert seed.shape == physics.runtime.forward.state.u.shape
    assert bool(
        torch.isfinite(old_q).all()
        and torch.isfinite(new_q).all()
        and torch.isfinite(old_pose_rad_m).all()
        and torch.isfinite(new_pose_rad_m).all()
        and torch.isfinite(seed).all()
    )
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    original_materials = clone_materials(model.get_materials())
    original_fixed = model.dof_map.fixed_values.detach().clone()
    shared_shift = float(runtime.adjoint_relative_shift)
    shared_rtol = float(runtime.tolerances["adjoint_rtol"])
    if predictor_relative_shift is None:
        predictor_relative_shift = shared_shift
    if predictor_rtol is None:
        predictor_rtol = shared_rtol
    assert predictor_relative_shift == shared_shift
    assert predictor_rtol == shared_rtol
    receipt: dict[str, Any] = {
        "success": False,
        "method": "damped-equilibrium-tangent-coupled-seed-with-ccd",
        "seed_only": True,
        "collision_disabled": False,
        "equilibrium_claimed": False,
        "final_strict_equilibrium_required": True,
        "physical_model_changed": False,
        "shared_runtime_linear_settings": {
            "predictor_relative_shift": predictor_relative_shift,
            "predictor_rtol": predictor_rtol,
            "runtime_relative_shift": shared_shift,
            "runtime_adjoint_rtol": shared_rtol,
            "exact_match": True,
        },
    }
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise ForwardConvergenceError("declared seed wall budget exhausted")
        old_fixed = physics.boundary(old_pose_rad_m)
        torch.testing.assert_close(
            seed.flatten()[model.dof_map.fixed_indices], old_fixed, rtol=0, atol=1e-14
        )
        new_fixed = physics.boundary(new_pose_rad_m)
        candidate, predictor = _damped_equilibrium_tangent(
            physics,
            clone_materials(materials(old_q)),
            clone_materials(materials(new_q)),
            seed,
            new_fixed,
            deadline=deadline,
            predictor_relative_shift=predictor_relative_shift,
            predictor_rtol=predictor_rtol,
            material_changed=not torch.equal(old_q, new_q),
        )
        receipt["predictor"] = predictor

        old = old_pose_rad_m.detach().cpu().numpy()
        new = new_pose_rad_m.detach().cpu().numpy()
        angle = float(
            (
                Rotation.from_rotvec(new[:3]) * Rotation.from_rotvec(old[:3]).inv()
            ).magnitude()
        )
        assert math.isfinite(angle)
        assert angle <= math.pi
        pivot = torch.as_tensor(physics.pivot_t, dtype=seed.dtype, device=seed.device)
        radius, rigid_vertex_count = _mandible_arc_radius(
            physics, collision, pivot, old_fixed, new_fixed
        )
        margin = rotation_sagitta(radius, angle)
        contact = audit_coupled_motion(
            collision, seed, candidate, rotation_margin_m=margin
        )
        receipt["motion_audit"] = {
            "old_to_candidate": contact,
            "delta_rigid_rotation_rad": angle,
            "rigid_arc_vertices": "complete appended mandible source mesh only",
            "rigid_arc_vertex_count": rigid_vertex_count,
            "maximum_mandible_source_vertex_radius_m": radius,
            "soft_motion_path": "linear old-to-candidate endpoint path screened by audit_coupled_motion",
            "rotation_sagitta_bound_m": margin,
        }
        if not contact["admitted"]:
            raise ForwardConvergenceError(
                "damped coupled seed failed collision CCD admission", receipt=receipt
            )
        receipt["success"] = True
        return candidate, receipt
    except ForwardConvergenceError as error:
        if error.receipt is not receipt:
            receipt["failure"] = {"message": str(error), "receipt": error.receipt}
        else:
            receipt["failure"] = {"message": str(error)}
        raise ForwardConvergenceError(str(error), receipt=receipt) from error
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed
        _assert_materials_equal(model.get_materials(), original_materials)
        torch.testing.assert_close(
            model.dof_map.fixed_values, original_fixed, rtol=0, atol=0
        )
        receipt["model_state_restored"] = True
        receipt["seconds"] = time.perf_counter() - started
        write_json(output_dir / "summary.json", receipt)


__all__ = ["prepare_coupled_seed"]
