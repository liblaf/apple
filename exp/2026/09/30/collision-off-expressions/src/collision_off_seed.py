# Copyright (c) 2026 liblaf
# ruff: noqa: EM101, TRY003, TRY300, TRY301
"""Coupled material and mandible tangent seed with collision disabled."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from joint_equilibrium import ForwardConvergenceError

from liblaf.apple.inverse._diff_forward import _AdjointProblem


def _clone_materials(materials: dict) -> dict:
    return {
        name: {field: value.detach().clone() for field, value in fields.items()}
        for name, fields in materials.items()
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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
    """Solve the old collision-off tangent for a coupled parameter change."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise ForwardConvergenceError("declared seed wall budget exhausted")
    runtime = physics.runtime
    assert runtime.forward.model.collision is None
    assert predictor_relative_shift == runtime.adjoint_relative_shift
    assert predictor_rtol == runtime.tolerances["adjoint_rtol"]
    model = runtime.forward.model
    dofs = model.dof_map
    u = old_full.detach().clone()
    fixed_old = u.flatten()[dofs.fixed_indices].clone()
    assert fixed_target.shape == fixed_old.shape
    model.set_materials(old_materials)
    dofs.fixed_values = fixed_old
    state = model.State(u=u)
    assert state.collision is None
    old_force = model.grad(state)
    old_free_force = float(torch.linalg.vector_norm(dofs.to_free_grad(old_force)))
    assert old_free_force <= runtime.tolerances["atol"], old_free_force
    material_force_change = torch.zeros_like(u)
    if material_changed:
        model.set_materials(new_materials)
        material_force_change = model.grad(state) - old_force
        model.set_materials(old_materials)
    boundary_change = torch.zeros_like(u)
    boundary_change.flatten()[dofs.fixed_indices] = fixed_target - fixed_old
    boundary_force_change = model.hess_prod(state, boundary_change)
    rhs = -dofs.to_free_grad(material_force_change + boundary_force_change)
    assert bool(torch.isfinite(rhs).all())
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if rhs_norm == 0:
        free_change = torch.zeros_like(rhs)
        linear_receipt = {"method": "zero-right-hand-side"}
        linear_result = "zero right hand side"
    else:
        system = _AdjointProblem(b=rhs, model=model, model_state=state)
        solution = runtime.solver.solve(system, torch.zeros_like(rhs))
        free_change = solution.params.detach().clone()
        assert bool(torch.isfinite(free_change).all())
        linear_receipt = copy.deepcopy(runtime.last_sparse_adjoint)
        assert linear_receipt["shifted_relative_residual"] <= predictor_rtol
        assert linear_receipt["native_shifted_relative_residual"] <= predictor_rtol
        linear_result = str(solution.result)
    dofs.fixed_values = fixed_target.detach().clone()
    candidate = dofs.to_full(dofs.to_free(u) + free_change).detach().clone()
    assert bool(torch.isfinite(candidate).all())
    torch.testing.assert_close(
        candidate.flatten()[dofs.fixed_indices], fixed_target, rtol=0, atol=0
    )
    return candidate, {
        "equation": "(H_ff(old)+lambda I) du_f = -(delta_f_material + H_old delta_u_fixed)_f",
        "old_free_force_norm": old_free_force,
        "rhs_norm": rhs_norm,
        "parameter_force_change_norm": float(
            torch.linalg.vector_norm(dofs.to_free_grad(material_force_change))
        ),
        "boundary_force_change_norm": float(
            torch.linalg.vector_norm(dofs.to_free_grad(boundary_force_change))
        ),
        "maximum_free_displacement_m": float(free_change.abs().max()),
        "maximum_boundary_displacement_m": float(boundary_change.abs().max()),
        "linear_result": linear_result,
        "sparse_solver": linear_receipt,
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
    """Predict the coupled state; the caller performs the strict nonlinear solve.

    The old state must already be a collision-off equilibrium.  The predictor
    omits its small residual so a zero parameter step yields zero motion.
    """
    assert old_q.shape == new_q.shape
    assert old_pose_rad_m.shape == new_pose_rad_m.shape == (6,)
    assert bool(torch.isfinite(old_q).all() and torch.isfinite(new_q).all())
    assert bool(
        torch.isfinite(old_pose_rad_m).all()
        and torch.isfinite(new_pose_rad_m).all()
        and torch.isfinite(seed).all()
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime = physics.runtime
    model = runtime.forward.model
    assert model.collision is None
    assert runtime.forward.state.collision is None
    assert seed.shape == runtime.forward.state.u.shape
    dofs = model.dof_map
    original_materials = _clone_materials(model.get_materials())
    original_fixed = dofs.fixed_values.detach().clone()
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
        "method": "collision-off-coupled-equilibrium-tangent",
        "seed_only": True,
        "collision_enabled": False,
        "ccd_enabled": False,
        "equilibrium_claimed": False,
        "final_strict_equilibrium_required": True,
        "predictor_relative_shift": shared_shift,
        "predictor_rtol": shared_rtol,
    }
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise ForwardConvergenceError("declared seed wall budget exhausted")
        old_fixed = physics.boundary(old_pose_rad_m)
        new_fixed = physics.boundary(new_pose_rad_m)
        torch.testing.assert_close(
            seed.flatten()[dofs.fixed_indices], old_fixed, rtol=0, atol=1e-14
        )
        candidate, receipt["predictor"] = _damped_equilibrium_tangent(
            physics,
            _clone_materials(materials(old_q)),
            _clone_materials(materials(new_q)),
            seed,
            new_fixed,
            deadline=deadline,
            predictor_relative_shift=shared_shift,
            predictor_rtol=shared_rtol,
            material_changed=not torch.equal(old_q, new_q),
        )
        receipt["success"] = True
        return candidate, receipt
    except ForwardConvergenceError as error:
        receipt["failure"] = {"message": str(error), "receipt": error.receipt}
        raise ForwardConvergenceError(str(error), receipt=receipt) from error
    finally:
        model.set_materials(original_materials)
        dofs.fixed_values = original_fixed
        assert model.collision is None
        assert runtime.forward.state.collision is None
        receipt["model_state_restored"] = True
        receipt["seconds"] = time.perf_counter() - started
        _write_json(output_dir / "summary.json", receipt)


__all__ = ["_damped_equilibrium_tangent", "prepare_coupled_seed"]
