# ruff: noqa: EM101, TRY003
"""Contact-aware equilibrium prediction with CCD-screened coupled motion.

Prediction changes only the numerical seed. The original fixed-boundary IPC
potential and strict PNCG corrector determine every accepted equilibrium.
"""

from __future__ import annotations

import math
import time
from typing import Any

import ipctk
import torch
from joint_equilibrium import ForwardConvergenceError

from liblaf.apple.inverse._diff_forward import _AdjointProblem


def clone_materials(materials: dict) -> dict:
    return {
        name: {key: value.detach().clone() for key, value in fields.items()}
        for name, fields in materials.items()
    }


def rotation_sagitta(radius: float, angle: float) -> float:
    """Maximum deviation of a fixed-axis circular arc from its endpoint chord."""
    assert math.isfinite(radius)
    assert radius >= 0
    assert math.isfinite(angle)
    assert abs(angle) <= math.pi
    return 2 * radius * math.sin(angle / 4) ** 2


@torch.no_grad()
def equilibrium_predictor(  # noqa: PLR0915
    *,
    model: Any,
    solver: Any,
    displacement: torch.Tensor,
    old_materials: dict,
    new_materials: dict,
    fixed_target: torch.Tensor,
    linear_rtol: float,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Solve Hff du = -r_old - delta_parameter_force - Hfc delta_fixed.

    Both Hessian blocks include the exact old contact Hessian. Parameter-force
    differences use the same old geometry and only the constitutive potential;
    IPC is independent of the active-stress parameters. All model mutations are
    restored even on failure, and the old geometry/contact state is owned here.
    """
    assert 0 < linear_rtol <= 1e-7
    assert 0 < residual_scale <= 1
    assert bool(
        torch.isfinite(displacement).all() and torch.isfinite(fixed_target).all()
    )
    started = time.perf_counter()
    original_materials = clone_materials(model.get_materials())
    original_fixed = model.dof_map.fixed_values.detach().clone()
    old = clone_materials(old_materials)
    new = clone_materials(new_materials)
    dofs = model.dof_map
    u = displacement.detach().clone()
    fixed_old = u.flatten()[dofs.fixed_indices].clone()
    assert fixed_target.shape == fixed_old.shape
    try:
        dofs.fixed_values = fixed_old
        model.set_materials(old)
        state = model.State(u=u)
        assert model.collision is not None
        state.collision = model.collision.state_at(u)
        residual = model.grad(state)
        old_bulk = torch.zeros_like(u)
        model.warp_model.grad(u, old_bulk)
        model.set_materials(new)
        new_bulk = torch.zeros_like(u)
        model.warp_model.grad(u, new_bulk)
        model.set_materials(old)
        delta_material_force = new_bulk - old_bulk
        boundary_delta = torch.zeros_like(u)
        boundary_delta.flatten()[dofs.fixed_indices] = fixed_target - fixed_old
        boundary_force = model.hess_prod(state, boundary_delta)
        rhs = -dofs.to_free_grad(
            residual_scale * residual + delta_material_force + boundary_force
        )
        assert bool(torch.isfinite(rhs).all())
        system = _AdjointProblem(b=rhs, model=model, model_state=state)
        norm = float(torch.linalg.vector_norm(rhs))
        if norm == 0:
            delta_free = torch.zeros_like(rhs)
            relative, absolute, result = 0.0, 0.0, "zero right-hand side"
        else:
            solution = solver.solve(system, torch.zeros_like(rhs))
            delta_free = solution.params.detach().clone()
            absolute = float(torch.linalg.vector_norm(system.matvec(delta_free) - rhs))
            relative = absolute / norm
            result = str(solution.result)
            if (
                not bool(solution.success)
                or not bool(torch.isfinite(delta_free).all())
                or not math.isfinite(relative)
                or relative > 1.05 * linear_rtol
            ):
                raise ForwardConvergenceError(
                    "coupled predictor linear solve unresolved",
                    receipt={
                        "success": False,
                        "relative_residual": relative,
                        "linear_result": result,
                    },
                )
        dofs.fixed_values = fixed_target.detach().clone()
        predicted = dofs.to_full(dofs.to_free(u) + delta_free).detach().clone()
        assert bool(torch.isfinite(predicted).all())
        return predicted, {
            "success": True,
            "method": "exact-equilibrium-tangent-with-residual-correction",
            "seconds": time.perf_counter() - started,
            "linear_result": result,
            "linear_relative_residual": relative,
            "linear_absolute_residual": absolute,
            "linear_rtol": linear_rtol,
            "rhs_norm": norm,
            "residual_scale": residual_scale,
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
            "seed_only_no_physical_model_change": True,
        }
    finally:
        model.set_materials(original_materials)
        dofs.fixed_values = original_fixed


@torch.no_grad()
def audit_coupled_motion(
    collision: Any,
    old_u: torch.Tensor,
    new_u: torch.Tensor,
    *,
    rotation_margin_m: float = 0.0,
) -> dict:
    """CCD on both soft and rigid motion, with an optional curved-hinge bound.

    Adding the sagitta to the CCD separation certifies the rigid arc against
    linear soft-tissue motion: only one side of each allowed pair is rigid.
    The candidate inflation covers the same enlarged separation. No production
    contact parameters or accepted collision states are modified.
    """
    assert math.isfinite(rotation_margin_m)
    assert rotation_margin_m >= 0
    x0 = (collision.vertices + old_u[collision.indices]).numpy(force=True)
    x1 = (collision.vertices + new_u[collision.indices]).numpy(force=True)
    minimum = collision.min_distance + rotation_margin_m
    candidates = ipctk.Candidates()
    candidates.build(
        mesh=collision.collision_mesh,
        vertices_t0=x0,
        vertices_t1=x1,
        inflation_radius=max(collision.inflation_radius, minimum),
        broad_phase=ipctk.LBVH(),
    )
    fraction = float(
        candidates.compute_collision_free_stepsize(
            mesh=collision.collision_mesh,
            vertices_t0=x0,
            vertices_t1=x1,
            min_distance=minimum,
            narrow_phase_ccd=collision.narrow_phase_ccd,
        )
    )
    assert math.isfinite(fraction)
    assert 0 <= fraction <= 1
    receipt = {
        "admitted": False,
        "ccd_fraction": fraction,
        "rotation_chord_deviation_bound_m": rotation_margin_m,
        "ccd_separation_m": minimum,
        "physical_buffer_m": collision.min_distance,
        "path": "linear soft motion and chord-screened rigid hinge arc with conservative deviation bound",
    }
    if fraction < 1:
        return receipt
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, x1, ipctk.LBVH())
    )
    contact = collision.diagnostics(collision.state_at(new_u), new_u)
    receipt.update(endpoint_intersections=intersects, endpoint_contact=contact)
    receipt["admitted"] = not intersects and contact["contact_numerically_valid"]
    return receipt
