"""Validate contact derivatives, CCD, and implicit material/jaw gradients."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Literal

import ipctk
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_contact import OwnedContact
from joint_equilibrium import (
    Equilibrium,
    ForwardConvergenceError,
    configure_cuda,
    rigid_displacement,
)
from joint_materials import StableNeoHookeanStress

from liblaf import cherries
from liblaf.apple.common import FRACTION, LAMBDA, MU
from liblaf.apple.forward import Forward, ModelBuilder

FD_STEPS = (1.0e-2, 3.0e-3)
FD_RELATIVE_TOLERANCE = 0.02


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    contact_spec: Path = GROUP / "data/contact/config.json"
    output_dir: Path = cherries.output("contact-validation", mkdir=True)
    forward_method: Literal["pncg", "newton_cg"] = "pncg"
    newton_linear_rtol: float = 1e-3
    newton_max_steps: int = 12


def geometry() -> tuple[np.ndarray, np.ndarray]:
    # Node 0 is a free point 50 um above the fixed triangle 1-2-3. The four
    # tetrahedra subdivide the outer tetrahedron 1-2-3-4 around that point.
    points = np.asarray(
        [
            [0.0, -3.25e-4, 5.0e-5],
            [-1.0e-3, -1.0e-3, 0.0],
            [1.0e-3, -1.0e-3, 0.0],
            [0.0, 1.0e-3, 0.0],
            [0.0, 0.0, 2.0e-3],
        ],
        dtype=np.float64,
    )
    tets = np.asarray(
        [[0, 2, 3, 4], [0, 1, 4, 3], [0, 1, 2, 4], [0, 1, 3, 2]],
        dtype=np.int64,
    )
    for tet in tets:
        determinant = np.linalg.det((points[tet[1:]] - points[tet[0]]).T)
        if determinant < 0:
            tet[2], tet[3] = tet[3], tet[2]
    volumes = np.asarray(
        [np.linalg.det((points[tet[1:]] - points[tet[0]]).T) / 6 for tet in tets]
    )
    assert np.all(volumes > 0), volumes
    return points, tets


def make_contact(points: torch.Tensor, config: dict) -> OwnedContact:
    faces = np.asarray([[1, 2, 3]], dtype=np.int32)
    positions = points.detach().cpu().numpy()
    mesh = ipctk.CollisionMesh(positions, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.asarray([0, 1, 1, 1, 0], dtype=np.int32)
    )
    mesh.init_adjacencies()
    potential = ipctk.BarrierPotential(
        dhat=config["dhat_m"],
        stiffness=config["stiffness_mpa"],
        use_physical_barrier=True,
    )
    return OwnedContact(
        collision_mesh=mesh,
        indices=torch.arange(len(points), dtype=torch.long),
        potential=potential,
        broad_phase=ipctk.LBVH(),
        use_physical_barrier=True,
        vertices=points.detach().clone(),
    )


def make_runtime(
    config: dict,
    *,
    forward_method: Literal["pncg", "newton_cg"] = "pncg",
    newton_linear_rtol: float = 1e-3,
    newton_max_steps: int = 12,
) -> tuple[Equilibrium, torch.Tensor, OwnedContact]:
    points_np, tets = geometry()
    mesh = pv.UnstructuredGrid(
        np.column_stack((np.full(len(tets), 4), tets)).ravel(),
        np.full(len(tets), pv.CellType.TETRA),
        points_np,
    )
    mesh.cell_data[MU.vtk] = np.full(len(tets), 0.02)
    mesh.cell_data[LAMBDA.vtk] = np.full(len(tets), 0.08)
    mesh.cell_data[FRACTION.vtk] = np.ones(len(tets))
    mesh.point_data["FixedMask"] = np.repeat(
        (np.arange(len(points_np)) > 0)[:, None], 3, axis=1
    )
    mesh.point_data["FixedValue"] = np.zeros_like(points_np)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookeanStress.from_pyvista(mesh, name="bulk"))
    model = builder.finalize()
    points = torch.as_tensor(points_np)
    contact = make_contact(points, config)
    model.collision = contact
    runtime = Equilibrium(
        Forward(model),
        rtol=1.0e-8,
        atol=1.0e-14,
        adjoint_rtol=1.0e-10,
        max_steps=2000,
        forward_method=forward_method,
        newton_linear_rtol=newton_linear_rtol,
        newton_max_steps=newton_max_steps,
    )
    return runtime, points, contact


def vector_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.linalg.vector_norm(expected).clamp_min(1.0e-14)
    return float(torch.linalg.vector_norm(actual - expected) / scale)


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cuda()
    config = __import__("json").loads(cfg.contact_spec.read_text())
    assert config["schema"] == "joint-bone-contact-v1"
    assert config["enabled"] is True
    if cfg.forward_method == "newton_cg":
        assert cfg.newton_linear_rtol == 1e-3
        assert cfg.newton_max_steps == 12
    runtime, points, contact = make_runtime(
        config,
        forward_method=cfg.forward_method,
        newton_linear_rtol=cfg.newton_linear_rtol,
        newton_max_steps=cfg.newton_max_steps,
    )
    zero = torch.zeros_like(points)
    state = contact.state_at(zero)
    diagnostics = contact.diagnostics(state, zero)
    assert diagnostics["contact_numerically_valid"], diagnostics
    assert diagnostics["active_contact_count"] > 0, diagnostics
    assert diagnostics["barrier_energy"] > 0, diagnostics

    direction = torch.tensor(
        [
            [0.13, -0.07, 0.21],
            [-0.04, 0.08, -0.03],
            [0.02, -0.05, 0.01],
            [0.06, 0.01, -0.02],
            [0.0, 0.0, 0.0],
        ],
        dtype=points.dtype,
        device=points.device,
    )
    direction /= torch.linalg.vector_norm(direction)
    gradient = torch.zeros_like(zero)
    contact.grad(state, zero, gradient)
    hessian_direction = torch.zeros_like(zero)
    contact.hess_prod(state, zero, direction, hessian_direction)
    energy_checks = []
    hessian_checks = []
    for step in (1.0e-7, 3.0e-8):
        plus_u = step * direction
        minus_u = -step * direction
        plus_state = contact.state_at(plus_u)
        minus_state = contact.state_at(minus_u)
        energy_fd = (
            contact.fun(plus_state, plus_u) - contact.fun(minus_state, minus_u)
        ) / (2 * step)
        energy_analytic = torch.sum(gradient * direction)
        energy_error = float(
            (energy_fd - energy_analytic).abs()
            / torch.maximum(energy_fd.abs(), energy_analytic.abs()).clamp_min(1.0e-14)
        )
        plus_gradient = torch.zeros_like(zero)
        minus_gradient = torch.zeros_like(zero)
        contact.grad(plus_state, plus_u, plus_gradient)
        contact.grad(minus_state, minus_u, minus_gradient)
        hessian_fd = (plus_gradient - minus_gradient) / (2 * step)
        hessian_error = vector_relative(hessian_direction, hessian_fd)
        energy_checks.append(
            {
                "step_m": step,
                "analytic": float(energy_analytic),
                "finite_difference": float(energy_fd),
                "relative_error": energy_error,
            }
        )
        hessian_checks.append(
            {
                "step_m": step,
                "analytic_norm": float(torch.linalg.vector_norm(hessian_direction)),
                "finite_difference_norm": float(torch.linalg.vector_norm(hessian_fd)),
                "relative_error": hessian_error,
            }
        )
        assert energy_error < 2.0e-4, energy_checks[-1]
        assert hessian_error < 2.0e-3, hessian_checks[-1]

    crossing = torch.zeros_like(zero)
    crossing[0, 2] = -1.0e-4
    ccd_state = contact.state_at(zero)
    ccd_fraction = float(contact.max_step_size(ccd_state, zero, crossing))
    assert 0 < ccd_fraction < 1, ccd_fraction

    base = runtime.forward.model.get_materials()
    stress_pattern = torch.zeros((4, 3, 3), dtype=points.dtype, device=points.device)
    stress_pattern[0] = torch.tensor(
        [[0.8, 0.12, -0.07], [0.12, 0.5, 0.09], [-0.07, 0.09, 0.3]],
        dtype=points.dtype,
        device=points.device,
    )
    jaw_ids = torch.tensor([1, 2, 3], dtype=torch.int64, device=points.device)
    pivot = torch.tensor(
        [2.0e-4, -1.0e-4, 8.0e-4], dtype=points.dtype, device=points.device
    )
    parameter_scale = points.new_tensor(
        [1.0, 0.015, 0.017, 0.013, 2.0e-5, 2.5e-5, 1.5e-5]
    )
    initial = points.new_tensor([0.18, 0.11, -0.09, 0.07, 0.08, -0.06, 0.05])
    target = points.new_tensor([2.0e-5, -3.0e-5, 1.5e-4])
    seed = torch.zeros_like(points)

    def materials(parameter: torch.Tensor) -> dict:
        values = {name: dict(fields) for name, fields in base.items()}
        values["bulk"]["active_stress"] = 0.02 * parameter[0] * stress_pattern
        return values

    def fixed_values(parameter: torch.Tensor) -> torch.Tensor:
        pose = parameter[1:] * parameter_scale[1:]
        displacement = torch.zeros_like(points).index_copy(
            0,
            jaw_ids,
            rigid_displacement(points[jaw_ids], pivot, pose),
        )
        return displacement.flatten()[runtime.forward.model.dof_map.fixed_indices]

    def solve(parameter: torch.Tensor, key: str) -> torch.Tensor:
        return runtime.solve(
            materials(parameter), fixed_values(parameter), seed, key=key
        )

    def loss(solution: torch.Tensor) -> torch.Tensor:
        residual = (solution[0] - target) * 1000
        return residual.square().sum() + 0.11 * residual.prod()

    parameter = initial.detach().clone().requires_grad_()
    value = loss(solve(parameter, "implicit-fd"))
    (gradient_parameter,) = torch.autograd.grad(value, parameter)
    assert torch.isfinite(gradient_parameter).all()
    assert torch.all(gradient_parameter.abs() > 1.0e-10), gradient_parameter

    def explicit_equilibrium(parameter_at: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Independent dense Newton solve for the three free center coordinates."""
        with torch.no_grad():
            model = runtime.forward.model
            model.set_materials(materials(parameter_at))
            model.dof_map.fixed_values = fixed_values(parameter_at)
            free = model.dof_map.to_free(seed)
            state = model.State(u=model.dof_map.to_full(free))
            state.collision = contact.state_at(state.u)
            gradients = []
            ccd_fractions = []
            line_steps = []
            for _iteration in range(40):
                gradient_free = model.dof_map.to_free_grad(model.grad(state))
                gradient_norm = float(torch.linalg.vector_norm(gradient_free))
                gradients.append(gradient_norm)
                if gradient_norm <= 1.0e-14:
                    break
                basis = torch.eye(
                    model.n_free, dtype=points.dtype, device=points.device
                )
                columns = []
                for column in basis:
                    product = model.hess_prod(state, model.dof_map.to_full_grad(column))
                    columns.append(model.dof_map.to_free_grad(product))
                hessian = torch.stack(columns, dim=1)
                newton = torch.linalg.solve(hessian, gradient_free)
                full_step = model.dof_map.to_full_grad(-newton)
                ccd = float(model.max_step_size(state, full_step))
                ccd_fractions.append(ccd)
                alpha = min(1.0, 0.99 * ccd)
                old_energy = float(model.fun(state))
                accepted = False
                for line_step in range(30):
                    candidate_free = free - alpha * newton
                    candidate = model.State(u=model.dof_map.to_full(candidate_free))
                    candidate.collision = contact.state_at(candidate.u)
                    if float(model.fun(candidate)) <= old_energy:
                        free = candidate_free
                        state = candidate
                        line_steps.append(line_step)
                        accepted = True
                        break
                    alpha *= 0.5
                assert accepted, (gradients, ccd_fractions, line_steps)
            else:
                raise AssertionError((gradients, ccd_fractions, line_steps))
            return state.u.detach().clone(), {
                "implementation": "explicit_contact_3x3_newton",
                "steps": len(gradients) - 1,
                "initial_gradient_norm": gradients[0],
                "final_gradient_norm": gradients[-1],
                "gradient_tolerance": 1.0e-14,
                "minimum_ccd_fraction": min(ccd_fractions, default=1.0),
                "line_search_steps": line_steps,
            }

    implicit_checks = []
    for step in FD_STEPS:
        finite = []
        solve_receipts = []
        for index in range(len(initial)):
            offset = torch.zeros_like(initial)
            offset[index] = step
            plus_solution, plus_receipt = explicit_equilibrium(initial + offset)
            minus_solution, minus_receipt = explicit_equilibrium(initial - offset)
            plus = float(loss(plus_solution))
            minus = float(loss(minus_solution))
            finite.append((plus - minus) / (2 * step))
            solve_receipts.append(
                {"component": index, "plus": plus_receipt, "minus": minus_receipt}
            )
        finite_tensor = torch.tensor(finite, dtype=points.dtype, device=points.device)
        component_scale = torch.maximum(
            finite_tensor.abs(), gradient_parameter.abs()
        ).clamp_min(1.0e-9)
        component_errors = (finite_tensor - gradient_parameter).abs() / component_scale
        relative_error = vector_relative(gradient_parameter, finite_tensor)
        receipt = {
            "step": step,
            "analytic": gradient_parameter.detach().cpu().tolist(),
            "finite_difference": finite,
            "relative_error": relative_error,
            "maximum_component_relative_error": float(component_errors.max()),
            "independent_solves": solve_receipts,
        }
        implicit_checks.append(receipt)
        assert relative_error < FD_RELATIVE_TOLERANCE, receipt
        assert float(component_errors.max()) < FD_RELATIVE_TOLERANCE, receipt

    # A fixed contact triangle translated through the free point must be rejected
    # before the nonlinear equilibrium solve starts.
    crossing_parameter = initial.clone()
    crossing_parameter[1:4] = 0
    crossing_parameter[4:7] = crossing_parameter.new_tensor([0.0, 0.0, 8.0])
    boundary_rejection: dict = {"rejected": False}
    try:
        solve(crossing_parameter, "crossing")
    except ForwardConvergenceError as error:
        boundary_rejection = {
            "rejected": True,
            "error": str(error),
            "receipt": copy.deepcopy(error.receipt),
        }
    assert boundary_rejection["rejected"], boundary_rejection
    assert boundary_rejection["receipt"]["contact"]["ccd_boundary_fraction"] < 1.0, (
        boundary_rejection
    )

    def separate(scale: float, key: str) -> torch.Tensor:
        current = (scale * initial).detach().requires_grad_()
        current_loss = loss(solve(current, key))
        (current_gradient,) = torch.autograd.grad(current_loss, current)
        return current_gradient.detach()

    separate_0 = separate(1.0, "separate-0")
    separate_1 = separate(0.71, "separate-1")
    queued_0 = initial.detach().clone().requires_grad_()
    queued_1 = (0.71 * initial).detach().requires_grad_()
    queued_solution_0 = solve(queued_0, "queued-0")
    queued_solution_1 = solve(queued_1, "queued-1")
    queued_gradients = torch.autograd.grad(
        loss(queued_solution_0) + loss(queued_solution_1), (queued_0, queued_1)
    )
    queued_errors = {
        "expression_0": vector_relative(queued_gradients[0], separate_0),
        "expression_1": vector_relative(queued_gradients[1], separate_1),
    }
    assert max(queued_errors.values()) < 1.0e-6, queued_errors

    receipt = {
        "schema": "joint-contact-validation-v1",
        "success": True,
        "scope": (
            "synthetic four-tet FEM with a separate active point-triangle IPC "
            "collider; validates numerics, not interface material properties"
        ),
        "contact_spec_sha256": sha256(cfg.contact_spec),
        "contact_config": config,
        "forward_solver": runtime.forward_solver,
        "contact_diagnostics": diagnostics,
        "energy_directional_derivatives": energy_checks,
        "hessian_vector_products": hessian_checks,
        "direct_ccd_crossing_fraction": ccd_fraction,
        "boundary_ccd_rejection": boundary_rejection,
        "implicit_parameter_order": [
            "material_stress",
            "jaw_rx",
            "jaw_ry",
            "jaw_rz",
            "jaw_tx",
            "jaw_ty",
            "jaw_tz",
        ],
        "implicit_material_and_all_six_jaw_finite_differences": implicit_checks,
        "queued_expression_gradient_errors": queued_errors,
        "forward": runtime.last_forward,
        "adjoint": runtime.last_adjoint,
        "solves": runtime.forward_count,
    }
    archive_sources(cfg.output_dir)
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics(
        {
            "contact/max_energy_fd_relative_error": max(
                item["relative_error"] for item in energy_checks
            ),
            "contact/max_hvp_relative_error": max(
                item["relative_error"] for item in hessian_checks
            ),
            "contact/max_implicit_relative_error": max(
                item["relative_error"] for item in implicit_checks
            ),
            "contact/max_queued_error": max(queued_errors.values()),
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
    os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
    cherries.main(main, profile=ProfileJoint)
