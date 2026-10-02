"""Validate the complete shared-field, activation, skin, and jaw composition."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import archive_sources, write_json
from joint_equilibrium import Equilibrium, configure_cuda, rigid_displacement
from joint_fields import (
    BULK_TISSUES,
    SharedFieldParameters,
    activation_stresses_mpa,
    project_activation_,
    symmetric_coordinates,
    symmetric_matrices,
)
from joint_materials import StableNeoHookeanMembrane, StableNeoHookeanStress

FD_STEPS = (1.0e-3, 3.0e-3)
FD_RELATIVE_TOLERANCE = 2.0e-3
BULK_FRACTIONS = {
    "fat": [0.70, 0.20, 0.45, 0.15],
    "aponeurosis": [0.10, 0.55, 0.20, 0.25],
    "muscle": [0.20, 0.25, 0.35, 0.60],
}

from liblaf import cherries  # noqa: E402 - local experiment imports configure paths.
from liblaf.apple.common import (  # noqa: E402 - local experiment imports configure paths.
    FRACTION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)
from liblaf.apple.forward import (  # noqa: E402 - local experiment imports configure paths.
    Forward,
    ModelBuilder,
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("coupled-validation", mkdir=True)


def make_runtime() -> tuple[Equilibrium, torch.Tensor]:
    points = 0.01 * np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.3, 0.9, 0.0],
            [-0.3, -0.45, 0.8],
            [-0.3, -0.45, -0.8],
        ]
    )
    tets = np.asarray([[0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]])
    for tet in tets:
        if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0.0:
            tet[2], tet[3] = tet[3], tet[2]
    mesh = pv.UnstructuredGrid(
        np.column_stack((np.full(len(tets), 4), tets)).ravel(),
        np.full(len(tets), pv.CellType.TETRA),
        points,
    )
    fixed = np.repeat((np.arange(len(points)) > 0)[:, None], 3, axis=1)
    mesh.point_data["FixedMask"] = fixed
    mesh.point_data["FixedValue"] = np.zeros_like(points)

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    shared = SharedFieldParameters(dtype=torch.float64)
    for name in BULK_TISSUES:
        material = shared.config["materials"][name]
        mesh.cell_data[MU.vtk] = np.full(len(tets), material["mu_mpa"])
        mesh.cell_data[LAMBDA.vtk] = np.full(len(tets), material["lambda_code_mpa"])
        mesh.cell_data[FRACTION.vtk] = np.asarray(BULK_FRACTIONS[name])
        builder.add_potential(StableNeoHookeanStress.from_pyvista(mesh, name=name))

    # This triangle includes the free center (global node 0), so both its
    # resultant and stiffness change the solved observation rather than only
    # contributing reactions on fixed nodes.
    skin = pv.PolyData(points[[0, 1, 2]], faces=np.asarray([3, 0, 1, 2]))
    skin.point_data[GLOBAL_POINT_ID.vtk] = np.asarray([0, 1, 2])
    skin_material = shared.config["materials"]["skin"]
    skin.cell_data[MU.vtk] = np.asarray([skin_material["mu_mpa"]])
    skin.cell_data[LAMBDA.vtk] = np.asarray([skin_material["lambda_code_mpa"]])
    skin.cell_data[FRACTION.vtk] = np.ones(1)
    builder.add_potential(
        StableNeoHookeanMembrane.from_pyvista(
            skin, name="skin", thickness=skin_material["thickness_m"]
        )
    )
    runtime = Equilibrium(
        Forward(builder.finalize()),
        rtol=1.0e-8,
        atol=1.0e-15,
        adjoint_rtol=1.0e-10,
    )
    return runtime, torch.as_tensor(points)


def initial_shared(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    values = torch.tensor(
        [
            0.025,
            -0.018,
            0.012,
            0.009,
            -0.006,
            0.007,
            -0.020,
            0.017,
            0.011,
            -0.008,
            0.006,
            -0.005,
            0.018,
            -0.014,
            0.022,
            0.007,
            0.005,
            -0.009,
            0.012,
            0.04,
        ],
        dtype=dtype,
        device=device,
    )
    checker = SharedFieldParameters(dtype=dtype, device=device)
    with torch.no_grad():
        checker.coefficients.copy_(values)
        receipt = checker.project_()
    assert receipt["projection_coordinate_rms"] < 1.0e-14
    return checker.coefficients.detach().clone()


def initial_activation(
    dtype: torch.dtype, device: torch.device, maximum: float
) -> torch.Tensor:
    matrices = torch.stack(
        [
            torch.tensor(
                [[0.055, 0.006, -0.003], [0.006, 0.035, 0.004], [-0.003, 0.004, 0.022]],
                dtype=dtype,
                device=device,
            ),
            torch.tensor(
                [[0.032, -0.004, 0.003], [-0.004, 0.046, 0.005], [0.003, 0.005, 0.026]],
                dtype=dtype,
                device=device,
            ),
            torch.tensor(
                [[0.044, 0.005, 0.004], [0.005, 0.028, -0.003], [0.004, -0.003, 0.038]],
                dtype=dtype,
                device=device,
            ),
            torch.tensor(
                [[0.025, -0.003, 0.002], [-0.003, 0.039, 0.004], [0.002, 0.004, 0.048]],
                dtype=dtype,
                device=device,
            ),
        ]
    )
    coordinates = symmetric_coordinates(matrices)
    project_activation_(coordinates, maximum)
    eigenvalues = torch.linalg.eigvalsh(symmetric_matrices(coordinates))
    assert float(eigenvalues.min()) > 0.0
    assert float(eigenvalues.max()) < maximum
    return coordinates


def relative_scalar_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.maximum(actual.abs(), expected.abs()).clamp_min(1.0e-10)
    return float(((actual - expected).abs() / scale).detach())


def main(  # noqa: C901, PLR0915 - integrated receipt exercises every coupled block.
    cfg: Config,
) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cuda()
    runtime, points = make_runtime()
    base = runtime.forward.model.get_materials()
    shared = SharedFieldParameters(dtype=points.dtype, device=points.device)
    shared_values = initial_shared(points.dtype, points.device)
    with torch.no_grad():
        shared.coefficients.copy_(shared_values)
    activation_values = initial_activation(
        points.dtype, points.device, shared.activation_maximum_dimensionless
    )
    pose_values = torch.tensor(
        [0.16, -0.11, 0.09, 0.13, -0.08, 0.10],
        dtype=points.dtype,
        device=points.device,
    )
    jaw_ids = torch.tensor([1, 2], dtype=torch.int64, device=points.device)
    pivot = points[1:].mean(dim=0)
    seed = torch.zeros_like(points)
    targets = (
        torch.tensor(
            [0.00011, -0.00008, 0.00004], dtype=points.dtype, device=points.device
        ),
        torch.tensor(
            [-0.00007, 0.00010, -0.00003], dtype=points.dtype, device=points.device
        ),
    )

    def materials(activation: torch.Tensor) -> dict:
        values = {name: dict(fields) for name, fields in base.items()}
        bulk = shared.bulk_stresses_mpa()
        active = activation_stresses_mpa(activation, shared.activation_reference_mpa)
        for index, name in enumerate(BULK_TISSUES):
            total = bulk[index].expand(4, 3, 3).contiguous()
            if name == "muscle":
                total = total + active
            values[name]["active_stress"] = total
        values["skin"]["baseline_stress"] = shared.skin_resultant_mpa_m()[None]
        multiplier = shared.skin_stiffness_multiplier()
        values["skin"]["mu"] = base["skin"]["mu"] * multiplier
        values["skin"]["lmbda"] = base["skin"]["lmbda"] * multiplier
        return values

    def boundary(pose_parameters: torch.Tensor) -> torch.Tensor:
        scale = pose_parameters.new_tensor([0.02, 0.02, 0.02, 0.001, 0.001, 0.001])
        pose = pose_parameters * scale
        displacement = torch.zeros_like(points).index_copy(
            0,
            jaw_ids,
            rigid_displacement(points[jaw_ids], pivot, pose),
        )
        return displacement.flatten()[runtime.forward.model.dof_map.fixed_indices]

    def solve(
        activation: torch.Tensor,
        pose: torch.Tensor,
        *,
        key: str,
        initial_seed: torch.Tensor = seed,
    ) -> torch.Tensor:
        return runtime.solve(
            materials(activation), boundary(pose), initial_seed, key=key
        )

    def loss(solution: torch.Tensor, expression: int) -> torch.Tensor:
        residual = (solution[0] - targets[expression]) * 1000.0
        return residual.square().sum() + 0.17 * residual.prod()

    activation = activation_values.detach().clone().requires_grad_()
    pose = pose_values.detach().clone().requires_grad_()
    single_solution = solve(activation, pose, key="coupled-fd")
    single_loss = loss(single_solution, 0)
    shared_gradient, activation_gradient, pose_gradient = torch.autograd.grad(
        single_loss, (shared.coefficients, activation, pose)
    )
    gradients = {
        "shared": shared_gradient.detach(),
        "activation": activation_gradient.detach(),
        "jaw": pose_gradient.detach(),
    }
    for name, value in gradients.items():
        assert torch.isfinite(value).all(), name
    for block, value in {
        "fat": shared_gradient[:6],
        "aponeurosis": shared_gradient[6:12],
        "muscle": shared_gradient[12:18],
        "skin_baseline": shared_gradient[18:19],
        "skin_stiffness": shared_gradient[19:20],
        "activation": activation_gradient,
        "jaw": pose_gradient,
    }.items():
        assert float(torch.linalg.vector_norm(value)) > 1.0e-10, block

    directions = {
        "fat": torch.tensor([0.5, -0.3, 0.4, -0.2, 0.6, -0.1], device=points.device),
        "aponeurosis": torch.tensor(
            [-0.2, 0.5, 0.3, 0.4, -0.1, 0.6], device=points.device
        ),
        "muscle": torch.tensor([0.4, 0.2, -0.5, 0.3, 0.6, -0.1], device=points.device),
        "skin_baseline": torch.ones(1, dtype=points.dtype, device=points.device),
        "skin_stiffness": torch.ones(1, dtype=points.dtype, device=points.device),
        # Stay on an interior PSD ray so both central differences remain in the
        # admissible activation set and use the same well-conditioned root.
        "activation": activation_values.clone(),
        "jaw": torch.tensor([0.4, -0.3, 0.2, 0.6, -0.5, 0.1], device=points.device),
    }
    directions = {
        name: value.to(dtype=points.dtype) / torch.linalg.vector_norm(value)
        for name, value in directions.items()
    }
    blocks = {
        "fat": ("shared", slice(0, 6)),
        "aponeurosis": ("shared", slice(6, 12)),
        "muscle": ("shared", slice(12, 18)),
        "skin_baseline": ("shared", slice(18, 19)),
        "skin_stiffness": ("shared", slice(19, 20)),
        "activation": ("activation", (...,)),
        "jaw": ("jaw", (...,)),
    }

    def explicit_newton(
        material_values: dict,
        fixed_values: torch.Tensor,
        initial_seed: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            model = runtime.forward.model
            model.set_materials(material_values)
            model.dof_map.fixed_values = fixed_values
            initial_free = model.dof_map.to_free(initial_seed)
            state = model.State(u=model.dof_map.to_full(initial_free))
            free = model.dof_map.to_free(state.u)
            gradient_norms = []
            line_search_steps = []
            for _iteration in range(25):
                gradient = model.dof_map.to_free_grad(model.grad(state))
                gradient_norm = float(torch.linalg.vector_norm(gradient))
                gradient_norms.append(gradient_norm)
                if gradient_norm <= 1.0e-13:
                    break
                basis = torch.eye(
                    model.n_free, dtype=points.dtype, device=points.device
                )
                columns = []
                for column in basis:
                    product = model.hess_prod(state, model.dof_map.to_full_grad(column))
                    columns.append(model.dof_map.to_free_grad(product))
                hessian = torch.stack(columns, dim=1)
                newton_step = torch.linalg.solve(hessian, gradient)
                energy = float(model.fun(state))
                accepted = False
                alpha = 1.0
                for line_step in range(20):
                    candidate_free = free - alpha * newton_step
                    candidate = model.State(u=model.dof_map.to_full(candidate_free))
                    if float(model.fun(candidate)) <= energy:
                        state = candidate
                        free = candidate_free
                        line_search_steps.append(line_step)
                        accepted = True
                        break
                    alpha *= 0.5
                assert accepted, (gradient_norms, line_search_steps)
            else:
                raise AssertionError((gradient_norms, line_search_steps))
            receipt = {
                "implementation": "explicit_3x3_newton",
                "steps": len(gradient_norms) - 1,
                "initial_gradient_norm": gradient_norms[0],
                "final_gradient_norm": gradient_norms[-1],
                "gradient_tolerance": 1.0e-13,
                "line_search_steps": line_search_steps,
            }
            return state.u.detach().clone(), receipt

    def scalar_at(
        shared_at: torch.Tensor, activation_at: torch.Tensor, pose_at: torch.Tensor
    ) -> tuple[float, dict]:
        with torch.no_grad():
            shared.coefficients.copy_(shared_at)
            solution, receipt = explicit_newton(
                materials(activation_at), boundary(pose_at), seed
            )
            return float(loss(solution, 0)), receipt

    finite_differences: dict[str, dict] = {}
    for block, (owner, where) in blocks.items():
        direction = directions[block]
        if owner == "shared":
            analytic = (gradients[owner][where] * direction).sum()
        else:
            analytic = (gradients[owner] * direction).sum()
        step_receipts = {}
        for step in FD_STEPS:
            shared_plus, shared_minus = shared_values.clone(), shared_values.clone()
            activation_plus = activation_values.clone()
            activation_minus = activation_values.clone()
            pose_plus, pose_minus = pose_values.clone(), pose_values.clone()
            if owner == "shared":
                shared_plus[where] += step * direction
                shared_minus[where] -= step * direction
            elif owner == "activation":
                activation_plus += step * direction
                activation_minus -= step * direction
            else:
                pose_plus += step * direction
                pose_minus -= step * direction
            plus, plus_solve = scalar_at(shared_plus, activation_plus, pose_plus)
            minus, minus_solve = scalar_at(shared_minus, activation_minus, pose_minus)
            finite = torch.tensor((plus - minus) / (2.0 * step), device=points.device)
            error = relative_scalar_error(analytic, finite)
            step_receipts[f"{step:.0e}"] = {
                "finite_difference": float(finite),
                "relative_error": error,
                "plus_solve": plus_solve,
                "minus_solve": minus_solve,
            }
            assert error < FD_RELATIVE_TOLERANCE, (block, step_receipts)
        finite_differences[block] = {
            "analytic": float(analytic),
            "steps": step_receipts,
            "maximum_relative_error": max(
                item["relative_error"] for item in step_receipts.values()
            ),
        }
        print("finite difference", block, finite_differences[block], flush=True)
    with torch.no_grad():
        shared.coefficients.copy_(shared_values)

    def expression_gradients(expression: int) -> tuple[torch.Tensor, ...]:
        activation_i = (
            activation_values * (1.0 if expression == 0 else 0.73)
        ).requires_grad_()
        pose_i = (pose_values * (1.0 if expression == 0 else -0.61)).requires_grad_()
        value = loss(
            solve(activation_i, pose_i, key=f"separate-{expression}"), expression
        )
        return tuple(
            item.detach()
            for item in torch.autograd.grad(
                value, (shared.coefficients, activation_i, pose_i)
            )
        )

    separate_0 = expression_gradients(0)
    separate_1 = expression_gradients(1)
    activation_0 = activation_values.detach().clone().requires_grad_()
    activation_1 = (0.73 * activation_values).detach().requires_grad_()
    pose_0 = pose_values.detach().clone().requires_grad_()
    pose_1 = (-0.61 * pose_values).detach().requires_grad_()
    solution_0 = solve(activation_0, pose_0, key="interleaved-0")
    solution_1 = solve(activation_1, pose_1, key="interleaved-1")
    combined = torch.autograd.grad(
        loss(solution_0, 0) + loss(solution_1, 1),
        (shared.coefficients, activation_0, pose_0, activation_1, pose_1),
    )

    def vector_relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(expected).clamp_min(1.0e-14)
        return float(torch.linalg.vector_norm(actual.detach() - expected) / denominator)

    isolation = {
        "shared_sum": vector_relative(combined[0], separate_0[0] + separate_1[0]),
        "activation_0": vector_relative(combined[1], separate_0[1]),
        "jaw_0": vector_relative(combined[2], separate_0[2]),
        "activation_1": vector_relative(combined[3], separate_1[1]),
        "jaw_1": vector_relative(combined[4], separate_1[2]),
    }
    assert max(isolation.values()) < 1.0e-6, isolation

    zero_shared = torch.zeros_like(shared_values)
    zero_activation = torch.zeros_like(activation_values)
    zero_pose = torch.zeros_like(pose_values)
    perturbation = torch.zeros_like(seed)
    perturbation[0] = perturbation.new_tensor([4.0e-5, -3.0e-5, 2.0e-5])
    with torch.no_grad():
        shared.coefficients.copy_(zero_shared)
        assert runtime.forward.last_solution is not None
        neutral_reference = solve(
            zero_activation, zero_pose, key="neutral-reference", initial_seed=seed
        )
        neutral_reference_solve = runtime.last_forward
        assert runtime.forward.last_solution is None
        neutral_perturbed, neutral_perturbed_solve = explicit_newton(
            materials(zero_activation),
            boundary(zero_pose),
            perturbation,
        )
    neutral = {
        "reference_displacement_norm_m": float(
            torch.linalg.vector_norm(neutral_reference)
        ),
        "perturbed_resolve_difference_m": float(
            torch.linalg.vector_norm(neutral_perturbed - neutral_reference)
        ),
        "activation_norm": float(torch.linalg.vector_norm(zero_activation)),
        "reference_solve": neutral_reference_solve,
        "perturbed_solve": neutral_perturbed_solve,
    }
    assert neutral["reference_displacement_norm_m"] < 1.0e-10, neutral
    assert neutral["perturbed_resolve_difference_m"] < 1.0e-9, neutral
    assert neutral_reference_solve["result"] == "initial_equilibrium"
    assert neutral_reference_solve["steps"] == 0

    receipt = {
        "success": True,
        "scope": (
            "four-tet, one-skin-triangle synthetic coupled derivative check; "
            "not a face readiness receipt"
        ),
        "shared_coefficient_count": shared_values.numel(),
        "activation_shape": list(activation_values.shape),
        "skin_triangle_global_nodes": [0, 1, 2],
        "bulk_fractions": BULK_FRACTIONS,
        "gradient_norms": {
            "fat": float(torch.linalg.vector_norm(shared_gradient[:6])),
            "aponeurosis": float(torch.linalg.vector_norm(shared_gradient[6:12])),
            "muscle": float(torch.linalg.vector_norm(shared_gradient[12:18])),
            "skin_baseline": float(shared_gradient[18].abs()),
            "skin_stiffness": float(shared_gradient[19].abs()),
            "activation": float(torch.linalg.vector_norm(activation_gradient)),
            "jaw": float(torch.linalg.vector_norm(pose_gradient)),
        },
        "finite_differences": finite_differences,
        "expression_gradient_isolation": isolation,
        "neutral_zero_activation": neutral,
        "forward": runtime.last_forward,
        "adjoint": runtime.last_adjoint,
        "solves": runtime.forward_count,
    }
    archive_sources(cfg.output_dir)
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics(
        {
            "coupled/max_fd_relative_error": max(
                item["maximum_relative_error"] for item in finite_differences.values()
            ),
            "coupled/max_expression_isolation_error": max(isolation.values()),
            "coupled/neutral_perturb_resolve_m": neutral[
                "perturbed_resolve_difference_m"
            ],
        }
    )
    print(receipt)


if __name__ == "__main__":
    cherries.main(main)
