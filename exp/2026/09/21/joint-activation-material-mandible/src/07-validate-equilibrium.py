"""Check implicit material/jaw derivatives and interleaved expression snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import HISTORICAL, archive_sources, write_json
from joint_equilibrium import Equilibrium, configure_cuda, rigid_displacement

sys.path.insert(0, str(HISTORICAL))
from tensor_active import StableNeoHookeanTensorActive

from liblaf import cherries
from liblaf.apple.common import FRACTION, LAMBDA, MU
from liblaf.apple.forward import Forward, ModelBuilder


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("equilibrium-validation", mkdir=True)


def make_runtime():
    points = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [-0.3, 0.9, 0],
                [-0.3, -0.45, 0.8],
                [-0.3, -0.45, -0.8],
            ]
        )
        * 0.01
    )
    tets = np.array([[0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]])
    for tet in tets:
        if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0:
            tet[2], tet[3] = tet[3], tet[2]
    mesh = pv.UnstructuredGrid(
        np.column_stack((np.full(4, 4), tets)).ravel(),
        np.full(4, pv.CellType.TETRA),
        points,
    )
    mesh.cell_data[LAMBDA.vtk] = np.full(4, 0.08)
    mesh.cell_data[MU.vtk] = np.full(4, 0.02)
    mesh.cell_data[FRACTION.vtk] = np.ones(4)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    mesh.point_data["FixedMask"] = np.repeat((np.arange(5) > 0)[:, None], 3, axis=1)
    mesh.point_data["FixedValue"] = np.zeros((5, 3))
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookeanTensorActive.from_pyvista(mesh, name="bulk"))
    runtime = Equilibrium(
        Forward(builder.finalize()), rtol=1e-8, atol=1e-15, adjoint_rtol=1e-10
    )
    return runtime, torch.as_tensor(points)


def main(cfg: Config):  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    configure_cuda()
    archive_sources(cfg.output_dir)
    runtime, points = make_runtime()
    base = runtime.forward.model.get_materials()
    jaw_ids = torch.tensor([1, 2], dtype=torch.int64)
    seed = torch.zeros_like(points)
    stress_pattern = torch.zeros((4, 3, 3))
    stress_pattern[0] = torch.diag(torch.tensor([1.0, 0.6, 0.3]))
    target = torch.tensor([0.00013, -0.00007, 0.00002])

    def solve(parameters: torch.Tensor, key: str):
        materials = {name: dict(values) for name, values in base.items()}
        materials["bulk"]["active_stress"] = 0.002 * parameters[0] * stress_pattern
        pose = parameters[1:] * parameters.new_tensor(
            [0.02, 0.02, 0.02, 0.001, 0.001, 0.001]
        )
        boundary = torch.zeros_like(points).index_copy(
            0, jaw_ids, rigid_displacement(points[jaw_ids], points.mean(0), pose)
        )
        fixed = boundary.flatten()[runtime.forward.model.dof_map.fixed_indices]
        return runtime.solve(materials, fixed, seed, key=key)

    def loss(value: torch.Tensor):
        # Only the unconstrained center is observed: direct jaw-node terms cannot
        # conceal an omitted implicit Dirichlet derivative.
        return ((value[0] - target) * 1000).square().sum()

    initial = torch.tensor([0.2, 0.3, -0.2, 0.1, 0.2, -0.15, 0.1])
    parameter = initial.detach().clone().requires_grad_()
    value = loss(solve(parameter, "single"))
    (gradient,) = torch.autograd.grad(value, parameter)
    checks = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = []
        for index in range(7):
            direction = torch.zeros_like(initial)
            direction[index] = step
            plus = float(loss(solve(initial + direction, "fd")))
            minus = float(loss(solve(initial - direction, "fd")))
            finite.append((plus - minus) / (2 * step))
        fd = torch.tensor(finite)
        relative = float(
            torch.linalg.vector_norm(fd - gradient) / torch.linalg.vector_norm(gradient)
        )
        scaled_errors = (fd - gradient).abs() / torch.maximum(
            fd.abs(), gradient.abs()
        ).clamp_min(1e-8)
        checks.append(
            {
                "step": step,
                "relative_error": relative,
                "max_component_relative": float(scaled_errors.max()),
                "finite_difference": finite,
            }
        )
        assert relative < 0.02, checks[-1]
        assert scaled_errors.max() < 0.02, checks[-1]

    def combined(order: list[int], *, interleaved: bool):
        p = initial.detach().clone().requires_grad_()
        outputs = []
        for index in order:
            u = solve(p * (1 if index == 0 else 0.7), f"expression-{index}")
            if interleaved:
                outputs.append(loss(u))
            else:
                loss(u).backward()
        if interleaved:
            sum(outputs).backward()
        return p.grad.detach().clone()

    reference = combined([0, 1], interleaved=False)
    errors = {}
    for name, order, interleaved in [
        ("reversed", [1, 0], False),
        ("queued", [0, 1], True),
        ("queued_reversed", [1, 0], True),
    ]:
        grad = combined(order, interleaved=interleaved)
        errors[name] = float(
            torch.linalg.vector_norm(grad - reference)
            / torch.linalg.vector_norm(reference)
        )
        assert errors[name] < 1e-6, errors
    receipt = {
        "success": True,
        "scope": "four-tet free-center synthetic problem; not a face readiness receipt",
        "material_and_six_pose_gradients": gradient.cpu().tolist(),
        "finite_differences": checks,
        "multi_expression_gradient_errors": errors,
        "forward": runtime.last_forward,
        "adjoint": runtime.last_adjoint,
        "solves": runtime.forward_count,
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics(
        {
            "finite_difference_relative_max": max(c["relative_error"] for c in checks),
            "expression_gradient_relative_max": max(errors.values()),
        }
    )
    print(receipt)


if __name__ == "__main__":
    cherries.main(main)
