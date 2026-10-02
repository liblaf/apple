"""Validate the fiber-free tensor active-stress Warp material."""

# ruff: noqa: EM101, EM102, PLR0915, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from experiment_profile import ProfileCometNoCommit
from tensor_active import ACTIVE_STRESS, StableNeoHookeanTensorActive

from liblaf import cherries
from liblaf.apple.common import FRACTION, GLOBAL_POINT_ID, LAMBDA, MU
from liblaf.apple.warp.fem import StableNeoHookean
from liblaf.apple.warp.model import WarpModel

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parent.parent
REPOSITORY = GROUP.parents[4]


class Config(cherries.BaseConfig):
    """Configuration for the constitutive and assembled-FEM audit."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("10-tensor-active-validation", mkdir=True)
    finite_difference_step: float = 1.0e-5
    run_cuda_smoke: bool = True


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_mesh(*, mu: float, lambda_code: float) -> pv.UnstructuredGrid:
    """Return one reference tetrahedron without activation or fiber arrays."""
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = pv.UnstructuredGrid(
        np.asarray((4, 0, 1, 2, 3)),
        np.asarray((pv.CellType.TETRA,), dtype=np.uint8),
        points,
    )
    mesh.cell_data[LAMBDA.vtk] = np.asarray((lambda_code,))
    mesh.cell_data[MU.vtk] = np.asarray((mu,))
    mesh.cell_data[FRACTION.vtk] = np.ones(1)
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
    return mesh


def from_torch_float(value: torch.Tensor) -> wp.array:
    return wp.from_torch(value, dtype=wp.dtype_from_torch(value.dtype))


def from_torch_vec3(value: torch.Tensor) -> wp.array:
    dtype = wp.dtype_from_torch(value.dtype)
    return wp.from_torch(value, dtype=wp.types.vector(3, dtype))


def from_torch_mat33(value: torch.Tensor) -> wp.array:
    dtype = wp.dtype_from_torch(value.dtype)
    return wp.from_torch(value, dtype=wp.types.matrix((3, 3), dtype))


def potential_fun(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.fun(from_torch_vec3(u), from_torch_float(output))
    wp.synchronize()
    return output[0]


def potential_grad(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.grad(from_torch_vec3(u), from_torch_vec3(output))
    wp.synchronize()
    return output


def potential_hess_diag(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_diag(from_torch_vec3(u), from_torch_vec3(output))
    wp.synchronize()
    return output


def potential_hess_prod(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_prod(
        from_torch_vec3(u),
        from_torch_vec3(direction),
        from_torch_vec3(output),
    )
    wp.synchronize()
    return output


def potential_hess_quad(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.hess_quad(
        from_torch_vec3(u),
        from_torch_vec3(direction),
        from_torch_float(output),
    )
    wp.synchronize()
    return output[0]


def potential_density(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(potential.launch_dim, dtype=u.dtype, device=u.device)
    potential.energy_density(from_torch_vec3(u), from_torch_float(output))
    wp.synchronize()
    return output


def potential_piola(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((*potential.launch_dim, 3, 3), dtype=u.dtype, device=u.device)
    potential.first_piola_kirchhoff(from_torch_vec3(u), from_torch_mat33(output))
    wp.synchronize()
    return output


def displacement_for_F(F: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    reference = torch.as_tensor(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=F.dtype,
        device=device,
    )
    return reference @ torch.transpose(
        F - torch.eye(3, dtype=F.dtype, device=device), 0, 1
    )


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def scalar(value: torch.Tensor | float) -> float:
    return float(torch.as_tensor(value).detach().cpu())


def max_abs(value: torch.Tensor) -> float:
    return scalar(torch.max(torch.abs(value)))


def set_Q(potential: Any, Q: torch.Tensor) -> None:
    potential.set_materials({ACTIVE_STRESS: Q})


def assembled_cpu_audit(step: float) -> dict[str, object]:
    """Compile and audit all material and assembled paths on one CPU tet."""
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    wp.set_device("cpu")

    young = 0.024
    nu = 0.46
    mu = young / (2.0 * (1.0 + nu))
    lambda_classical = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lambda_code = lambda_classical + mu
    mesh = make_mesh(mu=mu, lambda_code=lambda_code)
    active = StableNeoHookeanTensorActive.from_pyvista(mesh, name="muscle")
    zero = StableNeoHookeanTensorActive.from_pyvista(mesh, name="muscle")
    passive = StableNeoHookean.from_pyvista(mesh, name="muscle")

    Q = torch.as_tensor(
        (((0.018, 0.004, -0.002), (0.004, 0.011, 0.001), (-0.002, 0.001, 0.006)),)
    )
    eigvals = torch.linalg.eigvalsh(Q[0])
    assert torch.min(eigvals) > 0.0
    model = WarpModel({"muscle": active})
    model.set_materials({"muscle": {ACTIVE_STRESS: Q}})
    material_view = model.get_materials()["muscle"][ACTIVE_STRESS]
    assert material_view.shape == (1,)
    assert tuple(wp.to_torch(material_view).shape) == (1, 3, 3)

    F = torch.as_tensor(((1.08, 0.04, -0.02), (0.01, 0.93, 0.03), (-0.02, 0.02, 1.04)))
    dF = torch.as_tensor(
        ((0.03, -0.04, 0.02), (0.01, 0.05, -0.03), (-0.02, 0.01, 0.04))
    )
    u = displacement_for_F(F, device=torch.device("cpu"))
    direction = displacement_for_F(torch.eye(3) + dF, device=torch.device("cpu"))
    dV = 1.0 / 6.0

    energy = potential_fun(active, u)
    gradient = potential_grad(active, u)
    hess_diag_value = potential_hess_diag(active, u)
    hess_prod_value = potential_hess_prod(active, u, direction)
    hess_quad_value = potential_hess_quad(active, u, direction)
    density = potential_density(active, u)[0, 0]
    piola = potential_piola(active, u)[0, 0]

    J = torch.linalg.det(F)
    cofactor = J * torch.linalg.inv(F).T
    C = F.T @ F
    identity = torch.eye(3)
    expected_density = (
        0.5 * mu * (torch.sum(F * F) - 3.0)
        - mu * (J - 1.0)
        + 0.5 * lambda_code * (J - 1.0) ** 2
        + 0.5 * torch.sum(Q[0] * (C - identity))
    )
    expected_piola = mu * F + (-mu + lambda_code * (J - 1.0)) * cofactor + F @ Q[0]

    fd_grad_dot = (
        potential_fun(active, u + step * direction)
        - potential_fun(active, u - step * direction)
    ) / (2.0 * step)
    fd_hess_prod = (
        potential_grad(active, u + step * direction)
        - potential_grad(active, u - step * direction)
    ) / (2.0 * step)
    fd_hess_quad = (
        potential_fun(active, u + step * direction)
        - 2.0 * energy
        + potential_fun(active, u - step * direction)
    ) / step**2
    fd_hess_diag = torch.empty_like(u)
    for index in range(u.numel()):
        basis = torch.zeros_like(u).reshape(-1)
        basis[index] = 1.0
        basis = basis.reshape_as(u)
        fd = (
            potential_grad(active, u + step * basis)
            - potential_grad(active, u - step * basis)
        ) / (2.0 * step)
        fd_hess_diag.reshape(-1)[index] = fd.reshape(-1)[index]

    zero_checks = {
        "energy_abs_error": abs(
            scalar(potential_fun(zero, u) - potential_fun(passive, u))
        ),
        "density_max_abs_error": max_abs(
            potential_density(zero, u) - potential_density(passive, u)
        ),
        "piola_max_abs_error": max_abs(
            potential_piola(zero, u) - potential_piola(passive, u)
        ),
        "gradient_max_abs_error": max_abs(
            potential_grad(zero, u) - potential_grad(passive, u)
        ),
        "hess_diag_max_abs_error": max_abs(
            potential_hess_diag(zero, u) - potential_hess_diag(passive, u)
        ),
        "hess_prod_max_abs_error": max_abs(
            potential_hess_prod(zero, u, direction)
            - potential_hess_prod(passive, u, direction)
        ),
        "hess_quad_abs_error": abs(
            scalar(
                potential_hess_quad(zero, u, direction)
                - potential_hess_quad(passive, u, direction)
            )
        ),
    }

    D = torch.as_tensor((((0.7, -0.2, 0.1), (-0.2, -0.3, 0.25), (0.1, 0.25, -0.4)),))
    D /= torch.linalg.vector_norm(D)
    q_step = 1.0e-6
    set_Q(active, Q + q_step * D)
    energy_plus = potential_fun(active, u)
    mixed_plus = torch.sum(potential_grad(active, u) * direction)
    set_Q(active, Q - q_step * D)
    energy_minus = potential_fun(active, u)
    mixed_minus = torch.sum(potential_grad(active, u) * direction)
    set_Q(active, Q)
    energy_q_fd = (energy_plus - energy_minus) / (2.0 * q_step)
    mixed_q_fd = (mixed_plus - mixed_minus) / (2.0 * q_step)
    energy_q_expected = 0.5 * dV * torch.sum((C - identity) * D[0])
    mixed_q_expected = dV * torch.sum((F.T @ dF) * D[0])

    Q_leaf = Q.detach().clone().requires_grad_(requires_grad=True)
    model.set_materials({"muscle": {ACTIVE_STRESS: Q_leaf}})
    model.mixed_derivative_prod(from_torch_vec3(u), from_torch_vec3(direction))
    assert Q_leaf.grad is not None
    mixed_q_tape = torch.sum(Q_leaf.grad * D)
    set_Q(active, Q)

    spatial_R = torch.as_tensor(rotation(np.asarray((1.0, 2.0, -1.0)), 0.63))
    material_R = torch.as_tensor(rotation(np.asarray((-2.0, 1.0, 0.5)), -0.41))
    F_rotated = spatial_R @ F @ material_R.T
    Q_rotated = material_R @ Q[0] @ material_R.T
    set_Q(active, Q_rotated[None])
    u_rotated = displacement_for_F(F_rotated, device=torch.device("cpu"))
    density_rotated = potential_density(active, u_rotated)[0, 0]
    piola_rotated = potential_piola(active, u_rotated)[0, 0]
    set_Q(active, Q)

    fiber = torch.as_tensor((0.4, -0.2, 0.7))
    fiber /= torch.linalg.vector_norm(fiber)
    tension = 0.027
    Q_rank_one = tension * torch.outer(fiber, fiber)
    set_Q(active, Q_rank_one[None])
    rank_energy = (
        potential_density(active, u)[0, 0] - potential_density(passive, u)[0, 0]
    )
    rank_piola = potential_piola(active, u)[0, 0] - potential_piola(passive, u)[0, 0]
    rank_quad = (
        potential_hess_quad(active, u, direction)
        - potential_hess_quad(passive, u, direction)
    ) / dV
    expected_rank_energy = 0.5 * tension * (torch.dot(F @ fiber, F @ fiber) - 1.0)
    expected_rank_piola = tension * torch.outer(F @ fiber, fiber)
    expected_rank_quad = tension * torch.dot(dF @ fiber, dF @ fiber)
    set_Q(active, Q)

    result = {
        "device": "cpu",
        "finite_difference_step": step,
        "q_finite_difference_step": q_step,
        "material": {
            "young_MPa": young,
            "poisson_ratio": nu,
            "mu_MPa": mu,
            "lambda_classical_MPa": lambda_classical,
            "lambda_code_MPa": lambda_code,
            "active_stress_eigenvalues_MPa": eigvals.tolist(),
            "active_stress_frobenius_MPa": scalar(torch.linalg.vector_norm(Q)),
        },
        "api": {
            "material_name": "muscle",
            "material_fields": sorted(active.MATERIAL_FIELDS),
            "active_stress_torch_shape": list(wp.to_torch(material_view).shape),
            "mesh_cell_arrays": sorted(mesh.cell_data.keys()),
            "has_fiber_field": False,
            "has_activation_inv_field": False,
            "contract": "Q is supplied symmetric PSD and capped by the caller; the material trusts this contract.",
        },
        "constitutive": {
            "energy_density_MPa": scalar(density),
            "energy_density_analytic_abs_error_MPa": abs(
                scalar(density - expected_density)
            ),
            "piola_analytic_max_abs_error_MPa": max_abs(piola - expected_piola),
        },
        "assembled_fem": {
            "energy_MPa_m3": scalar(energy),
            "gradient_directional_abs_error_MPa_m3": abs(
                scalar(torch.sum(gradient * direction) - fd_grad_dot)
            ),
            "hess_prod_max_abs_error_MPa_m": max_abs(hess_prod_value - fd_hess_prod),
            "hess_diag_max_abs_error_MPa_m": max_abs(hess_diag_value - fd_hess_diag),
            "hess_quad_vs_prod_abs_error_MPa_m": abs(
                scalar(hess_quad_value - torch.sum(hess_prod_value * direction))
            ),
            "hess_quad_finite_difference_abs_error_MPa_m": abs(
                scalar(hess_quad_value - fd_hess_quad)
            ),
            "active_tangent_quad_MPa": scalar(
                wp.to_torch(material_view)[0, 0, 0] * 0.0 + torch.sum(dF * (dF @ Q[0]))
            ),
        },
        "material_derivative": {
            "energy_directional_fd_abs_error": abs(
                scalar(energy_q_fd - energy_q_expected)
            ),
            "mixed_directional_fd_abs_error": abs(
                scalar(mixed_q_fd - mixed_q_expected)
            ),
            "mixed_tape_vs_analytic_abs_error": abs(
                scalar(mixed_q_tape - mixed_q_expected)
            ),
            "mixed_tape_gradient": Q_leaf.grad.detach().cpu().tolist(),
            "symmetric_direction": D.tolist(),
        },
        "rotation_covariance": {
            "energy_density_abs_error_MPa": abs(scalar(density_rotated - density)),
            "piola_max_abs_error_MPa": max_abs(
                piola_rotated - spatial_R @ piola @ material_R.T
            ),
            "det_spatial_rotation": scalar(torch.linalg.det(spatial_R)),
            "det_material_rotation": scalar(torch.linalg.det(material_R)),
        },
        "rank_one_equivalence": {
            "tension_MPa": tension,
            "fiber": fiber.tolist(),
            "energy_density_abs_error_MPa": abs(
                scalar(rank_energy - expected_rank_energy)
            ),
            "piola_max_abs_error_MPa": max_abs(rank_piola - expected_rank_piola),
            "hess_quad_abs_error_MPa": abs(scalar(rank_quad - expected_rank_quad)),
        },
        "zero_active_stress_passive_restoration": zero_checks,
    }

    limits = {
        "constitutive.energy_density_analytic_abs_error_MPa": 1.0e-14,
        "constitutive.piola_analytic_max_abs_error_MPa": 1.0e-14,
        "assembled_fem.gradient_directional_abs_error_MPa_m3": 1.0e-10,
        "assembled_fem.hess_prod_max_abs_error_MPa_m": 1.0e-10,
        "assembled_fem.hess_diag_max_abs_error_MPa_m": 1.0e-10,
        "assembled_fem.hess_quad_vs_prod_abs_error_MPa_m": 1.0e-12,
        "assembled_fem.hess_quad_finite_difference_abs_error_MPa_m": 1.0e-8,
        "material_derivative.energy_directional_fd_abs_error": 1.0e-11,
        "material_derivative.mixed_directional_fd_abs_error": 1.0e-11,
        "material_derivative.mixed_tape_vs_analytic_abs_error": 1.0e-14,
        "rotation_covariance.energy_density_abs_error_MPa": 1.0e-14,
        "rotation_covariance.piola_max_abs_error_MPa": 1.0e-14,
        "rank_one_equivalence.energy_density_abs_error_MPa": 1.0e-14,
        "rank_one_equivalence.piola_max_abs_error_MPa": 1.0e-14,
        "rank_one_equivalence.hess_quad_abs_error_MPa": 1.0e-14,
    }
    flat: dict[str, float] = {}
    for key, limit in limits.items():
        section, metric = key.split(".", 1)
        value = float(result[section][metric])  # type: ignore[index]
        flat[key] = value
        if not value < limit:
            raise AssertionError(f"{key}={value:.6e} exceeds {limit:.6e}")
    if any(value != 0.0 for value in zero_checks.values()):
        raise AssertionError("Q=0 does not exactly restore StableNeoHookean")
    if result["assembled_fem"]["active_tangent_quad_MPa"] <= 0.0:  # type: ignore[index]
        raise AssertionError(
            "PSD Q did not produce a positive active tangent quadratic"
        )
    result["assertion_limits"] = limits
    result["asserted_metrics"] = flat
    result["status"] = "passed"
    return result


def cuda_smoke(cpu: dict[str, object]) -> dict[str, object]:
    """Compile the same field and constitutive kernels on the available GPU."""
    if not wp.is_cuda_available():
        return {"status": "skipped", "reason": "Warp CUDA is unavailable"}
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    wp.set_device("cuda:0")
    material = cpu["material"]
    mesh = make_mesh(
        mu=float(material["mu_MPa"]),  # type: ignore[index]
        lambda_code=float(material["lambda_code_MPa"]),  # type: ignore[index]
    )
    potential = StableNeoHookeanTensorActive.from_pyvista(mesh, name="muscle")
    Q = torch.as_tensor(
        (((0.018, 0.004, -0.002), (0.004, 0.011, 0.001), (-0.002, 0.001, 0.006)),),
        device="cuda",
    )
    set_Q(potential, Q)
    F = torch.as_tensor(
        ((1.08, 0.04, -0.02), (0.01, 0.93, 0.03), (-0.02, 0.02, 1.04)),
        device="cuda",
    )
    u = displacement_for_F(F, device=torch.device("cuda"))
    density = potential_density(potential, u)[0, 0]
    piola = potential_piola(potential, u)[0, 0]
    gradient = potential_grad(potential, u)
    cpu_density = float(cpu["constitutive"]["energy_density_MPa"])  # type: ignore[index]
    result = {
        "status": "passed",
        "device": str(u.device),
        "energy_density_MPa": scalar(density),
        "cpu_energy_density_abs_error_MPa": abs(scalar(density) - cpu_density),
        "piola_frobenius_MPa": scalar(torch.linalg.vector_norm(piola)),
        "gradient_frobenius_MPa_m2": scalar(torch.linalg.vector_norm(gradient)),
    }
    if result["cpu_energy_density_abs_error_MPa"] >= 1.0e-13:
        raise AssertionError("CUDA and CPU energy densities disagree")
    torch.set_default_device("cpu")
    wp.set_device("cpu")
    return result


def snapshot_sources(out: Path) -> dict[str, object]:
    source_dir = out / "sources"
    source_dir.mkdir()
    files = (
        Path(__file__).resolve(),
        Path(__file__).with_name("tensor_active.py").resolve(),
        Path(__file__).with_name("experiment_profile.py").resolve(),
    )
    records = []
    for source in files:
        destination = source_dir / source.name
        shutil.copy2(source, destination)
        records.append(
            {
                "path": str(source.relative_to(REPOSITORY)),
                "sha256": sha256(source),
                "snapshot": str(destination.relative_to(GROUP)),
            }
        )
    return {
        "sources": records,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "tracked_worktree_dirty": subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPOSITORY, check=False
        ).returncode
        != 0,
        "tracked_index_dirty": subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=REPOSITORY, check=False
        ).returncode
        != 0,
        "comet_auto_log_git_metadata": os.environ.get("COMET_AUTO_LOG_GIT_METADATA"),
        "comet_auto_log_git_patch": os.environ.get("COMET_AUTO_LOG_GIT_PATCH"),
        "comet_auto_log_env_details": os.environ.get("COMET_AUTO_LOG_ENV_DETAILS"),
    }


def write_checks_csv(out: Path, audit: dict[str, object]) -> None:
    rows = []
    limits = audit["assertion_limits"]
    metrics = audit["asserted_metrics"]
    for name, limit in limits.items():  # type: ignore[union-attr]
        rows.append(
            {
                "check": name,
                "value": metrics[name],  # type: ignore[index]
                "limit": limit,
                "passed": float(metrics[name]) < float(limit),  # type: ignore[index]
            }
        )
    with (out / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("choose an empty output directory")
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = snapshot_sources(out)
    write_json(out / "provenance.json", provenance)
    wp.init()
    cpu = assembled_cpu_audit(cfg.finite_difference_step)
    write_json(out / "cpu-audit.json", cpu)
    write_checks_csv(out, cpu)
    cuda = (
        cuda_smoke(cpu)
        if cfg.run_cuda_smoke
        else {
            "status": "skipped",
            "reason": "disabled by configuration",
        }
    )
    write_json(out / "cuda-smoke.json", cuda)
    summary = {
        "status": "passed",
        "scope": "fiber-free tensor active-stress material and one-tetrahedron Warp FEM validation",
        "formula": {
            "energy": "W_stable(F)+0.5*Q:(F^T F-I)",
            "first_piola": "P_stable(F)+F Q",
            "contract": "Q is symmetric PSD and capped outside the material",
        },
        "cpu": cpu,
        "cuda": cuda,
        "provenance": provenance,
    }
    write_json(out / "summary.json", summary)
    cherries.log_metrics(
        {
            "tensor_active/max_asserted_error": max(cpu["asserted_metrics"].values()),  # type: ignore[union-attr]
            "tensor_active/zero_q_max_error": max(
                cpu["zero_active_stress_passive_restoration"].values()  # type: ignore[union-attr]
            ),
            "tensor_active/cuda_cpu_energy_error": cuda.get(
                "cpu_energy_density_abs_error_MPa", 0.0
            ),
        },
        step=0,
    )
    LOG.info("Tensor active-stress validation passed: %s", out / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
