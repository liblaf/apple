"""Validate the experiment-local signed-stress bulk and skin materials."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from joint_common import ProfileJoint, archive_sources, write_json
from joint_materials import (
    ACTIVE_STRESS,
    BASELINE_STRESS,
    THICKNESS,
    StableNeoHookeanMembrane,
    StableNeoHookeanStress,
    rest_tangent_frames,
)

from liblaf import cherries
from liblaf.apple.common import FRACTION, GLOBAL_POINT_ID, LAMBDA, MU
from liblaf.apple.warp.model import WarpModel


class Config(cherries.BaseConfig):
    """Configuration for the one-cell constitutive validation."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("material-validation", mkdir=True)
    finite_difference_step: float = 1.0e-5
    run_cuda_smoke: bool = False


def from_torch_float(value: torch.Tensor) -> wp.array:
    return wp.from_torch(value, dtype=wp.dtype_from_torch(value.dtype))


def from_torch_vec3(value: torch.Tensor) -> wp.array:
    dtype = wp.dtype_from_torch(value.dtype)
    return wp.from_torch(value, dtype=wp.types.vector(3, dtype))


def potential_fun(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, dtype=u.dtype, device=u.device)
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
        from_torch_vec3(u), from_torch_vec3(direction), from_torch_vec3(output)
    )
    wp.synchronize()
    return output


def potential_hess_quad(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros(1, dtype=u.dtype, device=u.device)
    potential.hess_quad(
        from_torch_vec3(u), from_torch_vec3(direction), from_torch_float(output)
    )
    wp.synchronize()
    return output[0]


def rotation(axis: torch.Tensor, angle: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis
    zero = axis.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return identity + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def make_bulk_mesh(
    points: np.ndarray, lmbda: float, mu: float, stress: np.ndarray
) -> pv.UnstructuredGrid:
    mesh = pv.UnstructuredGrid(
        np.asarray((4, 0, 1, 2, 3)),
        np.asarray((pv.CellType.TETRA,), dtype=np.uint8),
        points,
    )
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.asarray((lmbda,))
    mesh.cell_data[MU.vtk] = np.asarray((mu,))
    mesh.cell_data[FRACTION.vtk] = np.ones(1)
    mesh.cell_data["ActiveStress"] = stress[None]
    return mesh


def make_skin_mesh(
    points: np.ndarray, lmbda: float, mu: float, stress: np.ndarray
) -> pv.PolyData:
    mesh = pv.PolyData(points, np.asarray((3, 0, 1, 2)))
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(3, dtype=np.int32)
    mesh.cell_data[LAMBDA.vtk] = np.asarray((lmbda,))
    mesh.cell_data[MU.vtk] = np.asarray((mu,))
    mesh.cell_data[FRACTION.vtk] = np.ones(1)
    mesh.cell_data["BaselineStress"] = stress[None]
    return mesh


def bulk_reference(
    u: torch.Tensor,
    points: torch.Tensor,
    stress: torch.Tensor,
    lmbda: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    rest_edges = torch.stack(tuple(points[i] - points[0] for i in range(1, 4)), 1)
    current = points + u
    current_edges = torch.stack(tuple(current[i] - current[0] for i in range(1, 4)), 1)
    F = current_edges @ torch.linalg.inv(rest_edges)
    J = torch.linalg.det(F)
    identity = torch.eye(3, dtype=u.dtype, device=u.device)
    density = (
        0.5 * mu * ((F * F).sum() - 3.0)
        - mu * (J - 1.0)
        + 0.5 * lmbda * (J - 1.0).square()
        + 0.5 * (stress * (F.T @ F - identity)).sum()
    )
    return torch.abs(torch.linalg.det(rest_edges)) * density / 6.0


def skin_reference(
    u: torch.Tensor,
    points: torch.Tensor,
    frame: torch.Tensor,
    stress: torch.Tensor,
    lmbda: torch.Tensor,
    mu: torch.Tensor,
    thickness: torch.Tensor,
) -> torch.Tensor:
    rest_edges = torch.stack((points[1] - points[0], points[2] - points[0]), 1)
    coordinates = frame.T @ rest_edges
    current = points + u
    current_edges = torch.stack((current[1] - current[0], current[2] - current[0]), 1)
    surface_F = current_edges @ torch.linalg.inv(coordinates)
    C = surface_F.T @ surface_F
    area_ratio = torch.sqrt(torch.linalg.det(C))
    z = area_ratio * (lmbda + mu) / (lmbda * area_ratio.square() + mu)
    J = area_ratio * z
    identity = torch.eye(2, dtype=u.dtype, device=u.device)
    passive = thickness * (
        0.5 * mu * (torch.trace(C) + z.square() - 3.0)
        - mu * (J - 1.0)
        + 0.5 * lmbda * (J - 1.0).square()
    )
    baseline = 0.5 * (stress * (C - identity)).sum()
    rest_area = torch.linalg.vector_norm(torch.linalg.cross(*rest_edges.T, dim=0)) / 2.0
    return rest_area * (passive + baseline)


def torch_derivatives(
    reference: Any, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = reference(u)
    gradient = torch.autograd.grad(value, u, create_graph=True)[0]
    hessian = torch.autograd.functional.hessian(
        lambda flat: reference(flat.reshape_as(u)), u.reshape(-1)
    )
    return value.detach(), gradient.detach(), hessian.detach()


def max_abs(value: torch.Tensor) -> float:
    return float(torch.max(torch.abs(value)).detach().cpu())


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = torch.maximum(
        torch.linalg.vector_norm(expected), expected.new_tensor(1.0e-14)
    )
    return float(torch.linalg.vector_norm(actual - expected) / scale)


def check(checks: list[dict[str, Any]], name: str, value: float, limit: float) -> None:
    passed = math.isfinite(value) and value <= limit
    checks.append({"name": name, "value": value, "limit": limit, "passed": passed})
    if not passed:
        message = f"{name}: {value:.6e} exceeds {limit:.6e}"
        raise AssertionError(message)


def state_checks(
    name: str,
    potential: Any,
    u: torch.Tensor,
    direction: torch.Tensor,
    reference: Any,
    checks: list[dict[str, Any]],
    step: float,
) -> dict[str, float]:
    u_leaf = u.detach().clone().requires_grad_()
    expected_energy, expected_grad, expected_hessian = torch_derivatives(
        reference, u_leaf
    )
    expected_hessian = expected_hessian.reshape(u.numel(), u.numel())
    expected_prod = (expected_hessian @ direction.reshape(-1)).reshape_as(u)
    expected_diag = torch.diagonal(expected_hessian).reshape_as(u)
    # These fixtures each contain one cell. The PNCG scalar is clamped,
    # while all physical derivatives above retain their signed curvature.
    expected_quad = torch.sum(expected_prod * direction).clamp_min(0)

    energy = potential_fun(potential, u)
    gradient = potential_grad(potential, u)
    product = potential_hess_prod(potential, u, direction)
    diagonal = potential_hess_diag(potential, u)
    quadratic = potential_hess_quad(potential, u, direction)

    metrics = {
        "energy_abs": float(torch.abs(energy - expected_energy)),
        "gradient_max_abs": max_abs(gradient - expected_grad),
        "product_max_abs": max_abs(product - expected_prod),
        "diagonal_max_abs": max_abs(diagonal - expected_diag),
        "quadratic_abs": float(torch.abs(quadratic - expected_quad)),
        "quadratic_product_abs": float(
            torch.abs(quadratic - torch.sum(product * direction).clamp_min(0))
        ),
    }
    scale = max(1.0, float(torch.abs(expected_energy)))
    for metric, value in metrics.items():
        limit = 2.0e-11 * scale if "quadratic" not in metric else 5.0e-11 * scale
        check(checks, f"{name}.{metric}", value, limit)

    fd_directional = (
        potential_fun(potential, u + step * direction)
        - potential_fun(potential, u - step * direction)
    ) / (2.0 * step)
    fd_product = (
        potential_grad(potential, u + step * direction)
        - potential_grad(potential, u - step * direction)
    ) / (2.0 * step)
    fd_metrics = {
        "fd_gradient_direction_relative": relative_error(
            torch.sum(gradient * direction), fd_directional
        ),
        "fd_hessian_product_relative": relative_error(product, fd_product),
    }
    check(
        checks,
        f"{name}.fd_gradient_direction_relative",
        fd_metrics["fd_gradient_direction_relative"],
        2.0e-7,
    )
    check(
        checks,
        f"{name}.fd_hessian_product_relative",
        fd_metrics["fd_hessian_product_relative"],
        2.0e-7,
    )
    return {**metrics, **fd_metrics}


def objectivity_checks(
    name: str,
    potential: Any,
    points: torch.Tensor,
    u: torch.Tensor,
    direction: torch.Tensor,
    checks: list[dict[str, Any]],
) -> dict[str, float]:
    transform = rotation(points.new_tensor((1.0, -2.0, 0.5)), 0.63)
    translation = points.new_tensor((0.17, -0.09, 0.13))
    current = points + u
    rotated_u = current @ transform.T + translation - points
    rotated_direction = direction @ transform.T
    energy_error = float(
        torch.abs(potential_fun(potential, rotated_u) - potential_fun(potential, u))
    )
    gradient_error = max_abs(
        potential_grad(potential, rotated_u)
        - potential_grad(potential, u) @ transform.T
    )
    product_error = max_abs(
        potential_hess_prod(potential, rotated_u, rotated_direction)
        - potential_hess_prod(potential, u, direction) @ transform.T
    )
    check(checks, f"{name}.objectivity_energy_abs", energy_error, 2.0e-13)
    check(checks, f"{name}.objectivity_gradient_max_abs", gradient_error, 2.0e-12)
    check(checks, f"{name}.objectivity_product_max_abs", product_error, 2.0e-12)
    return {
        "energy_abs": energy_error,
        "gradient_max_abs": gradient_error,
        "product_max_abs": product_error,
    }


def mixed_bulk_checks(
    mesh: pv.UnstructuredGrid,
    points: torch.Tensor,
    u: torch.Tensor,
    direction: torch.Tensor,
    stress: torch.Tensor,
    lmbda: float,
    mu: float,
    checks: list[dict[str, Any]],
) -> dict[str, float]:
    potential = StableNeoHookeanStress.from_pyvista(
        mesh, requires_grad=(ACTIVE_STRESS, LAMBDA.value, MU.value)
    )
    WarpModel({"bulk": potential}).mixed_derivative_prod(
        from_torch_vec3(u), from_torch_vec3(direction)
    )
    materials = potential.get_materials()
    warp_q = wp.to_torch(materials[ACTIVE_STRESS].grad)[0]
    warp_lmbda = wp.to_torch(materials[LAMBDA.value].grad)[0]
    warp_mu = wp.to_torch(materials[MU.value].grad)[0]

    q_leaf = stress.detach().clone().requires_grad_()
    lmbda_leaf = torch.tensor(lmbda, requires_grad=True)
    mu_leaf = torch.tensor(mu, requires_grad=True)
    u_leaf = u.detach().clone().requires_grad_()
    energy = bulk_reference(u_leaf, points, q_leaf, lmbda_leaf, mu_leaf)
    gradient = torch.autograd.grad(energy, u_leaf, create_graph=True)[0]
    expected_q, expected_lmbda, expected_mu = torch.autograd.grad(
        torch.sum(gradient * direction), (q_leaf, lmbda_leaf, mu_leaf)
    )
    q_direction = torch.tensor(
        ((0.7, -0.2, 0.1), (-0.2, -0.3, 0.25), (0.1, 0.25, -0.4))
    )
    q_direction /= torch.linalg.vector_norm(q_direction)
    metrics = {
        "stress_direction_abs": float(
            torch.abs(torch.sum((warp_q - expected_q) * q_direction))
        ),
        "lmbda_abs": float(torch.abs(warp_lmbda - expected_lmbda)),
        "mu_abs": float(torch.abs(warp_mu - expected_mu)),
    }
    for metric, value in metrics.items():
        check(checks, f"bulk.mixed_{metric}", value, 2.0e-12)
    return metrics


def mixed_skin_checks(
    mesh: pv.PolyData,
    points: torch.Tensor,
    frame: torch.Tensor,
    u: torch.Tensor,
    direction: torch.Tensor,
    stress: torch.Tensor,
    lmbda: float,
    mu: float,
    thickness: float,
    checks: list[dict[str, Any]],
) -> dict[str, float]:
    potential = StableNeoHookeanMembrane.from_pyvista(
        mesh,
        thickness=thickness,
        requires_grad=(BASELINE_STRESS, LAMBDA.value, MU.value, THICKNESS),
    )
    WarpModel({"skin": potential}).mixed_derivative_prod(
        from_torch_vec3(u), from_torch_vec3(direction)
    )
    materials = potential.get_materials()
    warp_stress = wp.to_torch(materials[BASELINE_STRESS].grad)[0]
    warp_lmbda = wp.to_torch(materials[LAMBDA.value].grad)[0]
    warp_mu = wp.to_torch(materials[MU.value].grad)[0]
    warp_thickness = wp.to_torch(materials[THICKNESS].grad)[0]

    stress_leaf = stress.detach().clone().requires_grad_()
    lmbda_leaf = torch.tensor(lmbda, requires_grad=True)
    mu_leaf = torch.tensor(mu, requires_grad=True)
    thickness_leaf = torch.tensor(thickness, requires_grad=True)
    u_leaf = u.detach().clone().requires_grad_()
    energy = skin_reference(
        u_leaf,
        points,
        frame,
        stress_leaf,
        lmbda_leaf,
        mu_leaf,
        thickness_leaf,
    )
    gradient = torch.autograd.grad(energy, u_leaf, create_graph=True)[0]
    expected_stress, expected_lmbda, expected_mu, expected_thickness = (
        torch.autograd.grad(
            torch.sum(gradient * direction),
            (stress_leaf, lmbda_leaf, mu_leaf, thickness_leaf),
        )
    )
    stress_direction = torch.tensor(((0.8, -0.3), (-0.3, -0.2)))
    stress_direction /= torch.linalg.vector_norm(stress_direction)
    metrics = {
        "stress_direction_abs": float(
            torch.abs(torch.sum((warp_stress - expected_stress) * stress_direction))
        ),
        "lmbda_abs": float(torch.abs(warp_lmbda - expected_lmbda)),
        "mu_abs": float(torch.abs(warp_mu - expected_mu)),
        "thickness_abs": float(torch.abs(warp_thickness - expected_thickness)),
    }
    for metric, value in metrics.items():
        check(checks, f"skin.mixed_{metric}", value, 2.0e-12)
    return metrics


def audit(cfg: Config) -> dict[str, Any]:
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    wp.set_device("cpu")
    checks: list[dict[str, Any]] = []

    young, poisson = 0.2, 0.45
    mu = young / (2.0 * (1.0 + poisson))
    lmbda_classical = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lmbda = lmbda_classical + mu
    bulk_points_np = np.asarray(
        ((0.1, -0.2, 0.3), (1.2, 0.1, 0.4), (0.3, 0.9, 0.8), (0.2, 0.1, 1.4))
    )
    bulk_stress_np = np.asarray(
        ((0.012, 0.003, -0.001), (0.003, -0.006, 0.002), (-0.001, 0.002, 0.004))
    )
    bulk_mesh = make_bulk_mesh(bulk_points_np, lmbda, mu, bulk_stress_np)
    bulk = StableNeoHookeanStress.from_pyvista(bulk_mesh, name="bulk")
    bulk_points = torch.as_tensor(bulk_points_np)
    bulk_stress = torch.as_tensor(bulk_stress_np)
    bulk_u = torch.tensor(
        (
            (0.02, -0.01, 0.03),
            (-0.01, 0.04, -0.02),
            (0.03, 0.01, 0.02),
            (-0.02, 0.02, 0.01),
        )
    )
    bulk_direction = torch.tensor(
        (
            (0.01, -0.03, 0.02),
            (0.02, 0.01, -0.01),
            (-0.02, 0.04, 0.01),
            (0.03, -0.01, -0.02),
        )
    )

    def bulk_reference_fn(value: torch.Tensor) -> torch.Tensor:
        return bulk_reference(
            value,
            bulk_points,
            bulk_stress,
            value.new_tensor(lmbda),
            value.new_tensor(mu),
        )

    bulk_state = state_checks(
        "bulk",
        bulk,
        bulk_u,
        bulk_direction,
        bulk_reference_fn,
        checks,
        cfg.finite_difference_step,
    )
    bulk_objectivity = objectivity_checks(
        "bulk", bulk, bulk_points, bulk_u, bulk_direction, checks
    )
    bulk_mixed = mixed_bulk_checks(
        bulk_mesh,
        bulk_points,
        bulk_u,
        bulk_direction,
        bulk_stress,
        lmbda,
        mu,
        checks,
    )

    skin_points_np = bulk_points_np[:3]
    thickness = 0.0012
    skin_stress_np = np.asarray(((4.0e-5, 1.2e-5), (1.2e-5, -2.0e-5)))
    skin_mesh = make_skin_mesh(skin_points_np, lmbda, mu, skin_stress_np)
    frame = rest_tangent_frames(skin_mesh)[0]
    skin = StableNeoHookeanMembrane.from_pyvista(
        skin_mesh, name="skin", thickness=thickness
    )
    skin_points = torch.as_tensor(skin_points_np)
    skin_stress = torch.as_tensor(skin_stress_np)
    skin_u = bulk_u[:3].clone()
    skin_direction = bulk_direction[:3].clone()

    def skin_reference_fn(value: torch.Tensor) -> torch.Tensor:
        return skin_reference(
            value,
            skin_points,
            frame,
            skin_stress,
            value.new_tensor(lmbda),
            value.new_tensor(mu),
            value.new_tensor(thickness),
        )

    skin_state = state_checks(
        "skin",
        skin,
        skin_u,
        skin_direction,
        skin_reference_fn,
        checks,
        cfg.finite_difference_step,
    )
    skin_objectivity = objectivity_checks(
        "skin", skin, skin_points, skin_u, skin_direction, checks
    )
    skin_mixed = mixed_skin_checks(
        skin_mesh,
        skin_points,
        frame,
        skin_u,
        skin_direction,
        skin_stress,
        lmbda,
        mu,
        thickness,
        checks,
    )

    undeformed = torch.zeros_like(skin_u)
    current_energy = skin_reference_fn(undeformed)
    zero_stress_energy = skin_reference(
        undeformed,
        skin_points,
        frame,
        torch.zeros_like(skin_stress),
        torch.tensor(lmbda),
        torch.tensor(mu),
        torch.tensor(thickness),
    )
    check(
        checks,
        "skin.rest_energy_with_baseline_abs",
        float(torch.abs(current_energy)),
        2.0e-15,
    )
    check(
        checks,
        "skin.rest_passive_energy_abs",
        float(torch.abs(zero_stress_energy)),
        2.0e-15,
    )
    area_ratio = torch.tensor(1.07)
    z = area_ratio * (lmbda + mu) / (lmbda * area_ratio.square() + mu)
    p33 = mu * z + (-mu + lmbda * (area_ratio * z - 1.0)) * area_ratio
    check(checks, "skin.plane_stress_p33_abs", float(torch.abs(p33)), 2.0e-15)

    return {
        "success": True,
        "status": "passed",
        "scope": "one signed-stress tetrahedron and one signed-resultant oblique triangle",
        "material": {
            "young_MPa": young,
            "poisson_ratio": poisson,
            "mu_MPa": mu,
            "lambda_classical_MPa": lmbda_classical,
            "lambda_code_MPa": lmbda,
            "bulk_stress_eigenvalues_MPa": torch.linalg.eigvalsh(bulk_stress).tolist(),
            "skin_resultant_eigenvalues_MPa_m": torch.linalg.eigvalsh(
                skin_stress
            ).tolist(),
            "skin_resultant_N_per_m": (1.0e6 * skin_stress).tolist(),
            "thickness_m": thickness,
        },
        "bulk": {
            "state": bulk_state,
            "objectivity": bulk_objectivity,
            "mixed_material": bulk_mixed,
        },
        "skin": {
            "state": skin_state,
            "objectivity": skin_objectivity,
            "mixed_material": skin_mixed,
            "plane_stress_p33_abs_MPa": float(torch.abs(p33)),
        },
        "checks": checks,
    }


def cuda_smoke(cpu: dict[str, Any]) -> dict[str, Any]:
    if not wp.is_cuda_available():
        return {"status": "skipped", "reason": "Warp CUDA unavailable"}
    torch.set_default_device("cuda")
    wp.set_device("cuda:0")
    material = cpu["material"]
    points = np.asarray(
        ((0.1, -0.2, 0.3), (1.2, 0.1, 0.4), (0.3, 0.9, 0.8), (0.2, 0.1, 1.4))
    )
    stress = np.asarray(
        ((0.012, 0.003, -0.001), (0.003, -0.006, 0.002), (-0.001, 0.002, 0.004))
    )
    potential = StableNeoHookeanStress.from_pyvista(
        make_bulk_mesh(points, material["lambda_code_MPa"], material["mu_MPa"], stress)
    )
    u = torch.tensor(
        (
            (0.02, -0.01, 0.03),
            (-0.01, 0.04, -0.02),
            (0.03, 0.01, 0.02),
            (-0.02, 0.02, 0.01),
        )
    )
    energy = float(potential_fun(potential, u).cpu())
    torch.set_default_device("cpu")
    wp.set_device("cpu")
    return {"status": "passed", "bulk_energy": energy, "device": "cuda:0"}


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if any(cfg.output_dir.iterdir()):
        message = f"choose an empty output directory: {cfg.output_dir}"
        raise ValueError(message)
    archive_sources(cfg.output_dir)
    wp.init()
    result = audit(cfg)
    result["cuda"] = (
        cuda_smoke(result)
        if cfg.run_cuda_smoke
        else {"status": "skipped", "reason": "disabled by configuration"}
    )
    write_json(cfg.output_dir / "summary.json", result)
    with (cfg.output_dir / "checks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("name", "value", "limit", "passed"))
        writer.writeheader()
        writer.writerows(result["checks"])
    cherries.log_metrics(
        {
            "materials/max_check_ratio": max(
                item["value"] / item["limit"] for item in result["checks"]
            ),
            "materials/check_count": len(result["checks"]),
        }
    )
    print({"status": result["status"], "checks": len(result["checks"])})


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
