"""Verify the local active-strain skin law against autograd and mapped stress."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import active_strain_materials
import numpy as np
import pyvista as pv
import torch
import warp as wp
from active_strain_materials import StableNeoHookeanActiveMembrane
from joint_materials import StableNeoHookeanMembrane

from liblaf import cherries
from liblaf.apple.common import FRACTION, LAMBDA, MU

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "exp/2026/09/22/solver-performance/src"))
from assembled_fem_hvp import _membrane_kernel  # noqa: E402


class Config(cherries.BaseConfig):
    output: Path = cherries.output("active-strain-check.json", mkdir=True)


def _mesh(B: np.ndarray, baseline: np.ndarray | None) -> pv.PolyData:
    points = np.asarray(((0.1, -0.2, 0.0), (1.2, 0.1, 0.2), (-0.1, 0.8, 0.3)))
    mesh = pv.PolyData(points, np.asarray((3, 0, 1, 2)))
    mesh.point_data["GlobalPointId"] = np.arange(3, dtype=np.int64)
    mesh.cell_data[FRACTION.vtk] = np.ones(1)
    mesh.cell_data[LAMBDA.vtk] = np.asarray((0.82,))
    mesh.cell_data[MU.vtk] = np.asarray((0.37,))
    mesh.cell_data["ActivationInv"] = B[None]
    if baseline is not None:
        mesh.cell_data["BaselineStress"] = baseline[None]
    return mesh


def _warp_u(value: torch.Tensor) -> wp.array:
    return wp.from_torch(
        value.contiguous(), dtype=wp.types.vector(3, wp.float64), requires_grad=False
    )


def _fun(potential: object, u: torch.Tensor) -> torch.Tensor:
    output = wp.zeros(1, dtype=wp.float64, device="cpu")
    potential.fun(_warp_u(u), output)  # type: ignore[attr-defined]
    return wp.to_torch(output).cpu()


def _grad(potential: object, u: torch.Tensor) -> torch.Tensor:
    output = wp.zeros(len(u), dtype=wp.types.vector(3, wp.float64), device="cpu")
    potential.grad(_warp_u(u), output)  # type: ignore[attr-defined]
    return wp.to_torch(output).cpu()


def _hvp(potential: object, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    output = wp.zeros(len(u), dtype=wp.types.vector(3, wp.float64), device="cpu")
    potential.hess_prod(_warp_u(u), _warp_u(p), output)  # type: ignore[attr-defined]
    return wp.to_torch(output).cpu()


def _diag(potential: object, u: torch.Tensor) -> torch.Tensor:
    output = wp.zeros(len(u), dtype=wp.types.vector(3, wp.float64), device="cpu")
    potential.hess_diag(_warp_u(u), output)  # type: ignore[attr-defined]
    return wp.to_torch(output).cpu()


def _quad(potential: object, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    output = wp.zeros(1, dtype=wp.float64, device="cpu")
    potential.hess_quad(_warp_u(u), _warp_u(p), output)  # type: ignore[attr-defined]
    return wp.to_torch(output).cpu()


def _assembled_local_hessian(potential: object, u: torch.Tensor) -> torch.Tensor:
    """Materialize the active-strain sparse assembler's one-triangle block."""
    output = wp.zeros((1, 9, 9), dtype=wp.float64, device="cpu")
    kernel = _membrane_kernel(active_strain_materials, active_strain=True)
    wp.launch(
        kernel,
        dim=(1, 9),
        inputs=[_warp_u(u), potential.cells, potential.materials, 0],  # type: ignore[attr-defined]
        outputs=[output],
        device="cpu",
    )
    return wp.to_torch(output).cpu()[0]


def _torch_energy(
    u: torch.Tensor, B: torch.Tensor, la: float, mu: float, h: float
) -> torch.Tensor:
    x = (
        torch.as_tensor(
            ((0.1, -0.2, 0.0), (1.2, 0.1, 0.2), (-0.1, 0.8, 0.3)), dtype=u.dtype
        )
        + u
    )
    e0, e1 = x[1] - x[0], x[2] - x[0]
    D = torch.stack((e0, e1), dim=1)
    D0 = torch.stack(
        (
            torch.as_tensor((1.1, 0.3, 0.2), dtype=u.dtype),
            torch.as_tensor((-0.2, 1.0, 0.3), dtype=u.dtype),
        ),
        dim=1,
    )
    # c is the Cauchy-Green tensor in an orthonormal neutral tangent frame.
    q, _ = torch.linalg.qr(D0, mode="reduced")
    R = torch.linalg.solve(q.T @ D0, torch.eye(2, dtype=u.dtype))
    F = D @ R
    c = F.T @ F
    area = torch.sqrt(torch.linalg.det(c))
    z = area * (la + mu) / (la * area.square() + mu)
    J = area * z
    return (
        h
        * (
            0.5 * mu * (torch.trace(B.T @ c @ B) + z.square() - 3.0)
            - mu * (J - 1.0)
            + 0.5 * la * (J - 1.0).square()
        )
        * (torch.linalg.norm(torch.cross(D0[:, 0], D0[:, 1], dim=0)) / 2)
    )


def main(cfg: Config) -> None:
    wp.init()
    torch.set_default_dtype(torch.float64)
    h, la, mu = 0.004, 0.82, 0.37
    B = np.asarray(((1.17, 0.08), (0.08, 0.91)))
    T = h * mu * (B @ B.T - np.eye(2))
    u = torch.tensor(((0.02, -0.01, 0.03), (-0.01, 0.02, -0.02), (0.01, 0.015, 0.005)))
    p = torch.tensor(((-0.02, 0.01, 0.005), (0.03, -0.02, 0.01), (0.01, 0.02, -0.015)))
    active = StableNeoHookeanActiveMembrane.from_pyvista(_mesh(B, None), thickness=h)
    stress = StableNeoHookeanMembrane.from_pyvista(_mesh(np.eye(2), T), thickness=h)
    identity = StableNeoHookeanActiveMembrane.from_pyvista(
        _mesh(np.eye(2), None), thickness=h
    )
    passive = StableNeoHookeanMembrane.from_pyvista(
        _mesh(np.eye(2), np.zeros((2, 2))), thickness=h
    )
    t_u = u.detach().clone().requires_grad_()
    ref = _torch_energy(t_u, torch.as_tensor(B), la, mu, h)
    ref_grad = torch.autograd.grad(ref, t_u, create_graph=True)[0]
    ref_hvp = torch.autograd.grad((ref_grad * p).sum(), t_u)[0]
    epsilon = 1.0e-6
    fd_diag = torch.empty_like(u)
    for index in range(u.numel()):
        basis = torch.zeros_like(u)
        basis.flatten()[index] = 1.0
        fd_diag.flatten()[index] = (
            (_grad(active, u + epsilon * basis) - _grad(active, u - epsilon * basis))
            / (2 * epsilon)
        ).flatten()[index]
    hvp = _hvp(active, u, p)
    local = _assembled_local_hessian(active, u)
    basis = torch.eye(9, dtype=u.dtype).reshape(9, 3, 3)
    direct_local = torch.stack(
        [_hvp(active, u, vector).reshape(-1) for vector in basis], dim=1
    )
    metrics = {
        "identity_energy_abs": float(torch.abs(_fun(identity, u) - _fun(passive, u))),
        "identity_gradient_max_abs": float(
            torch.max(torch.abs(_grad(identity, u) - _grad(passive, u)))
        ),
        "mapped_energy_difference_shift_abs": float(
            torch.abs(
                (_fun(active, u) - _fun(stress, u))
                - (
                    _fun(active, torch.zeros_like(u))
                    - _fun(stress, torch.zeros_like(u))
                )
            )
        ),
        "mapped_gradient_max_abs": float(
            torch.max(torch.abs(_grad(active, u) - _grad(stress, u)))
        ),
        "mapped_hvp_max_abs": float(
            torch.max(torch.abs(_hvp(active, u, p) - _hvp(stress, u, p)))
        ),
        "autograd_gradient_max_abs": float(
            torch.max(torch.abs(_grad(active, u) - ref_grad.detach()))
        ),
        "autograd_hvp_max_abs": float(
            torch.max(torch.abs(_hvp(active, u, p) - ref_hvp.detach()))
        ),
        "finite_difference_hvp_max_abs": float(
            torch.max(
                torch.abs(
                    hvp
                    - (_grad(active, u + epsilon * p) - _grad(active, u - epsilon * p))
                    / (2 * epsilon)
                )
            )
        ),
        "finite_difference_diagonal_max_abs": float(
            torch.max(torch.abs(_diag(active, u) - fd_diag))
        ),
        "pncg_quadratic_max_abs": float(
            torch.abs(_quad(active, u, p) - torch.clamp((hvp * p).sum(), min=0.0))
        ),
        "sparse_local_hessian_max_abs": float(
            torch.max(torch.abs(local - direct_local))
        ),
    }
    tolerances = {
        "identity_energy_abs": 1e-12,
        "identity_gradient_max_abs": 1e-11,
        "mapped_energy_difference_shift_abs": 1e-11,
        "mapped_gradient_max_abs": 1e-11,
        "mapped_hvp_max_abs": 1e-10,
        "autograd_gradient_max_abs": 1e-11,
        "autograd_hvp_max_abs": 1e-10,
        "finite_difference_hvp_max_abs": 2e-8,
        "finite_difference_diagonal_max_abs": 2e-8,
        "pncg_quadratic_max_abs": 1e-12,
        "sparse_local_hessian_max_abs": 1e-11,
    }
    checks = {key: value <= tolerances[key] for key, value in metrics.items()}
    result = {
        "success": all(checks.values()),
        "B": B.tolist(),
        "mapped_baseline_stress_MPa_m": T.tolist(),
        "metrics": metrics,
        "checks": checks,
    }
    cfg.output.write_text(json.dumps(result, indent=2) + "\n")
    cherries.log_metrics(metrics)
    if not result["success"]:
        raise AssertionError(result)


if __name__ == "__main__":
    cherries.main(main)
