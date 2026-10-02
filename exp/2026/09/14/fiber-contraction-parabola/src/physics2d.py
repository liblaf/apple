"""Plane-strain reduction of Apple's corrected active Stable Neo-Hookean law.

Only the historical mesh and target helpers are reused. Energy, residual,
Hessian, and implicit control derivatives below all use physical det(F).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[6]
MESH_SOURCE = ROOT / "exp/2026/09/02/pork-shared-release-study/src/10-run-pork-2d.py"
spec = importlib.util.spec_from_file_location("historical_parabola_mesh", MESH_SOURCE)
assert spec is not None
assert spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
build_mesh = base.build_mesh
unpack = base.unpack
loss = base.loss
BASES = base.BASES
cof = base.cof


@dataclass
class State:
    u: np.ndarray
    energy: float
    residual: np.ndarray
    hessian: sp.csc_matrix
    det_f: np.ndarray
    iterations: int


class ForwardSolveError(RuntimeError):
    """A proposed control did not yield an accepted equilibrium."""


def matrices(mesh: Any, controls: np.ndarray, mode: str):
    B = np.broadcast_to(np.eye(2), (len(mesh.tri), 2, 2)).copy()
    if mode == "x_contraction":
        assert np.all(controls >= 0)
        B[mesh.muscle, 0, 0] += controls
    else:
        assert mode in {"contraction_only", "unconstrained"}
        B[mesh.muscle] += np.einsum("ec,cij->eij", controls.reshape(-1, 3), BASES)
        if mode == "contraction_only":
            eigenvalues = np.linalg.eigvalsh(B[mesh.muscle])
            assert np.all(eigenvalues[:, 0] >= 1 - 1e-12), eigenvalues.min()
    return B


def assemble(mesh: Any, u: np.ndarray, B: np.ndarray, *, hessian: bool = True):
    F = np.einsum("eia,eib->eab", (mesh.p + unpack(mesh, u))[mesh.tri], mesh.grad)
    G = F @ B
    J = np.linalg.det(F)
    C = cof(F)
    k = -mesh.mu + mesh.lam * (J - 1)
    density = (
        0.5 * mesh.mu * (np.sum(G * G, (1, 2)) - 2)
        - mesh.mu * (J - 1)
        + 0.5 * mesh.lam * (J - 1) ** 2
    )
    P = mesh.mu[:, None, None] * G @ B.swapaxes(1, 2) + k[:, None, None] * C
    local = mesh.area[:, None, None] * np.einsum("eab,eib->eia", P, mesh.grad)
    lookup = mesh.lookup[mesh.edof]
    mask = lookup >= 0
    residual = np.bincount(
        lookup[mask], weights=local.reshape(-1, 6)[mask], minlength=mesh.nfree
    )
    H = None
    if hessian:
        local_h = np.empty((len(mesh.tri), 6, 6))
        for node in range(3):
            for dim in range(2):
                dF = np.zeros_like(F)
                dF[:, dim, :] = mesh.grad[:, node, :]
                dJ = np.sum(C * dF, (1, 2))
                dP = (
                    mesh.mu[:, None, None] * (dF @ B) @ B.swapaxes(1, 2)
                    + mesh.lam[:, None, None] * dJ[:, None, None] * C
                    + k[:, None, None] * cof(dF)
                )
                local_h[:, :, 2 * node + dim] = (
                    mesh.area[:, None, None] * np.einsum("eab,eib->eia", dP, mesh.grad)
                ).reshape(-1, 6)
        H = sp.coo_matrix(
            (local_h[mesh.ee, mesh.lr, mesh.lc], (mesh.rows, mesh.cols)),
            shape=(mesh.nfree, mesh.nfree),
        ).tocsc()
    return float(mesh.area @ density), residual, H, J


def solve(
    mesh: Any,
    B: np.ndarray,
    seed: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 250,
):
    """Damped Newton; positive-J Armijo line search leaves the energy unchanged."""
    u = seed.copy()
    for iteration in range(max_iterations + 1):
        energy, residual, H, J = assemble(mesh, u, B)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(residual))
        assert J.min() > 0, "Accepted state inverted"
        if np.linalg.norm(residual, np.inf) <= tolerance:
            return State(u, energy, residual, H, J, iteration)
        if iteration == max_iterations:
            break
        scale = max(np.max(np.abs(H.diagonal())), 1e-12)
        direction = None
        for damping in (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0):
            d = spla.spsolve(
                H + damping * scale * sp.eye(mesh.nfree, format="csc"), -residual
            )
            if np.all(np.isfinite(d)) and residual @ d < 0:
                direction = d
                break
        if direction is None:
            message = "No descent direction in forward Newton solve"
            raise ForwardSolveError(message)
        slope = residual @ direction
        for backtrack in range(40):
            alpha = 0.5**backtrack
            trial = u + alpha * direction
            trial_energy, _, _, trial_J = assemble(mesh, trial, B, hessian=False)
            rounding = 2e-15 * max(abs(energy), 1e-8)
            if (
                trial_J.min() > 1e-8
                and trial_energy <= energy + 1e-4 * alpha * slope + rounding
            ):
                u = trial
                break
        else:
            message = (
                f"Forward line search failed, residual={np.max(np.abs(residual)):.3e}"
            )
            raise ForwardSolveError(message)
    message = f"Forward iteration limit, residual={np.max(np.abs(residual)):.3e}"
    raise ForwardSolveError(message)


def control_gradient(
    mesh: Any, state: State, B: np.ndarray, displacement_gradient: np.ndarray, mode: str
):
    adjoint = spla.spsolve(state.hessian.T, displacement_gradient)
    relative_residual = np.linalg.norm(
        state.hessian.T @ adjoint - displacement_gradient
    ) / max(np.linalg.norm(displacement_gradient), 1e-30)
    assert relative_residual < 1e-7, relative_residual
    F = np.einsum("eia,eib->eab", (mesh.p + unpack(mesh, state.u))[mesh.tri], mesh.grad)
    L = np.einsum("eia,eib->eab", unpack(mesh, adjoint)[mesh.tri], mesh.grad)
    gradient_B = (
        -mesh.area[:, None, None]
        * mesh.mu[:, None, None]
        * (F.swapaxes(1, 2) @ L + L.swapaxes(1, 2) @ F)
        @ B
    )
    if mode == "x_contraction":
        gradient = gradient_B[mesh.muscle, 0, 0]
    else:
        gradient = np.einsum("eij,cij->ec", gradient_B[mesh.muscle], BASES).ravel()
    return gradient, float(relative_residual)


def diagnostics(
    mesh: Any, state: State, controls: np.ndarray, mode: str, height: float
):
    B = matrices(mesh, controls, mode)
    U = unpack(mesh, state.u)
    _, _, target = loss(mesh, state.u, height, "l2")
    error = U[mesh.top] - target
    F = np.einsum("eia,eib->eab", (mesh.p + U)[mesh.tri], mesh.grad)
    Z = B @ B.swapaxes(1, 2) - np.eye(2)
    eig = np.linalg.eigvalsh(Z[mesh.muscle])
    muscle_area = mesh.area[mesh.muscle]
    top_all = np.flatnonzero(np.isclose(mesh.p[:, 1], 0.1))
    surface = mesh.p[top_all] + U[top_all]
    # The exact piecewise-linear top curve includes both fixed corner nodes.
    domain_area = np.sum(mesh.area * state.det_f)
    return {
        "fit_rms": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "top_y_error_rms": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "top_x_error_rms": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "motion_rms": float(np.sqrt(np.mean(np.sum(U[mesh.top] ** 2, axis=1)))),
        "peak_top_uy": float(U[mesh.top, 1].max()),
        "minimum_top_uy": float(U[mesh.top, 1].min()),
        "mean_top_uy": float(U[mesh.top, 1].mean()),
        "physical_area_ratio": float(domain_area / mesh.area.sum()),
        "rms_J_minus_1": float(
            np.sqrt(np.average((state.det_f - 1) ** 2, weights=mesh.area))
        ),
        "muscle_rms_J_minus_1": float(
            np.sqrt(
                np.average((state.det_f[mesh.muscle] - 1) ** 2, weights=muscle_area)
            )
        ),
        "min_J": float(state.det_f.min()),
        "max_J": float(state.det_f.max()),
        "inverted_cells": int(np.sum(state.det_f <= 0)),
        "min_singular_B": float(np.linalg.svd(B, compute_uv=False).min()),
        "max_singular_B": float(np.linalg.svd(B, compute_uv=False).max()),
        "negative_effective_mode_cell_fraction": float(np.mean(eig[:, 0] < -1e-10)),
        "max_principal_stretch_F": float(np.linalg.svd(F, compute_uv=False).max()),
        "control_rms": float(np.sqrt(np.mean(controls**2))),
        "force_residual_inf": float(np.linalg.norm(state.residual, np.inf)),
        "top_edge_backtracking_count": int(np.sum(np.diff(surface[:, 0]) <= 0)),
    }
