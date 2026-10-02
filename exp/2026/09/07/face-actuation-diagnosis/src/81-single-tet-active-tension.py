"""Validate a bounded explicit active-tension law on one tetrahedron."""

# ruff: noqa: C901, EM101, EM102, PLR0915, RUF046, TRY003

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from scipy.optimize import least_squares

from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = ROOT.parents[4]
PASSIVE_SOURCE = REPOSITORY / "src/liblaf/apple/warp/fem/_stable_neo_hookean.py"
ACTIVE_STRAIN_SOURCE = (
    REPOSITORY / "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py"
)
REFERENCE_POINTS = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    dtype=np.float64,
)
REFERENCE_VOLUME = 1.0 / 6.0
FIBER = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
FIBER_DYAD = np.outer(FIBER, FIBER)


class Config(cherries.BaseConfig):
    """Configuration for the CPU-only constitutive and single-tet checks."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output(
        "81-single-tet-active-tension-extended", mkdir=True
    )
    young_mpa: float = 0.024
    poisson_ratio: float = 0.46
    external_nominal_stress_mpa: float = 0.005
    gains: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0)
    finite_difference_step: float = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_parameters(young_mpa: float, nu: float) -> tuple[float, float]:
    """Return production ``(mu, lambda_code)`` values in MPa."""
    if not young_mpa > 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("Young's modulus must be positive and -1 < nu < 0.5")
    mu = young_mpa / (2.0 * (1.0 + nu))
    lambda_classical = young_mpa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lambda_classical + mu


def cofactor(matrix: np.ndarray) -> np.ndarray:
    return float(np.linalg.det(matrix)) * np.linalg.inv(matrix).T


def passive_energy_mpa(F: np.ndarray, mu: float, lambda_code: float) -> float:
    J = float(np.linalg.det(F))
    return float(
        0.5 * mu * (np.sum(F * F) - 3.0)
        - mu * (J - 1.0)
        + 0.5 * lambda_code * (J - 1.0) ** 2
    )


def passive_first_piola_mpa(F: np.ndarray, mu: float, lambda_code: float) -> np.ndarray:
    J = float(np.linalg.det(F))
    return mu * F + (-mu + lambda_code * (J - 1.0)) * cofactor(F)


def active_energy_mpa(F: np.ndarray, tension_mpa: float) -> float:
    """Energy of a constant reference-frame second-Piola fiber tension."""
    fiber_image = F @ FIBER
    return float(0.5 * tension_mpa * (fiber_image @ fiber_image - 1.0))


def active_first_piola_mpa(F: np.ndarray, tension_mpa: float) -> np.ndarray:
    return tension_mpa * F @ FIBER_DYAD


def total_energy_mpa(
    F: np.ndarray, mu: float, lambda_code: float, tension_mpa: float
) -> float:
    return passive_energy_mpa(F, mu, lambda_code) + active_energy_mpa(F, tension_mpa)


def total_first_piola_mpa(
    F: np.ndarray, mu: float, lambda_code: float, tension_mpa: float
) -> np.ndarray:
    return passive_first_piola_mpa(F, mu, lambda_code) + active_first_piola_mpa(
        F, tension_mpa
    )


def tangent_action_mpa(
    F: np.ndarray,
    direction: np.ndarray,
    mu: float,
    lambda_code: float,
    tension_mpa: float,
) -> np.ndarray:
    """Apply the exact material tangent ``dP(F)[direction]``."""
    J = float(np.linalg.det(F))
    inverse_transpose = np.linalg.inv(F).T
    cof = J * inverse_transpose
    dJ = float(np.sum(cof * direction))
    dcof = (
        dJ * inverse_transpose - J * inverse_transpose @ direction.T @ inverse_transpose
    )
    determinant_coefficient = -mu + lambda_code * (J - 1.0)
    passive = mu * direction + lambda_code * dJ * cof + determinant_coefficient * dcof
    active = tension_mpa * direction @ FIBER_DYAD
    return passive + active


def finite_difference_first_piola(
    F: np.ndarray,
    mu: float,
    lambda_code: float,
    tension_mpa: float,
    step: float,
) -> np.ndarray:
    result = np.empty((3, 3), dtype=np.float64)
    for index in range(9):
        direction = np.zeros((3, 3), dtype=np.float64)
        direction.flat[index] = 1.0
        result.flat[index] = (
            total_energy_mpa(F + step * direction, mu, lambda_code, tension_mpa)
            - total_energy_mpa(F - step * direction, mu, lambda_code, tension_mpa)
        ) / (2.0 * step)
    return result


def dense_tangent(
    F: np.ndarray, mu: float, lambda_code: float, tension_mpa: float
) -> np.ndarray:
    result = np.empty((9, 9), dtype=np.float64)
    for column in range(9):
        direction = np.zeros((3, 3), dtype=np.float64)
        direction.flat[column] = 1.0
        result[:, column] = tangent_action_mpa(
            F, direction, mu, lambda_code, tension_mpa
        ).ravel()
    return result


def derivative_audit(
    mu: float, lambda_code: float, reference_tension_mpa: float, step: float
) -> dict[str, Any]:
    F = np.asarray(
        ((1.08, 0.04, -0.02), (0.01, 0.93, 0.03), (-0.02, 0.02, 1.04)),
        dtype=np.float64,
    )
    direction = np.asarray(
        ((0.03, -0.04, 0.02), (0.01, 0.05, -0.03), (-0.02, 0.01, 0.04)),
        dtype=np.float64,
    )
    tension = 1.3 * reference_tension_mpa
    analytic_P = total_first_piola_mpa(F, mu, lambda_code, tension)
    finite_P = finite_difference_first_piola(F, mu, lambda_code, tension, step)
    analytic_action = tangent_action_mpa(F, direction, mu, lambda_code, tension)
    finite_action = (
        total_first_piola_mpa(F + step * direction, mu, lambda_code, tension)
        - total_first_piola_mpa(F - step * direction, mu, lambda_code, tension)
    ) / (2.0 * step)
    tangent = dense_tangent(F, mu, lambda_code, tension)
    finite_diagonal = np.empty(9, dtype=np.float64)
    for index in range(9):
        basis = np.zeros((3, 3), dtype=np.float64)
        basis.flat[index] = 1.0
        finite_diagonal[index] = (
            total_first_piola_mpa(F + step * basis, mu, lambda_code, tension).flat[
                index
            ]
            - total_first_piola_mpa(F - step * basis, mu, lambda_code, tension).flat[
                index
            ]
        ) / (2.0 * step)
    energy_step = math.sqrt(step)
    energy_quad = (
        total_energy_mpa(F + energy_step * direction, mu, lambda_code, tension)
        - 2.0 * total_energy_mpa(F, mu, lambda_code, tension)
        + total_energy_mpa(F - energy_step * direction, mu, lambda_code, tension)
    ) / energy_step**2
    analytic_quad = float(direction.ravel() @ tangent @ direction.ravel())
    active_tangent = np.kron(np.eye(3), tension * FIBER_DYAD)
    active_tangent_eigenvalues = np.linalg.eigvalsh(active_tangent)
    angle = 0.73
    axis = np.asarray((0.3, -0.4, 0.5), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    rotation = (
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross
    )
    rotated_F = rotation @ F
    rotated_direction = rotation @ direction
    objectivity_energy_error = abs(
        total_energy_mpa(rotated_F, mu, lambda_code, tension)
        - total_energy_mpa(F, mu, lambda_code, tension)
    )
    objectivity_stress_error = float(
        np.max(
            np.abs(
                total_first_piola_mpa(rotated_F, mu, lambda_code, tension)
                - rotation @ analytic_P
            )
        )
    )
    objectivity_tangent_error = float(
        np.max(
            np.abs(
                tangent_action_mpa(
                    rotated_F,
                    rotated_direction,
                    mu,
                    lambda_code,
                    tension,
                )
                - rotation @ analytic_action
            )
        )
    )
    result = {
        "generic_F": F.tolist(),
        "generic_detF": float(np.linalg.det(F)),
        "gain": 1.3,
        "tension_MPa": tension,
        "finite_difference_step": step,
        "first_piola_max_abs_error_MPa": float(np.max(np.abs(analytic_P - finite_P))),
        "tangent_action_max_abs_error_MPa": float(
            np.max(np.abs(analytic_action - finite_action))
        ),
        "tangent_diagonal_max_abs_error_MPa": float(
            np.max(np.abs(np.diag(tangent) - finite_diagonal))
        ),
        "hessian_symmetry_max_abs_error_MPa": float(
            np.max(np.abs(tangent - tangent.T))
        ),
        "hessian_quadratic_analytic_MPa": analytic_quad,
        "hessian_quadratic_finite_difference_MPa": float(energy_quad),
        "hessian_quadratic_abs_error_MPa": abs(analytic_quad - energy_quad),
        "active_tangent_min_eigenvalue_MPa": float(active_tangent_eigenvalues.min()),
        "active_tangent_max_eigenvalue_MPa": float(active_tangent_eigenvalues.max()),
        "active_tangent_rank": int(
            np.count_nonzero(active_tangent_eigenvalues > 1.0e-12)
        ),
        "objectivity_rotation": rotation.tolist(),
        "objectivity_energy_abs_error_MPa": objectivity_energy_error,
        "objectivity_first_piola_covariance_max_abs_error_MPa": objectivity_stress_error,
        "objectivity_tangent_covariance_max_abs_error_MPa": objectivity_tangent_error,
    }
    if result["first_piola_max_abs_error_MPa"] >= 1.0e-9:
        raise AssertionError("finite-difference energy gradient disagrees with P")
    if result["tangent_action_max_abs_error_MPa"] >= 1.0e-9:
        raise AssertionError("finite-difference dP disagrees with tangent action")
    if result["tangent_diagonal_max_abs_error_MPa"] >= 1.0e-9:
        raise AssertionError("finite-difference tangent diagonal disagrees")
    if result["hessian_symmetry_max_abs_error_MPa"] >= 1.0e-12:
        raise AssertionError("material tangent is not symmetric")
    if result["hessian_quadratic_abs_error_MPa"] >= 1.0e-8:
        raise AssertionError("finite-difference energy curvature disagrees")
    if result["active_tangent_min_eigenvalue_MPa"] < -1.0e-14:
        raise AssertionError("active tangent must be positive semidefinite")
    if result["active_tangent_rank"] != 3:
        raise AssertionError("active tangent must have rank three for one fiber")
    if (
        max(
            objectivity_energy_error,
            objectivity_stress_error,
            objectivity_tangent_error,
        )
        >= 1.0e-12
    ):
        raise AssertionError("energy, stress, or tangent failed the objectivity check")
    return result


def active_strain_energy_mpa(
    F: np.ndarray, A_inv: np.ndarray, mu: float, lambda_code: float
) -> float:
    return passive_energy_mpa(F @ A_inv, mu, lambda_code)


def boundedness_audit(
    mu: float, lambda_code: float, reference_tension_mpa: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    stretches = np.geomspace(1.0e-3, 1.0e3, 241)
    safe = np.empty_like(stretches)
    unsafe = np.empty_like(stretches)
    A_inv = np.diag((2.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)))
    for index, transverse_stretch in enumerate(stretches):
        F = np.diag((1.0, transverse_stretch, 1.0 / transverse_stretch))
        safe[index] = total_energy_mpa(F, mu, lambda_code, reference_tension_mpa)
        unsafe[index] = passive_energy_mpa(F, mu, lambda_code) + 10.0 * (
            active_strain_energy_mpa(F, A_inv, mu, lambda_code)
            - passive_energy_mpa(F, mu, lambda_code)
        )
    analytic_lower_bound = (
        -1.5 * mu - 0.5 * mu * mu / lambda_code - 0.5 * reference_tension_mpa
    )
    transverse_quadratic_coefficient = 1.0 + 10.0 * (0.5 - 1.0)
    result = {
        "path": "F=diag(1,s,1/s), detF=1, s in [1e-3,1e3]",
        "safe_energy_min_sample_MPa": float(safe.min()),
        "safe_energy_at_s_1e_minus_3_MPa": float(safe[0]),
        "safe_energy_at_s_1e3_MPa": float(safe[-1]),
        "safe_global_lower_bound_MPa": analytic_lower_bound,
        "safe_bound_derivation": "complete the square in J; ||F||^2>=0 and I4>=0",
        "rejected_model": "W_passive(F)+10*(W_active_strain(F,Ainv_c50)-W_passive(F))",
        "rejected_c50_Ainv": A_inv.tolist(),
        "rejected_transverse_quadratic_coefficient": transverse_quadratic_coefficient,
        "rejected_energy_min_sample_MPa": float(unsafe.min()),
        "rejected_energy_at_s_1e_minus_3_MPa": float(unsafe[0]),
        "rejected_energy_at_s_1e3_MPa": float(unsafe[-1]),
        "interpretation": "The additive active-tension term is bounded below and adds a PSD tangent. The scaled active-strain energy difference has negative transverse quadratic coefficients and tends to minus infinity on this isochoric path.",
    }
    if transverse_quadratic_coefficient >= 0.0:
        raise AssertionError(
            "unsafe model counterexample lost its negative coefficient"
        )
    if not unsafe[-1] < unsafe[len(unsafe) // 2] - 1.0e3:
        raise AssertionError("unsafe model did not expose decreasing unbounded branch")
    if not safe[-1] > safe[len(safe) // 2] + 1.0e3:
        raise AssertionError("safe model did not remain coercive on sampled path")
    return result, stretches, safe, unsafe


def restricted_tangent_min_eigenvalue(
    F: np.ndarray, mu: float, lambda_code: float, tension_mpa: float
) -> float:
    directions = (np.diag((1.0, 0.0, 0.0)), np.diag((0.0, 1.0, 1.0)))
    hessian = np.asarray(
        [
            [
                float(
                    np.sum(
                        left
                        * tangent_action_mpa(F, right, mu, lambda_code, tension_mpa)
                    )
                )
                for right in directions
            ]
            for left in directions
        ]
    )
    return float(np.linalg.eigvalsh(hessian).min())


def solve_diagonal_tetrahedron(
    *,
    gain: float,
    reference_tension_mpa: float,
    external_stress_mpa: float,
    mu: float,
    lambda_code: float,
    initial_log_stretches: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    if gain < 0.0:
        raise ValueError("active-tension gain must be nonnegative")
    tension = gain * reference_tension_mpa

    def residual(log_stretches: np.ndarray) -> np.ndarray:
        axial, transverse = np.exp(log_stretches)
        F = np.diag((axial, transverse, transverse))
        P = total_first_piola_mpa(F, mu, lambda_code, tension)
        return np.asarray((P[0, 0] - external_stress_mpa, P[1, 1]))

    solution = least_squares(
        residual,
        initial_log_stretches,
        xtol=1.0e-14,
        ftol=1.0e-14,
        gtol=1.0e-14,
        max_nfev=1000,
    )
    stretches = np.exp(solution.x)
    F = np.diag((stretches[0], stretches[1], stretches[1]))
    P_passive = passive_first_piola_mpa(F, mu, lambda_code)
    P_active = active_first_piola_mpa(F, tension)
    P_total = P_passive + P_active
    residual_value = residual(solution.x)
    min_eigenvalue = restricted_tangent_min_eigenvalue(F, mu, lambda_code, tension)
    if not solution.success or np.max(np.abs(residual_value)) >= 1.0e-10:
        raise AssertionError(f"single-tet equilibrium failed for gain {gain}")
    if min_eigenvalue <= 0.0:
        raise AssertionError(f"single-tet equilibrium is not a local minimum: {gain}")
    points = REFERENCE_POINTS @ F.T
    result = {
        "gain": gain,
        "reference_tension_MPa": reference_tension_mpa,
        "applied_tension_MPa": tension,
        "external_nominal_axial_stress_MPa": external_stress_mpa,
        "axial_stretch": float(stretches[0]),
        "transverse_stretch": float(stretches[1]),
        "detF": float(np.linalg.det(F)),
        "deformed_volume": float(REFERENCE_VOLUME * np.linalg.det(F)),
        "energy_density_MPa": total_energy_mpa(F, mu, lambda_code, tension),
        "dead_load_potential_density_MPa": float(
            total_energy_mpa(F, mu, lambda_code, tension)
            - external_stress_mpa * F[0, 0]
        ),
        "passive_first_piola_diagonal_MPa": np.diag(P_passive).tolist(),
        "active_first_piola_diagonal_MPa": np.diag(P_active).tolist(),
        "total_first_piola_diagonal_MPa": np.diag(P_total).tolist(),
        "equilibrium_residual_max_abs_MPa": float(np.max(np.abs(residual_value))),
        "restricted_tangent_min_eigenvalue_MPa": min_eigenvalue,
        "solver": {
            "implementation": "scipy.optimize.least_squares on log axial/transverse stretches",
            "success": bool(solution.success),
            "status": int(solution.status),
            "message": str(solution.message),
            "function_evaluations": int(solution.nfev),
            "cost": float(solution.cost),
        },
        "boundary_family": "orthogonal single tetrahedron; origin fixed, three edge nodes remain on their reference axes, transverse stretches constrained equal",
        "points": points.tolist(),
    }
    return result, solution.x


def save_state(path: Path, row: dict[str, Any]) -> None:
    points = np.asarray(row["points"], dtype=np.float64)
    grid = pv.UnstructuredGrid(
        np.asarray((4, 0, 1, 2, 3), dtype=np.int64),
        np.asarray((pv.CellType.TETRA,), dtype=np.uint8),
        points,
    )
    grid.point_data["ReferencePosition"] = REFERENCE_POINTS
    grid.point_data["Displacement"] = points - REFERENCE_POINTS
    for key in (
        "gain",
        "reference_tension_MPa",
        "applied_tension_MPa",
        "external_nominal_axial_stress_MPa",
        "axial_stretch",
        "transverse_stretch",
        "detF",
        "restricted_tangent_min_eigenvalue_MPa",
    ):
        grid.cell_data[key] = np.asarray((row[key],), dtype=np.float64)
    grid.save(path, binary=True)


def plot_response(
    rows: list[dict[str, Any]],
    stretches: np.ndarray,
    safe_energy: np.ndarray,
    unsafe_energy: np.ndarray,
    output_base: Path,
) -> None:
    loads = sorted({float(row["external_nominal_axial_stress_MPa"]) for row in rows})
    colors = ("#3972a3", "#b25d43")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.18, wspace=0.3)
    for load, color in zip(loads, colors, strict=True):
        subset = [
            row
            for row in rows
            if float(row["external_nominal_axial_stress_MPa"]) == load
        ]
        gains = [float(row["gain"]) for row in subset]
        label = f"external axial stress {1000.0 * load:g} kPa"
        axes[0].plot(
            gains,
            [row["axial_stretch"] for row in subset],
            "o-",
            color=color,
            label=label,
        )
        axes[1].plot(
            gains,
            [row["detF"] for row in subset],
            "o-",
            color=color,
            label=label,
        )
    axes[0].axhline(1.0, color="#777777", lw=0.8)
    axes[0].set(xlabel="active-tension gain", ylabel="axial stretch")
    axes[0].set_title("A  Emergent fiber contraction", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(1.0, color="#777777", lw=0.8)
    axes[1].set(xlabel="active-tension gain", ylabel="det(F)")
    axes[1].set_title("B  Volume response", loc="left", fontweight="bold")
    axes[2].plot(
        stretches, safe_energy, color="#3972a3", label="additive active tension"
    )
    axes[2].plot(
        stretches,
        unsafe_energy,
        color="#b25d43",
        label="rejected energy difference",
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("symlog", linthresh=0.01)
    axes[2].set(xlabel="isochoric transverse stretch s", ylabel="energy density (MPa)")
    axes[2].set_title("C  Boundedness discriminator", loc="left", fontweight="bold")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle(
        "Single-tetrahedron explicit active-tension diagnostic",
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_tetrahedra(rows: list[dict[str, Any]], output_base: Path) -> None:
    selected = [row for row in rows if float(row["gain"]) in (0.0, 1.0, 2.0)]
    fig = plt.figure(figsize=(11.5, 7.2))
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for index, row in enumerate(selected, start=1):
        axis = fig.add_subplot(2, 3, index, projection="3d")
        points = np.asarray(row["points"])
        for begin, end in edges:
            axis.plot(*points[[begin, end]].T, color="#3972a3", lw=2)
        axis.scatter(*points.T, color="#b25d43", s=18)
        extent = 1.55
        axis.set(xlim=(0, extent), ylim=(0, extent), zlim=(0, extent))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=22, azim=-58)
        axis.set_title(
            f"gain {row['gain']:g}; external {1000.0 * row['external_nominal_axial_stress_MPa']:g} kPa\n"
            f"axial {row['axial_stretch']:.3f}; detF {row['detF']:.3f}",
            fontsize=9,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_zticks([])
    fig.suptitle(
        "Actual constrained single-tetrahedron equilibria",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main(cfg: Config) -> None:
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    if tuple(sorted(set(cfg.gains))) != cfg.gains or cfg.gains[0] != 0.0:
        raise ValueError("gains must be unique, ascending, and start at zero")
    if cfg.external_nominal_stress_mpa <= 0.0:
        raise ValueError("external nominal stress must be positive")

    mu, lambda_code = stable_parameters(cfg.young_mpa, cfg.poisson_ratio)
    # The production c50 volume-preserving active-strain map gives P_xx=3*mu at F=I.
    reference_tension_mpa = 3.0 * mu
    identity = np.eye(3)
    passive_rest_energy = passive_energy_mpa(identity, mu, lambda_code)
    passive_rest_stress = passive_first_piola_mpa(identity, mu, lambda_code)
    if passive_rest_energy != 0.0 or np.any(passive_rest_stress != 0.0):
        raise AssertionError("unchanged passive law must be exactly stress-free at F=I")
    if total_energy_mpa(identity, mu, lambda_code, 0.0) != passive_rest_energy:
        raise AssertionError("zero gain must restore passive energy exactly")
    if not np.array_equal(
        total_first_piola_mpa(identity, mu, lambda_code, 0.0),
        passive_rest_stress,
    ):
        raise AssertionError("zero gain must restore passive stress exactly")

    derivatives = derivative_audit(
        mu, lambda_code, reference_tension_mpa, cfg.finite_difference_step
    )
    boundedness, stretch_scan, safe_energy, unsafe_energy = boundedness_audit(
        mu, lambda_code, reference_tension_mpa
    )
    rows: list[dict[str, Any]] = []
    states = output / "states"
    states.mkdir()
    for external_stress in (0.0, cfg.external_nominal_stress_mpa):
        initial = np.zeros(2, dtype=np.float64)
        for gain in cfg.gains:
            row, initial = solve_diagonal_tetrahedron(
                gain=gain,
                reference_tension_mpa=reference_tension_mpa,
                external_stress_mpa=external_stress,
                mu=mu,
                lambda_code=lambda_code,
                initial_log_stretches=initial,
            )
            load_kpa = int(round(1000.0 * external_stress))
            gain_code = int(round(100.0 * gain))
            state_path = states / f"load-{load_kpa:03d}kpa-gain-{gain_code:03d}.vtu"
            save_state(state_path, row)
            row["state"] = record(state_path)
            row.pop("points")
            rows.append(row)

    unloaded = [row for row in rows if row["external_nominal_axial_stress_MPa"] == 0.0]
    loaded = [
        row
        for row in rows
        if row["external_nominal_axial_stress_MPa"] == cfg.external_nominal_stress_mpa
    ]
    if not all(
        later["axial_stretch"] < earlier["axial_stretch"]
        for earlier, later in itertools.pairwise(unloaded)
    ):
        raise AssertionError("unloaded axial stretch must decrease with gain")
    if not all(
        later["axial_stretch"] < earlier["axial_stretch"]
        for earlier, later in itertools.pairwise(loaded)
    ):
        raise AssertionError("loaded axial stretch must decrease with gain")

    response_base = output / "gain-response"
    tetrahedra_base = output / "tetrahedra"
    plot_response(rows, stretch_scan, safe_energy, unsafe_energy, response_base)
    # Reload points from immutable VTU states for the visual, so the figure checks saved geometry.
    visual_rows = []
    for row in rows:
        copy = dict(row)
        copy["points"] = np.asarray(pv.read(row["state"]["path"]).points).tolist()
        visual_rows.append(copy)
    plot_tetrahedra(visual_rows, tetrahedra_base)

    summary_path = output / "summary.json"
    summary = {
        "schema_version": 1,
        "scope": "CPU-only constitutive and constrained single-tetrahedron diagnostic; no face solve",
        "formulation": {
            "energy": "Psi=Psi_stable_NH(F;mu,lambda_code)+(T/2)*(I4-1)",
            "I4": "f^T F^T F f=||Ff||^2, with fixed unit reference fiber f",
            "first_piola": "P=P_passive+T*(Ff) outer f",
            "tangent_action": "dP[dF]=dP_passive[dF]+T*(dF f) outer f",
            "active_second_piola": "S_active=T*f outer f",
            "objectivity": "I4 is invariant under superposed rigid spatial rotation",
            "meaning": "T is explicit reference-frame active tension in MPa; it is not a prescribed natural contraction or a multiplier on passive Lamé parameters",
            "zero_gain": "T=0 restores the unchanged passive stable Neo-Hookean energy, stress, and tangent exactly",
            "boundedness": "the active addition is >= -T/2 and its tangent is PSD; combined with the coercive stable passive polynomial it is bounded below",
        },
        "material": {
            "young_MPa": cfg.young_mpa,
            "poisson_ratio": cfg.poisson_ratio,
            "mu_MPa": mu,
            "lambda_code_MPa": lambda_code,
            "lambda_convention": "lambda_code=lambda_classical+mu, matching the production stable Neo-Hookean implementation",
        },
        "gain_reference": {
            "reference_tension_MPa": reference_tension_mpa,
            "definition": "3*mu, equal to the axial P_xx at F=I for the current c50 Ainv=diag(2,1/sqrt(2),1/sqrt(2)) pure-muscle active-strain constituent",
            "non_equivalence": "matching this one initial axial stress component does not make the active-tension and active-strain laws equivalent",
            "gains": list(cfg.gains),
        },
        "passive_rest_check": {
            "F": identity.tolist(),
            "energy_density_MPa": passive_rest_energy,
            "first_piola_MPa": passive_rest_stress.tolist(),
            "exact_zero_gain_restoration": True,
        },
        "derivative_audit": derivatives,
        "boundedness_audit": boundedness,
        "single_tetrahedron": {
            "reference_points": REFERENCE_POINTS.tolist(),
            "reference_volume": REFERENCE_VOLUME,
            "fiber": FIBER.tolist(),
            "loads_MPa": [0.0, cfg.external_nominal_stress_mpa],
            "cases": rows,
            "interpretation": "Positive tension produces emergent fiber contraction against unchanged passive stiffness. The second load applies an opposing tensile nominal stress along the fiber.",
        },
        "readiness_criterion": {
            "passed": True,
            "statement": "The formulation is ready for one target-independent face diagnostic at fixed passive material, provided gains remain explicit stress parameters and face results retain solver, motion, stress, and geometry diagnostics.",
        },
        "source_references": {
            "diagnostic": record(Path(__file__).resolve()),
            "production_passive_law": record(PASSIVE_SOURCE),
            "current_multiplicative_active_strain_law": record(ACTIVE_STRAIN_SOURCE),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_path = output / "config.json"
    config_path.write_text(cfg.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sources = output / "sources"
    sources.mkdir()
    shutil.copy2(Path(__file__).resolve(), sources / Path(__file__).name)
    manifest = {
        "schema_version": 1,
        "scope": "Immutable records for the single-tetrahedron active-tension diagnostic",
        "summary": record(summary_path),
        "config": record(config_path),
        "figures": {
            path.name: record(path)
            for path in (
                response_base.with_suffix(".png"),
                response_base.with_suffix(".pdf"),
                tetrahedra_base.with_suffix(".png"),
                tetrahedra_base.with_suffix(".pdf"),
            )
        },
        "states": {path.name: record(path) for path in sorted(states.glob("*.vtu"))},
        "sources": {
            "executed_copy": record(sources / Path(__file__).name),
            "production_passive_law": record(PASSIVE_SOURCE),
            "current_multiplicative_active_strain_law": record(ACTIVE_STRAIN_SOURCE),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cherries.log_metrics(
        {
            "active_tension/derivative_P_error_MPa": derivatives[
                "first_piola_max_abs_error_MPa"
            ],
            "active_tension/derivative_tangent_error_MPa": derivatives[
                "tangent_action_max_abs_error_MPa"
            ],
            "active_tension/unloaded_gain1_axial_stretch": next(
                row["axial_stretch"] for row in unloaded if row["gain"] == 1.0
            ),
            "active_tension/unloaded_gain2_axial_stretch": next(
                row["axial_stretch"] for row in unloaded if row["gain"] == 2.0
            ),
            "active_tension/loaded_gain1_axial_stretch": next(
                row["axial_stretch"] for row in loaded if row["gain"] == 1.0
            ),
        }
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "reference_tension_MPa": reference_tension_mpa,
                "first_piola_fd_error_MPa": derivatives[
                    "first_piola_max_abs_error_MPa"
                ],
                "tangent_fd_error_MPa": derivatives["tangent_action_max_abs_error_MPa"],
                "unloaded_gain1_axial_stretch": next(
                    row["axial_stretch"] for row in unloaded if row["gain"] == 1.0
                ),
                "unloaded_gain2_axial_stretch": next(
                    row["axial_stretch"] for row in unloaded if row["gain"] == 2.0
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
