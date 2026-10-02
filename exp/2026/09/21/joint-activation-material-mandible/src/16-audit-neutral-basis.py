"""Audit exact-reference force representability of the constant bulk basis.

This is a linear force-space diagnostic at ``F = I``.  It deliberately does
not run an equilibrium solve and therefore cannot decide whether the 0.25 mm
neutral deformation budget is feasible.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import scipy.optimize
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import (
    BULK_TISSUES,
    research_informed_material_config,
    symmetric_matrices,
)
from joint_physics import JointPhysics

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False
COORDINATE_NAMES = ("xx", "yy", "zz", "sqrt2_xy", "sqrt2_yz", "sqrt2_xz")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    contact_spec: Path = GROUP / "data/contact/config.json"
    contact_validation: Path = (
        GROUP / "data/contact-initial-validation-002/summary.json"
    )
    output_dir: Path = cherries.output("neutral-basis-audit", mkdir=True)
    skin_target_n_per_m: float = 80.6
    ridge_weights: tuple[float, ...] = (0.0, 1.0e-4, 1.0e-2, 1.0)
    linearity_seed: int = 20260921
    linearity_tolerance: float = 2.0e-10


def matrix_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, yz, xz = coordinates
    root2 = math.sqrt(2.0)
    return np.asarray(
        (
            (xx, xy / root2, xz / root2),
            (xy / root2, yy, yz / root2),
            (xz / root2, yz / root2, zz),
        )
    )


def spectra(coordinates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            np.linalg.eigvalsh(matrix_from_coordinates(block))
            for block in coordinates.reshape(3, 6)
        ]
    )


def project_spectra(coordinates: np.ndarray, lower: float, upper: float) -> np.ndarray:
    blocks = []
    for block in coordinates.reshape(3, 6):
        values, vectors = np.linalg.eigh(matrix_from_coordinates(block))
        bounded = (vectors * np.clip(values, lower, upper)) @ vectors.T
        blocks.append(
            (
                bounded[0, 0],
                bounded[1, 1],
                bounded[2, 2],
                math.sqrt(2.0) * bounded[0, 1],
                math.sqrt(2.0) * bounded[1, 2],
                math.sqrt(2.0) * bounded[0, 2],
            )
        )
    return np.asarray(blocks).reshape(-1)


def solve_scaled_lstsq(matrix: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    column_norms = np.linalg.norm(matrix, axis=0)
    assert np.all(column_norms > 0), column_norms
    scaled = matrix / column_norms
    z, _residuals, rank, singular_values = np.linalg.lstsq(scaled, target, rcond=None)
    return {
        "coefficients": z / column_norms,
        "column_norms": column_norms,
        "rank": int(rank),
        "singular_values_preconditioned": singular_values,
        "condition_number_preconditioned": float(
            singular_values[0] / singular_values[-1]
        ),
    }


def solve_bounded(
    matrix: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    *,
    lower: float,
    upper: float,
    ridge_weight: float,
) -> dict[str, Any]:
    target_norm = np.linalg.norm(target)
    assert target_norm > 0
    normalized_matrix = matrix / target_norm
    normalized_target = target / target_norm
    gram = normalized_matrix.T @ normalized_matrix
    rhs = normalized_matrix.T @ normalized_target

    def objective(value: np.ndarray) -> float:
        residual = normalized_matrix @ value - normalized_target
        return float(
            0.5 * np.dot(residual, residual)
            + 0.5 * ridge_weight * np.dot(value, value) / value.size
        )

    def gradient(value: np.ndarray) -> np.ndarray:
        return gram @ value - rhs + ridge_weight * value / value.size

    def constraints(value: np.ndarray) -> np.ndarray:
        values = spectra(value)
        return np.concatenate(((values - lower).ravel(), (upper - values).ravel()))

    result = scipy.optimize.minimize(
        objective,
        project_spectra(initial, lower, upper),
        jac=gradient,
        constraints={"type": "ineq", "fun": constraints},
        method="SLSQP",
        options={"ftol": 1.0e-13, "maxiter": 2000, "disp": False},
    )
    minimum_constraint = float(constraints(result.x).min())
    assert result.success, result
    assert minimum_constraint >= -1.0e-8, (result, minimum_constraint)
    return {
        "coefficients": result.x,
        "optimizer_status": str(result.message),
        "optimizer_iterations": int(result.nit),
        "objective": float(result.fun),
        "minimum_spectral_slack": minimum_constraint,
        "ridge_weight": ridge_weight,
    }


def force_metrics(
    matrix: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    mu_mpa: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    residual = matrix @ coefficients - target
    weighted_residual = weights * residual
    weighted_target = weights * target
    eigen_dimensionless = spectra(coefficients)
    eigen_mpa = eigen_dimensionless * mu_mpa[:, None]
    raw_relative = float(np.linalg.norm(residual) / np.linalg.norm(target))
    weighted_relative = float(
        np.linalg.norm(weighted_residual) / np.linalg.norm(weighted_target)
    )
    tolerance = 1.0e-7
    nodal_force_n = residual.reshape(-1, 3) * 1.0e6
    return {
        "coefficients_dimensionless": {
            name: coefficients[index * 6 : (index + 1) * 6].tolist()
            for index, name in enumerate(BULK_TISSUES)
        },
        "coordinate_order": list(COORDINATE_NAMES),
        "eigenvalues_dimensionless": {
            name: eigen_dimensionless[index].tolist()
            for index, name in enumerate(BULK_TISSUES)
        },
        "eigenvalues_mpa": {
            name: eigen_mpa[index].tolist() for index, name in enumerate(BULK_TISSUES)
        },
        "lower_bound_occupancy": int(
            np.count_nonzero(eigen_dimensionless <= lower + tolerance)
        ),
        "upper_bound_occupancy": int(
            np.count_nonzero(eigen_dimensionless >= upper - tolerance)
        ),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "mean_squared_coordinate": float(np.mean(coefficients**2)),
        "coefficient_l2_by_tissue": {
            name: float(np.linalg.norm(coefficients[index * 6 : (index + 1) * 6]))
            for index, name in enumerate(BULK_TISSUES)
        },
        "raw_residual_norm_mpa_m2": float(np.linalg.norm(residual)),
        "raw_target_norm_mpa_m2": float(np.linalg.norm(target)),
        "raw_relative_residual": raw_relative,
        "raw_explained_squared_norm_fraction": 1.0 - raw_relative**2,
        "weighted_relative_residual": weighted_relative,
        "weighted_explained_squared_norm_fraction": 1.0 - weighted_relative**2,
        "residual_force_norm_n": float(np.linalg.norm(nodal_force_n)),
        "residual_nodal_force_rms_n": float(
            np.sqrt(np.mean(np.sum(nodal_force_n**2, axis=1)))
        ),
        "residual_nodal_force_max_n": float(
            np.max(np.linalg.norm(nodal_force_n, axis=1))
        ),
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    global COMPLETED  # noqa: PLW0603
    started = time.perf_counter()
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    spec = research_informed_material_config()
    proxy = float(
        spec["materials"]["skin"]["continuation"]["source_proxy_mean_n_per_m"]
    )
    assert math.isclose(cfg.skin_target_n_per_m, proxy, rel_tol=0, abs_tol=1.0e-12)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    contact_spec = json.loads(cfg.contact_spec.read_text())
    contact_validation = json.loads(cfg.contact_validation.read_text())
    assert contact_spec["enabled"] is True
    assert contact_validation["success"] is True
    assert contact_validation["contact_spec_sha256"] == sha256(cfg.contact_spec)

    configure_cuda()
    physics = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={
            name: float(spec["materials"][name]["young_mpa"]) for name in BULK_TISSUES
        },
        bulk_nu={
            name: float(spec["materials"][name]["poisson"]) for name in BULK_TISSUES
        },
        skin_young_mpa=float(spec["materials"]["skin"]["reference_map"]["young_mpa"]),
        skin_nu=float(spec["materials"]["skin"]["poisson"]),
        thickness_m=float(spec["materials"]["skin"]["thickness_m"]),
        contact_config=contact_spec,
    )
    model = physics.runtime.forward.model
    state = physics.runtime.forward.state
    zero_pose = torch.zeros(6)
    zero_skin = torch.zeros((2, 2))
    identity_skin = torch.eye(2) * cfg.skin_target_n_per_m
    one = torch.ones(())

    def gradient(bulk_coordinates: torch.Tensor, skin_resultant: torch.Tensor):
        bulk_mu = bulk_coordinates.new_tensor(
            [float(spec["materials"][name]["mu_mpa"]) for name in BULK_TISSUES]
        )
        bulk_stress = bulk_mu[:, None, None] * symmetric_matrices(bulk_coordinates)
        model.set_materials(
            physics.materials(bulk_stress, skin_resultant, one, active_stress=None)
        )
        model.dof_map.fixed_values = physics.boundary(zero_pose)
        state.u = torch.zeros_like(physics.points_t)
        assert model.collision is not None
        state.collision = model.collision.state_at(state.u)
        value = physics.runtime.forward.problem.grad(state)
        assert torch.isfinite(value).all()
        contact = model.collision.diagnostics(state.collision, state.u)
        assert contact["contact_numerically_valid"] is True
        return value.detach().cpu().numpy(), contact

    LOG.info("Assembling exact-reference passive/contact force")
    g0, contact = gradient(torch.zeros((3, 6)), zero_skin)
    prior_contact = contact_validation["reference"]
    assert math.isclose(
        contact["minimum_active_distance_m"],
        prior_contact["minimum_active_distance_m"],
        rel_tol=0,
        abs_tol=1.0e-14,
    )
    assert math.isclose(
        contact["barrier_energy"],
        prior_contact["barrier_energy"],
        rel_tol=1.0e-6,
        abs_tol=1.0e-20,
    )
    LOG.info("Assembling prescribed 80.6 N/m skin force")
    g_skin_total, _ = gradient(torch.zeros((3, 6)), identity_skin)
    skin_load = g_skin_total - g0

    columns = []
    for index in range(18):
        LOG.info("Assembling constant bulk-stress force column %d/18", index + 1)
        coordinate = torch.zeros((3, 6))
        coordinate.flatten()[index] = 1.0
        value, _ = gradient(coordinate, zero_skin)
        columns.append(value - g0)
    matrix = np.column_stack(columns)

    free_indices = model.dof_map.free_indices.detach().cpu().numpy()
    nodal_dual_volume = np.zeros(len(physics.points))
    for local in range(4):
        np.add.at(nodal_dual_volume, physics.tets[:, local], physics.volumes / 4.0)
    row_dual_volume = nodal_dual_volume[free_indices // 3]
    assert np.all(row_dual_volume > 0)
    weights = 1.0 / np.sqrt(row_dual_volume)
    weights /= np.median(weights)
    weighted_matrix = weights[:, None] * matrix

    rng = np.random.default_rng(cfg.linearity_seed)
    test_coordinates = rng.normal(scale=0.2, size=18)
    test_skin_fraction = 0.37
    test_gradient, _ = gradient(
        torch.as_tensor(test_coordinates.reshape(3, 6)),
        test_skin_fraction * identity_skin,
    )
    predicted_gradient = g0 + matrix @ test_coordinates + test_skin_fraction * skin_load
    linearity_relative_error = float(
        np.linalg.norm(test_gradient - predicted_gradient)
        / np.linalg.norm(test_gradient - g0)
    )
    assert linearity_relative_error <= cfg.linearity_tolerance, (
        linearity_relative_error,
        cfg.linearity_tolerance,
    )

    epsilon = float(spec["constraints"]["baseline_epsilon"])
    lower = -(1.0 - epsilon)
    upper = float(spec["constraints"]["baseline_upper_mu_multiple"])
    mu_mpa = np.asarray(
        [float(spec["materials"][name]["mu_mpa"]) for name in BULK_TISSUES]
    )
    target_definitions = {
        "skin_only": -skin_load,
        "skin_plus_reference_contact": -(g0 + skin_load),
    }
    targets: dict[str, Any] = {}
    small_arrays: dict[str, np.ndarray] = {
        "gram_raw": matrix.T @ matrix,
        "gram_weighted": weighted_matrix.T @ weighted_matrix,
        "column_norms_raw": np.linalg.norm(matrix, axis=0),
        "column_norms_weighted": np.linalg.norm(weighted_matrix, axis=0),
    }
    for target_name, target in target_definitions.items():
        weighted_target = weights * target
        raw_fit = solve_scaled_lstsq(matrix, target)
        weighted_fit = solve_scaled_lstsq(weighted_matrix, weighted_target)
        solutions: dict[str, Any] = {
            "unconstrained_raw": {
                **force_metrics(
                    matrix,
                    target,
                    weights,
                    raw_fit["coefficients"],
                    mu_mpa,
                    lower=lower,
                    upper=upper,
                ),
                "rank": raw_fit["rank"],
                "singular_values_preconditioned": raw_fit[
                    "singular_values_preconditioned"
                ].tolist(),
                "condition_number_preconditioned": raw_fit[
                    "condition_number_preconditioned"
                ],
            },
            "unconstrained_weighted": {
                **force_metrics(
                    matrix,
                    target,
                    weights,
                    weighted_fit["coefficients"],
                    mu_mpa,
                    lower=lower,
                    upper=upper,
                ),
                "rank": weighted_fit["rank"],
                "singular_values_preconditioned": weighted_fit[
                    "singular_values_preconditioned"
                ].tolist(),
                "condition_number_preconditioned": weighted_fit[
                    "condition_number_preconditioned"
                ],
            },
        }
        projected = project_spectra(
            weighted_fit["coefficients"], lower=lower, upper=upper
        )
        solutions["projected_unconstrained_weighted"] = force_metrics(
            matrix,
            target,
            weights,
            projected,
            mu_mpa,
            lower=lower,
            upper=upper,
        )
        for ridge_weight in cfg.ridge_weights:
            bounded = solve_bounded(
                weighted_matrix,
                weighted_target,
                projected,
                lower=lower,
                upper=upper,
                ridge_weight=ridge_weight,
            )
            label = f"bounded_weighted_ridge_{ridge_weight:g}"
            solutions[label] = {
                **force_metrics(
                    matrix,
                    target,
                    weights,
                    bounded["coefficients"],
                    mu_mpa,
                    lower=lower,
                    upper=upper,
                ),
                **{
                    key: value
                    for key, value in bounded.items()
                    if key != "coefficients"
                },
            }
        targets[target_name] = {
            "target_force_norm_n": float(np.linalg.norm(target) * 1.0e6),
            "target_nodal_force_rms_n": float(
                np.sqrt(np.mean(np.sum((target.reshape(-1, 3) * 1.0e6) ** 2, axis=1)))
            ),
            "solutions": solutions,
        }
        small_arrays[f"rhs_raw_{target_name}"] = matrix.T @ target
        small_arrays[f"rhs_weighted_{target_name}"] = (
            weighted_matrix.T @ weighted_target
        )
    np.savez(output / "force-space-normal-equations.npz", **small_arrays)

    summary = {
        "schema": "joint-neutral-constant-basis-audit-v1",
        "success": True,
        "status": "passed_exact_reference_force_space_audit",
        "scope": (
            "linear free-force representability at exact F=I; no equilibrium solve "
            "and no inference of the 0.25 mm deformation-budget feasibility"
        ),
        "skin_target_n_per_m": cfg.skin_target_n_per_m,
        "skin_target_status": (
            "frozen Flynn-derived literature proxy and continuation target; not a "
            "measurement on this subject or registered full-face stress map"
        ),
        "basis": {
            "shared_total_coefficients": 20,
            "solved_compensating_bulk_coefficients": 18,
            "skin_coordinate": "fixed to the prescribed 80.6 N/m target",
            "skin_stiffness_coordinate": (
                "fixed at multiplier 1; its passive reference force is zero at F=I"
            ),
            "bulk_tissue_order": list(BULK_TISSUES),
            "coordinate_order": list(COORDINATE_NAMES),
            "dimensionless_spectral_bounds": [lower, upper],
            "mu_mpa": dict(zip(BULK_TISSUES, mu_mpa.tolist(), strict=True)),
            "prior": "zero-centered dimensionless bulk coordinates",
            "ridge_sensitivity_weights": list(cfg.ridge_weights),
            "ridge_sensitivity_status": (
                "dimensionless sensitivity after normalizing the weighted force "
                "objective; not the final inverse objective's prior calibration"
            ),
        },
        "weighted_metric": {
            "definition": (
                "each free force coordinate is weighted by inverse square root of "
                "tetrahedral nodal dual volume, then normalized to median weight 1"
            ),
            "purpose": "diagnostic force-density balance across nonuniform mesh volume",
            "minimum_nodal_dual_volume_m3": float(row_dual_volume.min()),
            "median_nodal_dual_volume_m3": float(np.median(row_dual_volume)),
            "maximum_nodal_dual_volume_m3": float(row_dual_volume.max()),
        },
        "reference": {
            "free_coordinates": int(matrix.shape[0]),
            "free_nodes": int(matrix.shape[0] // 3),
            "bulk_force_columns": int(matrix.shape[1]),
            "passive_and_contact_force_norm_n": float(np.linalg.norm(g0) * 1.0e6),
            "skin_force_norm_n": float(np.linalg.norm(skin_load) * 1.0e6),
            "contact": contact,
            "prior_contact_validation_reference": prior_contact,
            "active_pair_count_delta_from_prior_receipt": int(
                contact["active_contact_count"] - prior_contact["active_contact_count"]
            ),
            "active_pair_count_note": (
                "IPC candidate enumeration count is recorded rather than required to "
                "equal the older receipt; barrier energy and minimum active distance "
                "must reproduce within declared tolerances"
            ),
            "contact_spec_sha256": sha256(cfg.contact_spec),
            "contact_validation_sha256": sha256(cfg.contact_validation),
            "prepared_manifest_sha256": sha256(prepared.manifest_path),
        },
        "linearity_check": {
            "relative_error": linearity_relative_error,
            "tolerance": cfg.linearity_tolerance,
            "random_seed": cfg.linearity_seed,
            "skin_fraction": test_skin_fraction,
        },
        "targets": targets,
        "interpretation": {
            "exact_rest": (
                "relative residuals quantify how much of the prescribed reference "
                "free-force vector is outside the 18-column constant-bulk-stress span"
            ),
            "deformation_budget": (
                "a nonzero exact-rest residual does not prove failure of the nonlinear "
                "equilibrium or its 0.25 mm motion budget; that depends on stiffness, "
                "contact, geometry, and finite-deformation response"
            ),
            "basis_refinement": (
                "a material bounded residual can motivate testing a spatial bulk-stress "
                "basis, but this audit does not change or reject the approved basis"
            ),
        },
        "gpu_timing": {
            "wall_seconds": time.perf_counter() - started,
            "contention": (
                "run concurrently with neutral segment-003 and an unrelated process 40; "
                "full-face validation 09 had released its GPU load"
            ),
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "material-config.json", spec)
    cherries.log_output(output)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
