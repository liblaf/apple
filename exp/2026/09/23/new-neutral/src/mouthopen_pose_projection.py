"""CPU active-set utilities for feasibility-aware 6-DoF jaw directions.

The inverse runner supplies determinant directional derivatives from its coupled
predictor.  This module only projects a six-coordinate jaw proposal against
those linearized constraints; it does not change FEM, contact, or tet policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


class PoseProjectionError(RuntimeError):
    """Raised when the declared linearized jaw-feasibility QP has no solution."""


@dataclass(frozen=True)
class RetainedDeterminants:
    """Signed determinant ratios for retained original tetrahedron IDs."""

    original_tetrahedron_ids: np.ndarray
    det_f: np.ndarray


def determinant_ratio(
    full_reference_points_m: np.ndarray,
    tetrahedra: np.ndarray,
    displacement_m: np.ndarray,
) -> np.ndarray:
    """Return det(F) for every supplied original tetrahedron on CPU."""
    reference = np.asarray(full_reference_points_m, dtype=np.float64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    displacement = np.asarray(displacement_m, dtype=np.float64)
    assert reference.ndim == 2
    assert reference.shape[1] == 3
    assert tets.ndim == 2
    assert tets.shape[1] == 4
    assert displacement.ndim == 2
    assert displacement.shape[1] == 3
    assert displacement.shape[0] >= reference.shape[0]
    assert np.isfinite(reference).all()
    assert np.isfinite(displacement[: len(reference)]).all()
    assert np.all((tets >= 0) & (tets < len(reference)))
    rest_edges = np.transpose(
        reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1)
    )
    deformed = reference + displacement[: len(reference)]
    deformed_edges = np.transpose(
        deformed[tets[:, 1:]] - deformed[tets[:, :1]], (0, 2, 1)
    )
    rest_det = np.linalg.det(rest_edges)
    assert np.isfinite(rest_det).all()
    assert np.all(rest_det > 0)
    result = np.linalg.det(deformed_edges) / rest_det
    assert np.isfinite(result).all()
    return result


def determinant_directional_derivative(
    reference: np.ndarray,
    tetrahedra: np.ndarray,
    displacement: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Differentiate det(F) by cofactors without inverting deformed cells."""
    reference = np.asarray(reference, dtype=np.float64)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    displacement = np.asarray(displacement, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    assert reference.ndim == 2
    assert reference.shape[1] == 3
    assert tetrahedra.ndim == 2
    assert tetrahedra.shape[1] == 4
    assert direction.shape == displacement.shape
    assert displacement.shape[0] >= len(reference)
    assert np.isfinite(direction).all()
    rest = reference[tetrahedra[:, 1:]] - reference[tetrahedra[:, :1]]
    points = reference + displacement[: len(reference)]
    edges = points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]]
    delta = direction[tetrahedra[:, 1:]] - direction[tetrahedra[:, :1]]
    rest_det = np.linalg.det(rest)
    assert np.all(rest_det > 0)
    derivative = (
        np.einsum("ij,ij->i", delta[:, 0], np.cross(edges[:, 1], edges[:, 2]))
        + np.einsum("ij,ij->i", edges[:, 0], np.cross(delta[:, 1], edges[:, 2]))
        + np.einsum("ij,ij->i", edges[:, 0], np.cross(edges[:, 1], delta[:, 2]))
    ) / rest_det
    assert np.isfinite(derivative).all()
    return derivative


def active_retained_noninverted_cells(
    full_reference_points_m: np.ndarray,
    tetrahedra: np.ndarray,
    displacement_m: np.ndarray,
    retained_tetrahedron_ids: np.ndarray,
    *,
    det_f_threshold: float = 0.05,
) -> RetainedDeterminants:
    """Select retained cells with ``0 < det(F) <= threshold``.

    Existing inverted cells are intentionally excluded: their count/volume is
    governed by the existing allowance.  The returned original cell IDs let the
    runner prevent another cell from crossing through zero.
    """
    assert det_f_threshold > 0
    retained = np.asarray(retained_tetrahedron_ids, dtype=np.int64)
    tets = np.asarray(tetrahedra, dtype=np.int64)
    assert retained.ndim == 1
    assert retained.size
    assert np.all((retained >= 0) & (retained < len(tets)))
    assert np.unique(retained).size == retained.size
    all_det_f = determinant_ratio(full_reference_points_m, tets, displacement_m)
    selected = retained[
        (all_det_f[retained] > 0) & (all_det_f[retained] <= det_f_threshold)
    ]
    return RetainedDeterminants(
        original_tetrahedron_ids=selected,
        det_f=all_det_f[selected],
    )


def _constraint_diagnostics(
    direction: np.ndarray,
    requested: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    slack = matrix @ direction - lower
    active = slack <= tolerance
    multipliers = np.zeros(len(lower), dtype=np.float64)
    kkt_residual = 0.0
    if active.any():
        # KKT for min .5||x-r||², lower-Ax <= 0: x-r-A.T lambda=0.
        solved, *_ = np.linalg.lstsq(
            matrix[active].T, direction - requested, rcond=None
        )
        multipliers[active] = solved
        kkt_residual = float(
            np.linalg.norm(direction - requested - matrix.T @ multipliers)
        )
    return {
        "minimum_slack": float(slack.min()) if slack.size else float("inf"),
        "maximum_violation": float(max(0.0, -slack.min())) if slack.size else 0.0,
        "active_constraints": int(active.sum()),
        "multipliers": multipliers.tolist(),
        "minimum_multiplier": float(multipliers.min()) if multipliers.size else 0.0,
        "kkt_stationarity_residual": kkt_residual,
    }


def project_pose_direction(
    requested_dp: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a normalized 6-DoF pose proposal onto ``matrix @ dp >= lower``.

    The returned vector is the Euclidean nearest feasible direction.  Fail fast
    on an infeasible or unresolved QP rather than silently weakening a geometry
    constraint.
    """
    requested = np.asarray(requested_dp, dtype=np.float64)
    constraints = np.asarray(matrix, dtype=np.float64)
    bounds = np.asarray(lower, dtype=np.float64)
    assert requested.shape == (6,)
    assert constraints.ndim == 2
    assert constraints.shape[1] == 6
    assert bounds.shape == (constraints.shape[0],)
    assert np.isfinite(requested).all()
    assert np.isfinite(constraints).all()
    assert np.isfinite(bounds).all()
    assert tolerance >= 0
    assert max_iterations > 0

    if not len(bounds) or bool(np.all(constraints @ requested >= bounds - tolerance)):
        diagnostics = _constraint_diagnostics(
            requested, requested, constraints, bounds, tolerance=tolerance
        )
        return requested.copy(), {
            "projection_applied": False,
            "objective": 0.0,
            "solver": "not-needed",
            "constraints": diagnostics,
        }

    result = minimize(
        fun=lambda value: 0.5 * float(np.dot(value - requested, value - requested)),
        x0=requested.copy(),
        jac=lambda value: value - requested,
        constraints={
            "type": "ineq",
            "fun": lambda value: constraints @ value - bounds,
            "jac": lambda _value: constraints,
        },
        method="SLSQP",
        options={
            "ftol": tolerance if tolerance > 0 else 1e-12,
            "maxiter": max_iterations,
        },
    )
    direction = np.asarray(result.x, dtype=np.float64)
    if not result.success or not np.isfinite(direction).all():
        message = (
            "jaw feasibility QP unresolved: "
            f"status={result.status} message={result.message}"
        )
        raise PoseProjectionError(message)
    diagnostics = _constraint_diagnostics(
        direction, requested, constraints, bounds, tolerance=tolerance
    )
    if diagnostics["maximum_violation"] > tolerance:
        message = (
            "jaw feasibility QP violates a linearized determinant constraint: "
            f"max_violation={diagnostics['maximum_violation']:.3e}"
        )
        raise PoseProjectionError(message)
    return direction, {
        "projection_applied": True,
        "objective": float(result.fun),
        "solver": "scipy.optimize.SLSQP",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "constraints": diagnostics,
    }


def project_pose_descent_direction(
    requested_dp: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    *,
    pose_gradient: np.ndarray,
    strain_directional: float,
    descent_target: float,
    feasible_witness: bool = False,
    witness_policy: str = "zero_pose",
    witness_receipt: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Require joint model descent as well as linearized tet feasibility.

    The strain proposal is fixed. The extra halfspace is normalized so the
    QP feasibility tolerance does not hide a small objective derivative.
    This only constrains the supplied gradient model; nonlinear acceptance
    and any independent stationarity audit remain the caller's responsibility.
    """
    gradient = np.asarray(pose_gradient, dtype=np.float64)
    assert gradient.shape == (6,)
    assert np.isfinite(gradient).all()
    assert np.isfinite(strain_directional)
    assert np.isfinite(descent_target)
    assert descent_target < 0
    norm = float(np.linalg.norm(gradient))
    assert norm > 0
    combined_matrix = np.vstack((matrix, -gradient / norm))
    combined_lower = np.append(lower, (strain_directional - descent_target) / norm)
    if feasible_witness:
        from mouthopen_descent_witness import solve_certified_projection

        diagnostics = {} if witness_receipt is None else witness_receipt
        try:
            direction, diagnostics = solve_certified_projection(
                np.asarray(requested_dp, dtype=np.float64),
                np.asarray(matrix, dtype=np.float64),
                np.asarray(lower, dtype=np.float64),
                gradient,
                strain_directional,
                descent_target,
                diagnostics,
                policy=witness_policy,
            )
        except (AssertionError, ValueError, RuntimeError) as error:
            diagnostics["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            message = "Certified pose descent witness unresolved"
            raise PoseProjectionError(message) from error
        descent_target = diagnostics["chosen_target"]
        combined_lower[-1] = (strain_directional - descent_target) / norm
        receipt = {
            "projection_applied": not np.array_equal(direction, requested_dp),
            "objective": 0.5 * float(np.sum((direction - requested_dp) ** 2)),
            "solver": "scipy.optimize.linprog(HiGHS)+minimize(SLSQP)",
            "constraints": _constraint_diagnostics(
                direction,
                requested_dp,
                combined_matrix,
                combined_lower,
                tolerance=1e-10,
            ),
            "feasible_witness": diagnostics,
        }
    else:
        direction, receipt = project_pose_direction(
            requested_dp, combined_matrix, combined_lower
        )
    projected_slope = float(strain_directional + gradient @ direction)
    assert projected_slope <= descent_target + norm * 1e-10
    receipt["descent"] = {
        "strain_directional": strain_directional,
        "requested_pose_directional": float(gradient @ requested_dp),
        "requested_joint_directional": float(
            strain_directional + gradient @ requested_dp
        ),
        "projected_pose_directional": float(gradient @ direction),
        "projected_joint_directional": projected_slope,
        "joint_directional_upper_bound": descent_target,
        "pose_gradient_norm": norm,
        "constraint_index": len(lower),
        "gradient_model_only": True,
    }
    return direction, receipt


__all__ = [
    "PoseProjectionError",
    "RetainedDeterminants",
    "active_retained_noninverted_cells",
    "determinant_directional_derivative",
    "determinant_ratio",
    "project_pose_descent_direction",
    "project_pose_direction",
]
