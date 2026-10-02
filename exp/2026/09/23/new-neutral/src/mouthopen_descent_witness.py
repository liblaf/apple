"""Explicit feasibility classification and certified descent witness for pose QPs.

This only selects a linearized descent target; nonlinear physical acceptance
remains the caller's responsibility. No optimizer state is changed here.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize, nnls


def attainable_descent_witness(
    matrix: np.ndarray,
    lower: np.ndarray,
    gradient: np.ndarray,
    strain_slope: float,
    receipt: dict,
) -> tuple[np.ndarray, float]:
    """Certify a finite geometry-only LP optimum before choosing half its descent."""
    optimum = linprog(
        gradient,
        A_ub=-matrix,
        b_ub=-lower,
        bounds=[(None, None)] * 6,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    receipt.update(status=int(optimum.status), message=str(optimum.message))
    assert optimum.status == 0, "Geometry descent LP has no certified finite optimum"
    point = np.asarray(optimum.x, dtype=np.float64)
    multipliers = -np.asarray(optimum.ineqlin.marginals, dtype=np.float64)
    assert np.isfinite(point).all()
    assert np.isfinite(multipliers).all()
    slack = matrix @ point - lower
    pose_slope = float(gradient @ point)
    best_joint_slope = strain_slope + pose_slope
    dual_objective = float(lower @ multipliers)
    stationarity = float(np.linalg.norm(gradient - matrix.T @ multipliers))
    gap = pose_slope - dual_objective
    complementarity = float(np.max(np.abs(multipliers * slack)))
    receipt.update(
        feasible_point=point.tolist(),
        multipliers=multipliers.tolist(),
        minimum_slack=float(slack.min()),
        minimum_multiplier=float(multipliers.min()),
        pose_slope=pose_slope,
        best_joint_slope=best_joint_slope,
        dual_objective=dual_objective,
        dual_stationarity_residual=stationarity,
        duality_gap=gap,
        complementarity_residual=complementarity,
    )
    assert float(slack.min()) >= -1e-10, receipt
    assert np.all(multipliers >= 0), receipt
    assert stationarity <= 1e-10, receipt
    assert abs(gap) <= 1e-10, receipt
    assert complementarity <= 1e-10, receipt
    assert best_joint_slope < 0, "Geometry LP optimum provides no joint descent"
    return point, 0.5 * best_joint_slope


def solve_certified_projection(  # noqa: PLR0915
    requested: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    gradient: np.ndarray,
    strain_slope: float,
    target: float,
    receipt: dict,
    *,
    policy: str = "zero_pose",
) -> tuple[np.ndarray, dict]:
    """Classify the original target and document every alternate solver branch."""
    assert policy in ("zero_pose", "attainable_optimum")
    receipt["policy"] = policy
    assert requested.shape == (6,)
    assert matrix.ndim == 2
    assert matrix.shape[1] == 6
    assert lower.shape == (len(matrix),)
    assert gradient.shape == (6,)
    assert np.isfinite(requested).all()
    assert np.isfinite(matrix).all()
    assert np.isfinite(lower).all()
    assert np.isfinite(gradient).all()
    assert np.isfinite(strain_slope)
    assert np.isfinite(target)
    norm = float(np.linalg.norm(gradient))
    assert norm > 0
    assert target < 0
    combined = np.vstack((matrix, -gradient / norm))
    bounds = np.r_[lower, (strain_slope - target) / norm]
    lp = linprog(
        np.zeros(6),
        A_ub=-combined,
        b_ub=-bounds,
        bounds=[(None, None)] * 6,
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    receipt.update(
        {
            "original_target": target,
            "strain_slope": strain_slope,
            "original_lp": {
                "status": int(lp.status),
                "message": str(lp.message),
                "feasible_point": lp.x.tolist() if lp.x is not None else None,
            },
        }
    )
    if lp.status == 2:
        if policy == "attainable_optimum":
            optimum_receipt = {}
            receipt["geometry_descent_lp"] = optimum_receipt
            initial, chosen_target = attainable_descent_witness(
                matrix, lower, gradient, strain_slope, optimum_receipt
            )
            assert optimum_receipt["best_joint_slope"] >= target - 1e-10
            target = chosen_target
            receipt["branch"] = "original_target_infeasible_use_half_attainable_descent"
        else:
            assert np.all(lower <= 0), "Zero-pose geometry witness unavailable"
            assert strain_slope < 0, "Zero-pose witness is not descending"
            target = 0.5 * strain_slope
            initial = np.zeros(6)
            receipt["zero_pose_witness"] = {
                "geometry_minimum_slack": float((-lower).min()),
                "joint_slope": strain_slope,
                "objective_slack_original_units": target - strain_slope,
            }
            receipt["branch"] = "original_target_infeasible_use_half_zero_pose_descent"
        bounds[-1] = (strain_slope - target) / norm
    else:
        assert lp.status == 0, receipt
        lp_geometry_slack = float(np.min(matrix @ lp.x - lower))
        lp_objective_slack = float(target - strain_slope - gradient @ lp.x)
        receipt["original_lp"]["geometry_minimum_slack"] = lp_geometry_slack
        receipt["original_lp"]["objective_slack_original_units"] = lp_objective_slack
        assert lp_geometry_slack >= -1e-10
        assert lp_objective_slack >= -1e-10
        assert float(np.min(combined @ lp.x - bounds)) >= -1e-10
        initial = lp.x.copy()
        receipt["branch"] = "original_target_feasible_start_from_lp_witness"
    receipt["chosen_target"] = target
    attempts = []
    receipt["qp_attempts"] = attempts

    def solve(initial_point: np.ndarray):
        result = minimize(
            lambda x: 0.5 * float(np.dot(x - requested, x - requested)),
            initial_point,
            jac=lambda x: x - requested,
            method="SLSQP",
            constraints={
                "type": "ineq",
                "fun": lambda x: combined @ x - bounds,
                "jac": lambda _: combined,
            },
            options={"ftol": 1e-10, "maxiter": 1000},
        )
        attempts.append(
            {
                "status": int(result.status),
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit),
                "minimum_slack": float(np.min(combined @ result.x - bounds)),
                "initial_point": initial_point.tolist(),
                "result": result.x.tolist(),
            }
        )
        return result

    result = solve(initial)
    receipt["qp_attempts"] = attempts
    assert result.success, receipt
    direction = np.asarray(result.x)
    slack = combined @ direction - bounds
    assert np.isfinite(direction).all()
    assert float(slack.min()) >= -1e-10, receipt
    active = slack <= 1e-8
    multipliers = np.zeros(len(bounds))
    if active.any():
        multipliers[active], _ = nnls(
            combined[active].T, direction - requested, maxiter=10000
        )
    stationarity = float(
        np.linalg.norm(direction - requested - combined.T @ multipliers)
    )
    complementarity = float(np.max(np.abs(multipliers * slack)))
    assert stationarity <= 1e-7, stationarity
    assert complementarity <= 1e-8, complementarity
    slope = float(strain_slope + gradient @ direction)
    assert slope < 0
    assert slope <= target + norm * 1e-10
    receipt.update(
        minimum_slack=float(slack.min()),
        kkt_stationarity_residual=stationarity,
        kkt_complementarity_residual=complementarity,
        nonnegative_multipliers=multipliers.tolist(),
        projected_joint_slope=slope,
        direction=direction.tolist(),
    )
    return direction, receipt
