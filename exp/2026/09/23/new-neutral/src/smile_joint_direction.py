"""CPU projection of actual joint increments in a persisted Adam inverse metric."""

# ruff: noqa: C901, PLR0915, TRY300, TRY301
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

DUAL_PROJECTED_GRADIENT_TOLERANCE = 1e-11


class JointProjectionError(RuntimeError):
    """A joint projection has no certified result within its declared policy."""


class JointTrustRegionInfeasibleError(JointProjectionError):
    """The linear rows and box cannot fit inside the declared metric trust ball."""

    def __init__(self, message: str, *, receipt: dict) -> None:
        super().__init__(message)
        self.receipt = receipt


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def record(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def immutable(value: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class JointProjectionCache:
    """Owned immutable arrays for one outer iteration, bound to a saved file."""

    original_ids: np.ndarray
    old_j: np.ndarray
    residual_delta_j: np.ndarray
    selected_ids: np.ndarray
    selected_indices: np.ndarray
    determinant_gradients: np.ndarray
    objective_gradient: np.ndarray
    inverse_metric: np.ndarray
    proposal: np.ndarray
    q_shape: tuple[int, ...]
    source_path: Path
    source_sha256: str


def save_joint_projection_cache(
    directory: Path,
    *,
    original_ids: np.ndarray,
    old_j: np.ndarray,
    residual_delta_j: np.ndarray,
    selected_ids: np.ndarray,
    determinant_gradients: np.ndarray,
    objective_gradient: np.ndarray,
    inverse_metric: np.ndarray,
    proposal: np.ndarray,
    q_shape: tuple[int, ...],
) -> JointProjectionCache:
    """Persist and own all numerical inputs; q/pose history remains with the runner."""
    arrays = {
        "original_ids": immutable(original_ids, np.int64),
        "old_j": immutable(old_j),
        "residual_delta_j": immutable(residual_delta_j),
        "selected_ids": immutable(selected_ids, np.int64),
        "determinant_gradients": immutable(determinant_gradients),
        "objective_gradient": immutable(objective_gradient),
        "inverse_metric": immutable(inverse_metric),
        "proposal": immutable(proposal),
    }
    ids, chosen = arrays["original_ids"], arrays["selected_ids"]
    assert np.all(np.diff(ids) > 0)
    assert np.all(np.diff(chosen) > 0)
    indices = np.searchsorted(ids, chosen)
    np.testing.assert_array_equal(ids[indices], chosen)
    arrays["selected_indices"] = immutable(indices, np.int64)
    n = int(np.prod(q_shape)) + 6
    assert arrays["old_j"].shape == arrays["residual_delta_j"].shape == ids.shape
    assert arrays["determinant_gradients"].shape == (len(chosen), n)
    assert (
        arrays["proposal"].shape
        == arrays["objective_gradient"].shape
        == arrays["inverse_metric"].shape
        == (n,)
    )
    assert len(chosen) <= 16
    assert np.all(arrays["old_j"][indices] > 0)
    assert np.all(arrays["inverse_metric"] > 0)
    assert all(np.isfinite(value).all() for value in arrays.values())
    directory.mkdir(parents=True, exist_ok=False)
    path = directory.resolve() / "coefficients.npz"
    np.savez_compressed(path, **arrays, q_shape=np.asarray(q_shape, dtype=np.int64))
    binding = record(path)
    write_json(
        directory / "manifest.json",
        {
            "coefficients": binding,
            "selected_original_ids": chosen.tolist(),
            "q_shape": list(q_shape),
            "residual_response_in_seed": False,
            "extra_positive_screen_is_physical_gate": False,
        },
    )
    return JointProjectionCache(
        **arrays,
        q_shape=tuple(q_shape),
        source_path=path,
        source_sha256=binding["sha256"],
    )


def project_joint_increment(
    cache: JointProjectionCache,
    alpha: float,
    output: Path,
    *,
    margin: float = 1e-6,
    strain_limit: float = 0.01,
    descent_fraction: float = 0.1,
    trust_ratio: float = 2.0,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return actual q and normalized-pose increments, already scaled by alpha."""
    assert 0 < alpha <= 1
    assert margin > 0
    assert strain_limit > 0
    assert 0 < descent_fraction <= 1
    assert trust_ratio > 0
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "running",
        "alpha": alpha,
        "margin": margin,
        "strain_limit": strain_limit,
        "descent_fraction": descent_fraction,
        "trust_ratio": trust_ratio,
        "cache": {"path": str(cache.source_path), "sha256": cache.source_sha256},
        "selected_original_ids": cache.selected_ids.tolist(),
        "returns_actual_increments": True,
        "residual_response_in_seed": False,
        "extra_positive_screen_is_physical_gate": False,
    }
    stage = "source_binding"
    try:
        assert record(cache.source_path) == receipt["cache"]
        if deadline is not None and time.perf_counter() >= deadline:
            message = "Declared joint projection deadline expired"
            raise JointProjectionError(message)
        x0 = alpha * cache.proposal
        baseline_slope = float(cache.objective_gradient @ x0)
        assert baseline_slope < 0
        target = descent_fraction * baseline_slope
        baseline_norm = float(np.linalg.norm(x0 / np.sqrt(cache.inverse_metric)))
        maximum_correction_objective = 0.5 * (trust_ratio * baseline_norm) ** 2
        matrix = np.vstack((cache.determinant_gradients, -cache.objective_gradient))
        selected_j = cache.old_j[cache.selected_indices]
        residual = cache.residual_delta_j[cache.selected_indices]
        lower = np.r_[margin - selected_j - np.minimum(residual, 0), -target]
        lower_box = np.full_like(x0, -strain_limit)
        upper_box = np.full_like(x0, strain_limit)
        lower_box[-6:], upper_box[-6:] = -np.inf, np.inf
        np.savez_compressed(
            output / "inputs.npz",
            selected_original_ids=cache.selected_ids,
            lower=lower,
            x0=x0,
            diagonal_inverse_metric=cache.inverse_metric,
            lower_box=lower_box,
            upper_box=upper_box,
            baseline_slope=baseline_slope,
            descent_target=target,
            maximum_correction_objective=maximum_correction_objective,
        )
        write_json(
            output / "input-receipt.json",
            {
                "cache": receipt["cache"],
                "inputs": record(output / "inputs.npz"),
                "row_order": [
                    *cache.selected_ids.tolist(),
                    "negative_objective_gradient",
                ],
            },
        )
        stage = "bounded_dual"
        increment, certificate = solve_bounded_dual(
            matrix,
            lower,
            x0,
            cache.inverse_metric,
            lower_box,
            upper_box,
            receipt_path=output / "certificate.json",
            deadline=deadline,
            maximum_correction_objective=maximum_correction_objective,
        )
        actual_slope = float(cache.objective_gradient @ increment)
        metric_correction = float(
            np.linalg.norm((increment - x0) / np.sqrt(cache.inverse_metric))
        )
        baseline_norm = float(np.linalg.norm(x0 / np.sqrt(cache.inverse_metric)))
        predicted_seed = selected_j + cache.determinant_gradients @ increment
        predicted_affine = predicted_seed + residual
        np.savez_compressed(
            output / "increment.npz",
            delta_q=increment[:-6].reshape(cache.q_shape),
            delta_pose=increment[-6:],
            actual_joint_increment=increment,
            selected_original_ids=cache.selected_ids,
            predicted_seed_J=predicted_seed,
            predicted_affine_J=predicted_affine,
        )
        receipt.update(
            baseline_joint_increment_slope=baseline_slope,
            chosen_descent_target=target,
            chosen_increment_target=target,
            active_cell_count=len(cache.selected_ids),
            minimum_linearized_projected_J=float(
                np.minimum(predicted_seed, predicted_affine).min()
            )
            if len(cache.selected_ids)
            else None,
            prediction_scope="selected determinant rows; minimum of seed and residual-affine predictions",
            projection={
                **certificate,
                "correction_metric_norm": metric_correction,
                "trust_norm_limit": trust_ratio * baseline_norm,
                "maximum_absolute_delta_q": float(np.max(abs(increment[:-6]))),
                "strain_limit": strain_limit,
            },
            actual_joint_increment_slope=actual_slope,
            correction_metric_norm=metric_correction,
            baseline_metric_norm=baseline_norm,
            trust_norm_limit=trust_ratio * baseline_norm,
            maximum_absolute_delta_q=float(np.max(abs(increment[:-6]))),
            actual_pose_increment=increment[-6:].tolist(),
            certificate=record(output / "certificate.json"),
            increments=record(output / "increment.npz"),
            all_retained_positive_predictions_available=False,
        )
        stage = "trust_and_descent"
        assert metric_correction <= trust_ratio * baseline_norm
        assert actual_slope <= target + 1e-12
        assert np.max(abs(increment[:-6])) <= strain_limit
        receipt["status"] = "certified_joint_increment"
        return increment[:-6].reshape(cache.q_shape), increment[-6:], receipt
    except JointTrustRegionInfeasibleError as error:
        receipt.update(
            status="joint_trust_region_infeasible",
            alpha_rejected=True,
            maximum_correction_objective=maximum_correction_objective,
            projection=error.receipt,
            certificate=record(output / "certificate.json"),
            failure={
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        error.receipt = receipt
        raise
    except Exception as error:
        receipt.update(
            status="joint_projection_failed",
            failure={
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            },
        )
        raise
    finally:
        write_json(output / "summary.json", receipt)


def clipped_dual_objective_change(
    raw: np.ndarray,
    raw_delta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    directional_change: float,
) -> float:
    """Evaluate the same dual's exact local change without scalar subtraction.

    For z=clip(raw), the separable conjugate is raw*z-.5*z**2.
    Its change minus z*raw_delta is the nonnegative clipped-quadratic
    remainder below, including steps that change the active box coordinates.
    """
    clipped = np.clip(raw, lower, upper)
    next_clipped = np.clip(raw + raw_delta, lower, upper)
    delta_clipped = next_clipped - clipped
    remainder = (raw_delta + (raw - clipped)) * delta_clipped - 0.5 * delta_clipped**2
    return directional_change + float(np.sum(remainder))


def dual_correction_lower_bound(
    matrix: np.ndarray,
    lower: np.ndarray,
    x0: np.ndarray,
    metric: np.ndarray,
    lower_box: np.ndarray,
    upper_box: np.ndarray,
    multipliers: np.ndarray,
) -> dict:
    """Bound the boxed correction optimum by dual feasibility and strong convexity.

    At a feasible boxed point xhat choose valid complementary box normals.
    With residual r=grad L-normal_lower+normal_upper, strong convexity gives
    inf L >= L(xhat,lambda)-.5*r.T*D*r. This does not require QP KKT convergence.
    Extended precision and an explicit conservative accumulation allowance
    account for the rounded evaluation; the raw point is never adopted.
    """
    assert np.isfinite(multipliers).all()
    assert np.all(multipliers >= 0)
    assert np.isfinite(matrix).all()
    assert np.isfinite(metric).all()
    assert np.all(metric > 0)
    point = np.clip(x0 + metric * (matrix.T @ multipliers), lower_box, upper_box)
    assert np.isfinite(point).all()
    assert np.all(point >= lower_box)
    assert np.all(point <= upper_box)
    extended = np.longdouble
    bmat = matrix.astype(extended)
    lam = multipliers.astype(extended)
    center, d, x = x0.astype(extended), metric.astype(extended), point.astype(extended)
    b = lower.astype(extended)
    gradient = (x - center) / d - bmat.T @ lam
    at_lower = np.isfinite(lower_box) & (point == lower_box)
    at_upper = np.isfinite(upper_box) & (point == upper_box)
    normal_lower = np.where(at_lower, np.maximum(gradient, 0), 0)
    normal_upper = np.where(at_upper, np.maximum(-gradient, 0), 0)
    residual = gradient - normal_lower + normal_upper
    correction = extended(0.5) * np.sum(d * residual**2, dtype=extended)
    primal = extended(0.5) * np.sum((x - center) ** 2 / d, dtype=extended)
    slack = bmat @ x - b
    multiplier_slack = np.sum(lam * slack, dtype=extended)
    lagrangian = primal - multiplier_slack
    raw_bound = lagrangian - correction
    # Deliberately broad double-precision envelope, even though reductions use
    # long double. The strict trust comparison uses the bound AFTER subtraction.
    count = matrix.shape[1] + matrix.shape[0] + 32
    unit = np.finfo(np.float64).eps
    gamma = count * unit / (1 - count * unit)
    absolute_row_terms = np.abs(bmat) @ np.abs(x) + np.abs(b)
    scale = (
        abs(primal)
        + abs(multiplier_slack)
        + correction
        + np.sum(abs(lam) * absolute_row_terms, dtype=extended)
        + 1
    )
    allowance = extended(64) * gamma * scale
    conservative = raw_bound - allowance
    assert np.isfinite(conservative)
    lower_complementarity = normal_lower[at_lower] * (x[at_lower] - lower_box[at_lower])
    upper_complementarity = normal_upper[at_upper] * (upper_box[at_upper] - x[at_upper])
    assert np.all(lower_complementarity == 0)
    assert np.all(upper_complementarity == 0)
    return {
        "method": "feasible nonnegative dual; strong-convexity residual correction; conservative floating allowance",
        "formula": "L(xhat,lambda) - .5*r.T*D*r - floating_allowance; r=gradL-lower_normal+upper_normal",
        "dual_multipliers": multipliers.tolist(),
        "minimum_dual_multiplier": float(multipliers.min()),
        "box_point_feasible": True,
        "box_normal_multipliers_nonnegative": True,
        "box_normal_complementarity_exact": True,
        "lower_active_count": int(at_lower.sum()),
        "upper_active_count": int(at_upper.sum()),
        "minimum_lower_normal": float(normal_lower.min()),
        "minimum_upper_normal": float(normal_upper.min()),
        "lagrangian_at_box_point": float(lagrangian),
        "primal_at_box_point": float(primal),
        "lambda_dot_slack": float(multiplier_slack),
        "box_stationarity_residual_inf": float(np.max(abs(residual))),
        "strong_convexity_residual_correction": float(correction),
        "dual_lower_bound_before_floating_allowance": float(raw_bound),
        "floating_allowance": float(allowance),
        "floating_allowance_formula": "64*gamma_(n+m+32)*(abs(primal)+abs(lambda_dot_slack)+residual_correction+sum(abs(lambda)*(abs(B)@abs(xhat)+abs(b)))+1)",
        "extended_precision_mantissa_bits": int(np.finfo(extended).nmant),
        "conservative_dual_lower_bound": float(conservative),
    }


def solve_bounded_dual(
    matrix: np.ndarray,
    lower: np.ndarray,
    x0: np.ndarray,
    metric: np.ndarray,
    lower_box: np.ndarray,
    upper_box: np.ndarray,
    *,
    receipt_path: Path | None = None,
    deadline: float | None = None,
    maximum_correction_objective: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve the exact clipped dual; an unresolved certificate is not infeasibility."""
    receipt = {
        "status": "running",
        "solver": "row/RHS normalized L-BFGS-B clipped dual plus bounded Newton polishing",
        "infeasibility_claimed": False,
        "polish_line_search": "Exact clipped-quadratic local dual change; Armijo coefficient1e-4; no objective-scalar subtraction",
        "polish_difference_expression": "slack.dot(delta) + sum((C.T@delta + raw - clip(raw))*delta_clip - .5*delta_clip**2); delta=proposal-mu",
    }

    def persist() -> None:
        if receipt_path is not None:
            write_json(receipt_path, receipt)

    def budget() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            message = "Declared bounded dual budget exhausted"
            raise JointProjectionError(message)

    try:
        assert np.isfinite(matrix).all()
        assert np.isfinite(metric).all()
        assert np.all(metric > 0)
        assert np.all(lower_box <= x0)
        assert np.all(x0 <= upper_box)
        root_metric = np.sqrt(metric)
        whitened = matrix * root_metric
        row_scales = np.linalg.norm(whitened, axis=1)
        assert np.all(row_scales > 0)
        c = whitened / row_scales[:, None]
        h = (lower - matrix @ x0) / row_scales
        rhs_scale = float(np.max(abs(h)))
        zero_deficit = rhs_scale == 0
        if zero_deficit:
            rhs_scale = 1.0
        hhat = h / rhs_scale
        lo = (lower_box - x0) / (rhs_scale * root_metric)
        hi = (upper_box - x0) / (rhs_scale * root_metric)
        receipt.update(
            row_scales=row_scales.tolist(),
            rhs_scale=rhs_scale,
            normalized_deficit=hhat.tolist(),
            zero_deficit=zero_deficit,
        )
        persist()

        def evaluate(mu: np.ndarray) -> tuple[float, np.ndarray]:
            budget()
            z = np.clip(c.T @ mu, lo, hi)
            slack = c @ z - hhat
            return float(mu @ slack - 0.5 * (z @ z)), slack

        result = minimize(
            evaluate,
            np.zeros(len(lower)),
            jac=True,
            bounds=[(0, None)] * len(lower),
            method="L-BFGS-B",
            options={
                "ftol": 1e-15,
                "gtol": DUAL_PROJECTED_GRADIENT_TOLERANCE,
                "maxiter": 1000,
                "maxls": 50,
            },
        )
        mu = np.asarray(result.x)
        receipt.update(
            solver_success=bool(result.success),
            solver_status=int(result.status),
            solver_message=str(result.message),
            solver_iterations=int(result.nit),
            dual_projected_gradient_tolerance=DUAL_PROJECTED_GRADIENT_TOLERANCE,
            polish=[],
        )
        persist()
        if maximum_correction_objective is not None:
            assert np.isfinite(maximum_correction_objective)
            assert maximum_correction_objective >= 0
            budget()
            bound = dual_correction_lower_bound(
                matrix,
                lower,
                x0,
                metric,
                lower_box,
                upper_box,
                rhs_scale * mu / row_scales,
            )
            bound["maximum_correction_objective"] = maximum_correction_objective
            bound["outside_trust_certified"] = (
                bound["conservative_dual_lower_bound"] > maximum_correction_objective
            )
            receipt["trust_region_lower_bound"] = bound
            persist()
            if bound["outside_trust_certified"]:
                receipt.update(
                    status="certified_outside_trust",
                    certified=False,
                    trust_region_infeasibility_certified=True,
                    linear_rows_box_infeasibility_claimed=False,
                )
                persist()
                message = "Dual lower bound exceeds declared metric trust correction objective"
                raise JointTrustRegionInfeasibleError(message, receipt=receipt)
        for iteration in range(12):
            _, slack = evaluate(mu)
            active = (mu > 1e-12) | (slack < -1e-11)
            projected_gradient = np.where(mu > 0, slack, np.minimum(slack, 0))
            receipt["polish"].append(
                {
                    "iteration": iteration,
                    "normalized_projected_gradient_inf": float(
                        np.max(abs(projected_gradient))
                    ),
                }
            )
            if np.max(abs(projected_gradient)) <= DUAL_PROJECTED_GRADIENT_TOLERANCE:
                receipt["polish_termination"] = "projected_gradient_tolerance"
                receipt["polish_termination_norm"] = float(
                    np.max(abs(projected_gradient))
                )
                break
            raw_z = c.T @ mu
            free = (raw_z > lo) & (raw_z < hi)
            hessian = c[active][:, free] @ c[active][:, free].T
            direction = np.zeros_like(mu)
            direction[active] = np.linalg.lstsq(hessian, -slack[active], rcond=1e-12)[0]
            assert float(direction @ slack) < 0, (
                "Bounded dual Newton direction unresolved"
            )
            maximum_step = 1.0
            negative = direction < 0
            if negative.any():
                maximum_step = min(
                    maximum_step, float(np.min(-mu[negative] / direction[negative]))
                )
            assert maximum_step > 0, "Bounded dual active-set step unresolved"
            for backtrack in range(50):
                proposal = mu + maximum_step * direction
                assert proposal.min() >= -1e-14
                proposal = np.maximum(proposal, 0)
                budget()
                actual_delta = proposal - mu
                directional_change = float(slack @ actual_delta)
                objective_change = clipped_dual_objective_change(
                    raw_z, c.T @ actual_delta, lo, hi, directional_change
                )
                if (
                    directional_change < 0
                    and objective_change <= 1e-4 * directional_change
                ):
                    receipt["polish"][-1].update(
                        accepted_step=maximum_step,
                        backtracks=backtrack,
                        stable_objective_change=objective_change,
                        armijo_required_change=1e-4 * directional_change,
                        maximum_multiplier_change=float(np.max(abs(actual_delta))),
                    )
                    mu = proposal
                    break
                maximum_step *= 0.5
            else:
                message = "Bounded dual Newton line search unresolved"
                raise AssertionError(message)
        multipliers = rhs_scale * mu / row_scales
        raw = x0 + metric * (matrix.T @ multipliers)
        increment = np.clip(raw, lower_box, upper_box)
        # This is the exact dual map itself; no subsequent proposal clipping.
        slack = matrix @ increment - lower
        normalized_slack = slack / (row_scales * rhs_scale)
        normalized_complementarity = mu * normalized_slack
        complementarity = multipliers * slack
        stationarity = (increment - x0) / metric - matrix.T @ multipliers
        at_lower = np.isfinite(lower_box) & (increment == lower_box)
        at_upper = np.isfinite(upper_box) & (increment == upper_box)
        interior = ~at_lower & ~at_upper
        lower_multipliers = np.where(at_lower, stationarity, 0)
        upper_multipliers = np.where(at_upper, -stationarity, 0)
        box_stationarity = stationarity - lower_multipliers + upper_multipliers
        box_sign_violation = max(
            0.0, -float(lower_multipliers.min()), -float(upper_multipliers.min())
        )
        box_complementarity = np.r_[
            lower_multipliers[at_lower] * (increment[at_lower] - lower_box[at_lower]),
            upper_multipliers[at_upper] * (upper_box[at_upper] - increment[at_upper]),
        ]
        correction = increment - x0
        primal = 0.5 * float(np.sum(correction**2 / metric))
        negative_dual = float(multipliers @ slack) - primal
        gap = primal + negative_dual
        reconstruction = np.clip(
            x0 + metric * (matrix.T @ multipliers), lower_box, upper_box
        )
        receipt.update(
            multipliers=multipliers.tolist(),
            minimum_dual_multiplier=float(multipliers.min()),
            original_unit_slack=slack.tolist(),
            normalized_slack=normalized_slack.tolist(),
            maximum_original_complementarity=float(np.max(abs(complementarity))),
            maximum_normalized_complementarity=float(
                np.max(abs(normalized_complementarity))
            ),
            primal_objective=primal,
            negative_dual_objective=negative_dual,
            primal_dual_gap=gap,
            normalized_primal_dual_gap=gap / rhs_scale**2,
            lower_active_count=int(at_lower.sum()),
            upper_active_count=int(at_upper.sum()),
            lower_bound_minimum_multiplier=float(lower_multipliers.min()),
            upper_bound_minimum_multiplier=float(upper_multipliers.min()),
            box_multiplier_sign_violation=box_sign_violation,
            interior_stationarity_inf=float(np.max(abs(stationarity[interior])))
            if interior.any()
            else 0.0,
            box_stationarity_inf=float(np.max(abs(box_stationarity))),
            box_complementarity_inf=float(np.max(abs(box_complementarity)))
            if box_complementarity.size
            else 0.0,
            capped_reconstruction_maximum_error=float(
                np.max(abs(increment - reconstruction))
            ),
            box_maximum_violation=float(
                max(0.0, np.max(lower_box - increment), np.max(increment - upper_box))
            ),
        )
        persist()
        assert np.isfinite(increment).all()
        assert multipliers.min() >= 0
        assert slack.min() >= -1e-10
        assert normalized_slack.min() >= -1e-8
        assert np.max(abs(normalized_complementarity)) <= 1e-8
        assert abs(gap / rhs_scale**2) <= 1e-8
        assert box_sign_violation <= 1e-10
        assert np.max(abs(box_stationarity)) <= 1e-10
        np.testing.assert_array_equal(increment, reconstruction)
        assert np.all(increment >= lower_box)
        assert np.all(increment <= upper_box)
        receipt.update(status="certified", certified=True)
        if receipt_path is not None:
            np.savez_compressed(
                receipt_path.with_suffix(".npz"),
                increment=increment,
                dual_multipliers=multipliers,
                lower_bound_multipliers=lower_multipliers,
                upper_bound_multipliers=upper_multipliers,
                box_stationarity=box_stationarity,
                original_slack=slack,
            )
        persist()
        return increment, receipt
    except JointTrustRegionInfeasibleError:
        persist()
        raise
    except (AssertionError, ValueError, RuntimeError, JointProjectionError) as error:
        receipt.update(
            status="unresolved",
            certified=False,
            failure={"type": type(error).__name__, "message": str(error)},
        )
        persist()
        raise
