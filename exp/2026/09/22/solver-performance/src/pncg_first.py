# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Undamped PNCG-first phase with a force-window Newton handoff signal."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

import torch
from accelerated_solvers import newton_ccd_fraction
from joint_equilibrium import ForwardConvergenceError

from liblaf.apple.solvers.optim.pncg._direction import DirectionUpdate


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ForwardConvergenceError(f"nonfinite PNCG {name}")
    return result


def run_pncg_phase(
    problem: Any,
    state: Any,
    *,
    atol: float,
    max_step_norm: float,
    window_steps: int = 20,
    minimum_reduction: float = 0.1,
    required_poor_windows: int = 2,
    callback: Any = None,
    adaptive_stiffness: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Run unshifted PNCG with per-contribution clamped model curvature.

    The returned operator remains physically unshifted.  This phase does not
    reject an energy increase, damp curvature, or impose a coarse force ratio.
    It only proposes a permanent handoff; the caller owns the later Newton
    phase.
    """
    if atol <= 0 or max_step_norm <= 0 or window_steps <= 0:
        raise ValueError("atol, max_step_norm, and window_steps must be positive")
    if not 0 < minimum_reduction < 1 or required_poor_windows <= 0:
        raise ValueError("invalid force-window policy")
    direction_update = DirectionUpdate()
    trace: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    forces: list[float] = []
    steps = 0
    previous_gradient: torch.Tensor | None = None
    previous_direction: torch.Tensor | None = None
    poor_comparisons = 0

    def observe(kind: str, accepted: dict[str, Any] | None = None) -> float:
        force = _finite_scalar(torch.linalg.vector_norm(problem.grad(state)), "force")
        energy = _finite_scalar(problem.fun(state), "energy")
        row = {"step": steps, "kind": kind, "force": force, "energy": energy}
        if accepted is not None:
            row.update(accepted)
        trace.append(row)
        if callback is not None:
            callback(row)
        return force

    force = observe("initial")
    if force <= atol:
        return state, {
            "reason": "converged",
            "steps": 0,
            "trace": trace,
            "windows": windows,
        }
    while True:
        gradient = problem.grad(state)
        if not torch.isfinite(gradient).all():
            raise ForwardConvergenceError("nonfinite PNCG gradient")
        diagonal = problem.hess_diag(state).abs()
        if not torch.isfinite(diagonal).all() or not bool(torch.all(diagonal > 0)):
            raise ForwardConvergenceError(
                "nonfinite or nonpositive absolute PNCG diagonal"
            )
        preconditioner = diagonal.reciprocal()
        direction = direction_update(
            gradient,
            gradient if previous_gradient is None else previous_gradient,
            preconditioner,
            torch.zeros_like(gradient)
            if previous_direction is None
            else previous_direction,
            restart=previous_gradient is None,
        )
        descent = torch.dot(gradient, direction)
        if not math.isfinite(float(descent)) or float(descent) >= 0:
            direction = -preconditioner * gradient
            descent = torch.dot(gradient, direction)
        if not math.isfinite(float(descent)) or float(descent) >= 0:
            raise ForwardConvergenceError(
                "PNCG direction is not descending after restart"
            )
        curvature = problem.hess_quad(state, direction)
        curvature_value = _finite_scalar(curvature, "curvature")
        if curvature_value <= 0:
            return state, {
                "reason": "nonpositive_curvature",
                "steps": steps,
                "trace": trace,
                "windows": windows,
                "curvature": curvature_value,
            }
        alpha_newton = -float(descent) / curvature_value
        alpha_edge = max_step_norm / float(direction.abs().max())
        alpha0 = min(alpha_newton, alpha_edge)
        if not math.isfinite(alpha0) or alpha0 <= 0:
            raise ForwardConvergenceError("invalid PNCG trial step")
        ccd = float(newton_ccd_fraction(problem, state, alpha0 * direction))
        if not math.isfinite(ccd) or ccd <= 0 or ccd > 1:
            raise ForwardConvergenceError("invalid PNCG CCD fraction")
        free = problem.model.dof_map.to_free(state.u)
        alpha = alpha0 * ccd
        problem.update(state, free + alpha * direction)
        steps += 1
        observation = (
            adaptive_stiffness.after_update(problem, state, phase="pncg", step=steps)
            if adaptive_stiffness is not None
            else None
        )
        if observation is not None and observation["stiffness_changed"]:
            # A kappa update changes the objective. Do not use the former
            # conjugacy pair or force window to judge this new objective.
            previous_gradient = previous_direction = None
            forces.clear()
            poor_comparisons = 0
            windows.append({"end_step": steps, "reset": "adaptive_stiffness"})
        else:
            previous_gradient = gradient.detach().clone()
            previous_direction = direction.detach().clone()
        base_limiter = "newton" if alpha_newton <= alpha_edge else "coordinate_cap"
        force = observe(
            "pncg",
            {
                "alpha_newton": alpha_newton,
                "alpha_edge": alpha_edge,
                "alpha_before_ccd": alpha0,
                "ccd_fraction": ccd,
                "alpha": alpha,
                "coordinate_displacement": float((alpha * direction).abs().max()),
                "slope": float(descent),
                "curvature": curvature_value,
                "limiter_reason": base_limiter if ccd == 1 else f"{base_limiter}+ccd",
                "adaptive_stiffness": observation,
            },
        )
        if force <= atol:
            return state, {
                "reason": "converged",
                "steps": steps,
                "trace": trace,
                "windows": windows,
            }
        forces.append(force)
        if len(forces) % window_steps:
            continue
        current = float(median(forces[-window_steps:]))
        window = {"end_step": steps, "median_force": current}
        if len(forces) < 2 * window_steps:
            window["baseline"] = True
            windows.append(window)
            continue
        previous = float(median(forces[-2 * window_steps : -window_steps]))
        reduction = 1 - current / previous
        poor = reduction < minimum_reduction
        poor_comparisons = poor_comparisons + 1 if poor else 0
        window.update(
            baseline=False,
            previous_median_force=previous,
            fractional_reduction=reduction,
            poor=poor,
            consecutive_poor_comparisons=poor_comparisons,
        )
        windows.append(window)
        if poor_comparisons >= required_poor_windows:
            return state, {
                "reason": "stalled",
                "steps": steps,
                "trace": trace,
                "windows": windows,
            }
