"""One-way accepted-state PNCG warmup for the hybrid primal solver."""

from __future__ import annotations

import math
from typing import Any

import torch
from joint_equilibrium import ForwardConvergenceError, StrictLineSearch
from joint_expression_equilibrium import AcceptedForcePncg


def run_pncg_warmup(
    problem: Any,
    state: Any,
    *,
    initial_force: float,
    atol: float,
    make_default_optimizer: Any,
    hessian_damping_initial: float,
    line_search_armijo: float,
    max_step_norm: float,
    pncg_restart_interval: int,
    newton_switch_atol: float = 0.0,
    max_pncg_steps: int = 100,
    window_steps: int = 20,
    minimum_reduction: float = 0.1,
    required_poor_windows: int = 2,
) -> tuple[Any, dict[str, Any]]:
    """Warm up with PNCG, then make one irreversible handoff to Newton.

    PNCG failures remain failures.  In particular, this helper does not retry
    PNCG after a Newton step and does not treat a coarse-force threshold as a
    physical convergence claim.
    """
    from adaptive_pncg import ForceWindows

    assert math.isfinite(initial_force) and initial_force >= 0
    assert atol > 0
    assert math.isfinite(newton_switch_atol) and newton_switch_atol >= 0
    assert max_pncg_steps > 0
    coarse_threshold = max(atol, initial_force * 1e-3, newton_switch_atol)
    windows: list[dict[str, Any]] = []
    coarse_steps = 0
    coarse_force = initial_force

    def receipt(reason: str) -> dict[str, Any]:
        return {
            "handoff_reason": reason,
            "initial_force": initial_force,
            "coarse_steps": coarse_steps,
            "coarse_terminal_force": coarse_force,
            "coarse_threshold": coarse_threshold,
            "newton_switch_atol": newton_switch_atol,
            "max_pncg_steps": max_pncg_steps,
            "window_steps": window_steps,
            "minimum_reduction": minimum_reduction,
            "required_poor_windows": required_poor_windows,
            "windows": windows,
        }

    if coarse_force <= coarse_threshold:
        return state, receipt("force_threshold")

    default = make_default_optimizer(coarse_threshold)
    optimizer = AcceptedForcePncg(
        criteria=default.criteria,
        hess_damping=AcceptedForcePncg.HessianDamping(initial=hessian_damping_initial),
        line_search=StrictLineSearch(
            armijo=line_search_armijo,
            max_steps=60,
            max_step_norm=max_step_norm,
        ),
    )
    optimizer.restart_interval = pncg_restart_interval
    opt_state = optimizer.init(problem, state, problem.model.dof_map.to_free(state.u))
    detector = ForceWindows(
        initial_force,
        window_steps=window_steps,
        minimum_reduction=minimum_reduction,
        required_poor_windows=required_poor_windows,
    )
    while coarse_steps < max_pncg_steps:
        problem.check_budget()
        optimizer.step(problem, state, opt_state)
        terminated, result = optimizer.terminate(problem, state, opt_state)
        coarse_steps += 1
        coarse_force = float(torch.linalg.vector_norm(problem.grad(state)))
        if not math.isfinite(coarse_force):
            raise ForwardConvergenceError("nonfinite accepted-state PNCG force")
        if coarse_force <= coarse_threshold:
            return state, receipt("force_threshold")
        trigger, window = detector.observe(coarse_force)
        if window is not None:
            window["pncg_steps"] = coarse_steps
            windows.append(window)
        if terminated:
            raise ForwardConvergenceError(
                f"PNCG warmup stopped above force threshold: {result}"
            )
        if trigger:
            return state, receipt("stalled")
    return state, receipt("pncg_step_cap")
