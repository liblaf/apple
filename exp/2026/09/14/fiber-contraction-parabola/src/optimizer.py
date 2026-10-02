"""Projected limited-memory BFGS with explicit equilibrium-trial rejection."""

from collections.abc import Callable
from typing import Any

import numpy as np
from physics2d import ForwardSolveError
from scipy.optimize import OptimizeResult


def minimize(  # noqa: C901, PLR0912, PLR0915
    evaluate: Callable,
    initial: np.ndarray,
    callback: Callable,
    reject: Callable,
    *,
    nonnegative: bool,
    cfg: Any,
):
    x = initial.copy()
    value, grad = evaluate(x)
    memory = []
    calls = 1
    message = "iteration budget reached"
    success = False
    iteration = 0
    for iteration in range(1, cfg.max_iterations + 1):
        projected = grad.copy()
        if nonnegative:
            projected[(x <= 1e-12) & (grad > 0)] = 0
        if np.linalg.norm(projected, np.inf) <= cfg.gradient_tolerance:
            message, success = "projected gradient tolerance reached", True
            break
        direction = projected.copy()
        alphas = []
        for s, y in reversed(memory):
            alpha = (s @ direction) / (s @ y)
            alphas.append(alpha)
            direction -= alpha * y
        if memory:
            s, y = memory[-1]
            direction *= (s @ y) / (y @ y)
        for (s, y), alpha in zip(memory, reversed(alphas), strict=True):
            direction += s * (alpha - (y @ direction) / (s @ y))
        direction *= -1
        if nonnegative:
            direction[(x <= 1e-12) & (direction < 0)] = 0
        if grad @ direction >= 0:
            # Restart the BFGS approximation when bounds destroy descent.
            memory.clear()
            direction = -projected
        norm = np.linalg.norm(direction, np.inf)
        direction *= min(1.0, cfg.maximum_control_step / max(norm, 1e-30))
        accepted = False
        termination = None
        for backtrack in range(35):
            alpha = 0.5**backtrack
            trial = x + alpha * direction
            if nonnegative:
                trial = np.maximum(trial, 0)
            step = trial - x
            slope = grad @ step
            if np.linalg.norm(step, np.inf) <= cfg.minimum_control_update:
                termination = "control step below resolution floor; not stationary"
                break
            if calls >= cfg.max_evaluations:
                termination = "forward-evaluation budget reached"
                break
            calls += 1
            try:
                trial_value, trial_grad = evaluate(trial)
            except ForwardSolveError as exc:
                reject(iteration, backtrack, alpha, str(exc))
                continue
            if slope < 0 and trial_value <= value + 1e-4 * slope:
                accepted = True
                break
        if not accepted:
            message = (
                termination
                or "outer Armijo search exhausted; last accepted state retained"
            )
            break
        y = trial_grad - grad
        if step @ y > 1e-10 * np.linalg.norm(step) * np.linalg.norm(y):
            memory.append((step.copy(), y.copy()))
            memory = memory[-10:]
        x, value, grad = trial, trial_value, trial_grad
        stop_reason = callback(x)
        if stop_reason is not None:
            message = stop_reason
            break
    return OptimizeResult(
        x=x,
        fun=value,
        jac=grad,
        nfev=calls,
        nit=iteration,
        success=success,
        message=message,
    )
