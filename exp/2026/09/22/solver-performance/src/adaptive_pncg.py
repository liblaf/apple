# ruff: noqa: C901, EM101, EM102, PT018, TRY003, TRY301, PLR0915
"""PNCG with one safeguarded Newton correction after sustained force stagnation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from joint_equilibrium import ForwardConvergenceError, StrictLineSearch
from joint_expression_equilibrium import AcceptedForcePncg

LOG = logging.getLogger(__name__)


@dataclass
class ForceWindows:
    """Compare best-so-far force at consecutive window boundaries within a segment."""

    best: float
    window_steps: int = 100
    minimum_reduction: float = 0.1
    required_poor_windows: int = 2
    count: int = 0
    poor_windows: int = 0
    start_best: float | None = None

    def __post_init__(self) -> None:
        assert math.isfinite(self.best) and self.best > 0
        assert self.window_steps > 0 and self.required_poor_windows > 0
        assert 0 < self.minimum_reduction < 1
        self.start_best = self.best

    def observe(self, force: float) -> tuple[bool, dict | None]:
        assert math.isfinite(force) and force >= 0
        self.best = min(self.best, force)
        self.count += 1
        if self.count % self.window_steps:
            return False, None
        assert self.start_best is not None and self.start_best > 0
        improvement = 1 - self.best / self.start_best
        poor = improvement < self.minimum_reduction
        self.poor_windows = self.poor_windows + 1 if poor else 0
        receipt = {
            "segment_pncg_steps": self.count,
            "start_best_force": self.start_best,
            "end_best_force": self.best,
            "fractional_improvement": improvement,
            "poor": poor,
            "consecutive_poor_windows": self.poor_windows,
        }
        self.start_best = self.best
        return self.poor_windows >= self.required_poor_windows, receipt


def adaptive_pncg_newton(
    problem: Any,
    state: Any,
    *,
    atol: float,
    make_default_optimizer: Any,
    hessian_damping_initial: float,
    line_search_armijo: float,
    max_step_norm: float,
    pncg_restart_interval: int,
    linear_rtol: float,
    max_newton_steps: int,
    newton_max_step_norm: float,
    max_pncg_steps: int,
    window_steps: int = 100,
    minimum_reduction: float = 0.1,
    required_poor_windows: int = 2,
    callback: Any = None,
) -> tuple[Any, dict]:
    from accelerated_solvers import safeguarded_newton_step

    assert atol > 0 and max_pncg_steps > 0 and max_newton_steps > 0
    started = time.perf_counter()
    trace: list[dict] = []
    windows: list[dict] = []
    corrections: list[dict] = []
    pncg_steps = 0
    problem_callback = getattr(problem, "callback", None)

    def sample(kind: str, energy: float) -> float:
        force = float(torch.linalg.vector_norm(problem.grad(state)))
        if not math.isfinite(force) or not math.isfinite(energy):
            raise ForwardConvergenceError("nonfinite accepted-state diagnostic")
        row = {
            "accepted_step": pncg_steps + len(corrections),
            "pncg_steps": pncg_steps,
            "newton_steps": len(corrections),
            "kind": kind,
            "seconds": time.perf_counter() - started,
            "force": force,
            "energy": energy,
        }
        trace.append(row)
        if callback is not None:
            callback(row)
        return force

    def start_pncg() -> tuple[Any, Any]:
        default = make_default_optimizer(atol)
        optimizer = AcceptedForcePncg(
            criteria=default.criteria,
            hess_damping=AcceptedForcePncg.HessianDamping(
                initial=hessian_damping_initial
            ),
            line_search=StrictLineSearch(
                armijo=line_search_armijo,
                max_steps=60,
                max_step_norm=max_step_norm,
            ),
        )
        optimizer.restart_interval = pncg_restart_interval
        opt_state = optimizer.init(
            problem, state, problem.model.dof_map.to_free(state.u)
        )
        return optimizer, opt_state

    def receipt() -> dict:
        return {
            "steps": pncg_steps + len(corrections),
            "pncg_steps": pncg_steps,
            "newton_steps": len(corrections),
            "trace": trace,
            "windows": windows,
            "corrections": corrections,
            "last_observed_accepted_force": trace[-1]["force"] if trace else None,
        }

    try:
        force = sample("initial", float(problem.fun(state)))
        if force <= atol:
            return state, receipt()
        optimizer, opt_state = start_pncg()
        detector = ForceWindows(
            force, window_steps, minimum_reduction, required_poor_windows
        )
        while pncg_steps < max_pncg_steps:
            problem.check_budget()
            optimizer.step(problem, state, opt_state)
            if problem_callback is not None:
                problem_callback(state, opt_state)
            terminated, result = optimizer.terminate(problem, state, opt_state)
            pncg_steps += 1
            force = sample("pncg", float(opt_state.fun))
            if force <= atol:
                return state, receipt()
            if terminated:
                raise ForwardConvergenceError(
                    f"PNCG segment stopped above force tolerance: {result}"
                )
            trigger, window = detector.observe(force)
            if window is not None:
                window.update(pncg_steps=pncg_steps, newton_steps=len(corrections))
                windows.append(window)
                LOG.info(
                    "PNCG window %d force %.6g best %.6g poor %d",
                    pncg_steps,
                    force,
                    detector.best,
                    detector.poor_windows,
                )
            if not trigger:
                continue
            if len(corrections) >= max_newton_steps:
                raise ForwardConvergenceError(
                    "adaptive Newton correction budget exhausted"
                )
            correction_started = time.perf_counter()
            state, correction = safeguarded_newton_step(
                problem,
                state,
                linear_rtol=linear_rtol,
                max_step_norm=newton_max_step_norm,
                preconditioner="diag",
            )
            correction.update(
                pncg_steps=pncg_steps,
                seconds=time.perf_counter() - correction_started,
                triggering_windows=windows[-required_poor_windows:],
            )
            corrections.append(correction)
            force = sample("newton", float(problem.fun(state)))
            LOG.info(
                "Newton correction %d after PNCG %d: force %.6g",
                len(corrections),
                pncg_steps,
                force,
            )
            if force <= atol:
                return state, receipt()
            optimizer, opt_state = start_pncg()
            detector = ForceWindows(
                force, window_steps, minimum_reduction, required_poor_windows
            )
        raise ForwardConvergenceError("global PNCG iteration budget exhausted")
    except ForwardConvergenceError as error:
        details = receipt()
        details["interrupted_operation"] = error.receipt
        raise ForwardConvergenceError(str(error), receipt=details) from error
