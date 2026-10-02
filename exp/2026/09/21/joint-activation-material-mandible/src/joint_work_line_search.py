# ruff: noqa: EM101, TRY003
"""Armijo PNCG with resolved energy differences near floating-point cancellation.

For small decreases, integrate the conservative force along the proposed line.
Independent three- and five-point Gauss rules must agree before acceptance.
The physical energy, gradient, Hessian and collision potential are unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from joint_equilibrium import ForwardConvergenceError, StrictLineSearch

RULES = {
    order: tuple(
        (float((x + 1) / 2), float(w / 2))
        for x, w in zip(*np.polynomial.legendre.leggauss(order), strict=True)
    )
    for order in (3, 5)
}


class WorkLineSearch(StrictLineSearch):
    """Use a shifted, directly resolved energy difference for tiny steps."""

    def __call__(  # noqa: C901, PLR0912, PLR0915
        self,
        state: Any,
        problem: Any,
        model_state: Any,
        m: torch.Tensor,
        p: torch.Tensor,
        params: torch.Tensor,
        pHp: torch.Tensor,
    ) -> None:
        f0 = problem.fun(model_state)
        state.f0, state.f_alpha = f0, f0
        alpha = torch.minimum(
            self.line_search_upper(p=p, max_step_norm=self.max_step_norm),
            self.line_search_newton(m=m, pHp=pHp),
        )
        fraction = problem.max_step_size(model_state, alpha * p)
        alpha = alpha * torch.clamp(fraction, 0, 1)
        if not bool(torch.isfinite(alpha)) or float(alpha) <= 0 or float(m) >= 0:
            raise ForwardConvergenceError(
                "invalid precise PNCG direction",
                receipt={"alpha": float(alpha), "slope": float(m)},
            )
        state.ok = False
        work_used = False
        work_error = None
        start = params.detach().clone()
        for step in range(self.max_steps + 1):
            if step:
                alpha = alpha * 0.5
            trial = start + alpha * p
            problem.update(model_state, trial)
            energy = problem.fun(model_state)
            if not bool(torch.isfinite(energy)):
                continue
            predicted = alpha * m
            work_used = abs(float(predicted)) <= 1e-6 * max(abs(float(f0)), 1e-300)
            if work_used:
                integrals = []
                feasible = True
                for order in (3, 5):
                    integral = torch.zeros_like(m)
                    for location, weight in RULES[order]:
                        problem.update(model_state, start + location * alpha * p)
                        if not bool(torch.isfinite(problem.fun(model_state))):
                            feasible = False
                            break
                        integral = integral + weight * torch.dot(
                            problem.grad(model_state), p
                        )
                    if not feasible:
                        break
                    integrals.append(alpha * integral)
                problem.update(model_state, trial)
                if not feasible:
                    continue
                difference = integrals[1]
                error = torch.abs(integrals[1] - integrals[0])
                work_error = float(error)
                resolved = error <= 1e-3 * torch.maximum(
                    torch.abs(difference), torch.abs(predicted) * 1e-12
                )
                decrease = difference + 2 * error <= self.armijo * predicted
                # Shift both scalars together so PNCG damping sees the actual
                # decrease even when f0 + difference would round back to f0.
                state.f0 = torch.zeros_like(f0)
                state.f_alpha = difference
                if bool(resolved and decrease):
                    state.ok = True
            else:
                state.f0, state.f_alpha = f0, energy
                if bool(self.armijo_condition(energy, f0, alpha, m)):
                    state.ok = True
            if state.ok:
                break
        state.alpha, state.step = alpha, step
        receipt = {
            "alpha": float(alpha),
            "slope": float(m),
            "physical_f0": float(f0),
            "physical_f_alpha": float(energy),
            "backtracks": step,
            "stable_work_used": work_used,
            "quadrature_error": work_error,
            "resolved_energy_difference": float(state.f_alpha - state.f0),
        }
        if not all(
            math.isfinite(receipt[key])
            for key in (
                "alpha",
                "slope",
                "physical_f0",
                "physical_f_alpha",
                "resolved_energy_difference",
            )
        ):
            raise ForwardConvergenceError(
                "nonfinite precise Armijo diagnostics", receipt=receipt
            )
        if not hasattr(problem, "work_line_search_records"):
            problem.work_line_search_records = []
            problem.work_line_search_count = 0
        problem.work_line_search_count += int(work_used)
        problem.work_line_search_records.append(receipt)
        problem.work_line_search_records = problem.work_line_search_records[-10:]
        if not state.ok:
            raise ForwardConvergenceError(
                "precise Armijo search exhausted", receipt=receipt
            )
