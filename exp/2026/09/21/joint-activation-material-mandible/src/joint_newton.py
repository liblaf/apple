"""Owned-state inexact Newton-CG refinement for contact equilibrium."""

from __future__ import annotations

import math
import time
from typing import Any

import attrs
import torch

from liblaf.apple.solvers.linalg.cupy import CupyCG


class NewtonCgError(RuntimeError):
    """Numerical rejection with a machine-readable Newton-CG receipt."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


@attrs.define
class NewtonSystem:
    """Exact free-DOF Hessian operator with diagonal preconditioning only."""

    def _default_preconditioner(self) -> torch.Tensor:
        diagonal_full = self.model.hess_diag(self.model_state)
        diagonal = self.model.dof_map.to_free_hess_diag(diagonal_full).abs()
        return diagonal.reciprocal()

    b: torch.Tensor
    model: Any
    model_state: Any
    _preconditioner: torch.Tensor = attrs.field(
        default=attrs.Factory(_default_preconditioner, takes_self=True)
    )
    matvec_count: int = 0

    def matvec(self, direction: torch.Tensor) -> torch.Tensor:
        self.matvec_count += 1
        direction_full = self.model.dof_map.to_full_grad(direction)
        product_full = self.model.hess_prod(self.model_state, direction_full)
        return self.model.dof_map.to_free_grad(product_full)

    def rmatvec(self, direction: torch.Tensor) -> torch.Tensor:
        return self.matvec(direction)

    def precondition(self, value: torch.Tensor) -> torch.Tensor:
        return self._preconditioner * value

    def rprecondition(self, value: torch.Tensor) -> torch.Tensor:
        return self.precondition(value)

    def preconditioner(self, value: torch.Tensor) -> torch.Tensor:
        return self.precondition(value)

    def rpreconditioner(self, value: torch.Tensor) -> torch.Tensor:
        return self.rprecondition(value)


def _fresh_state(model: Any, free: torch.Tensor) -> Any:
    state = model.State(u=model.dof_map.to_full(free).detach().clone())
    if model.collision is not None:
        state.collision = model.collision.state_at(state.u)
    return state


def _inherit_contact_history(
    previous: Any, candidate: Any, ccd_fraction: float
) -> None:
    if previous.collision is None:
        return
    assert candidate.collision is not None
    candidate.collision.minimum_ccd_fraction = min(
        previous.collision.minimum_ccd_fraction, ccd_fraction
    )
    candidate.collision.boundary_ccd_fraction = previous.collision.boundary_ccd_fraction


def refine_newton_cg(  # noqa: C901, PLR0915
    forward: Any,
    *,
    rtol: float,
    atol: float,
    linear_rtol: float,
    linear_max_iterations: int,
    max_newton_steps: int,
    armijo_coefficient: float = 1e-4,
    max_line_search_steps: int = 40,
) -> dict[str, Any]:
    """Refine ``forward.state`` in place or fail without a solver fallback."""
    assert 0 < rtol < 1
    assert atol >= 0
    assert 0 < linear_rtol < 1
    assert linear_max_iterations > 0
    assert max_newton_steps > 0
    assert 0 < armijo_coefficient < 1
    assert max_line_search_steps > 0

    model = forward.model
    problem = forward.problem
    state = forward.state
    free = model.dof_map.to_free(state.u).detach().clone()
    initial_gradient = problem.grad(state)
    initial_gradient_norm = float(torch.linalg.vector_norm(initial_gradient))
    assert math.isfinite(initial_gradient_norm)
    force_threshold = max(atol, rtol * initial_gradient_norm)
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []

    def fail(stage: str, message: str, **details: Any) -> None:
        raise NewtonCgError(
            message,
            {
                "success": False,
                "method": "inexact_newton_cg",
                "stage": stage,
                "message": message,
                "initial_gradient_norm": initial_gradient_norm,
                "force_threshold": force_threshold,
                "elapsed_seconds": time.perf_counter() - started,
                "trace": trace,
                **details,
            },
        )

    for iteration in range(max_newton_steps):
        gradient = problem.grad(state)
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        if not math.isfinite(gradient_norm):
            fail("force", "non-finite Newton-CG free-force norm")
        if gradient_norm <= force_threshold:
            break
        energy = float(problem.fun(state))
        if not math.isfinite(energy):
            fail("energy", "non-finite Newton-CG energy")

        system = NewtonSystem(b=-gradient, model=model, model_state=state)
        linear_started = time.perf_counter()
        solution = CupyCG(
            maxiter=linear_max_iterations,
            rtol=linear_rtol,
            atol=0.0,
        ).solve(system, torch.zeros_like(gradient))
        torch.cuda.synchronize()
        linear_seconds = time.perf_counter() - linear_started
        direction = solution.params.detach()
        residual = float(torch.linalg.vector_norm(system.matvec(direction) - system.b))
        relative_residual = residual / gradient_norm
        slope = float(torch.dot(gradient, direction))
        linear_valid = bool(
            solution.success
            and torch.isfinite(direction).all()
            and math.isfinite(relative_residual)
            and relative_residual <= linear_rtol * 1.05
            and math.isfinite(slope)
            and slope < 0
        )
        if not linear_valid:
            fail(
                "linear_solve_or_descent",
                "Newton-CG linear solve failed validation",
                linear_result=str(solution.result),
                linear_relative_residual=relative_residual,
                slope=slope,
            )

        ccd_fraction = float(problem.max_step_size(state, direction))
        if not math.isfinite(ccd_fraction) or not 0 < ccd_fraction <= 1:
            fail(
                "ccd",
                "Newton-CG contact CCD returned invalid fraction",
                ccd_fraction=ccd_fraction,
            )
        alpha = 1.0 if ccd_fraction == 1.0 else 0.99 * ccd_fraction
        accepted = False
        trials: list[dict[str, Any]] = []
        for line_step in range(max_line_search_steps + 1):
            candidate_free = free + alpha * direction
            candidate = _fresh_state(model, candidate_free)
            _inherit_contact_history(state, candidate, ccd_fraction)
            candidate_energy = float(problem.fun(candidate))
            armijo_rhs = energy + armijo_coefficient * alpha * slope
            finite = math.isfinite(candidate_energy)
            trials.append(
                {
                    "line_step": line_step,
                    "alpha": alpha,
                    "energy": candidate_energy,
                    "armijo_rhs": armijo_rhs,
                    "finite": finite,
                }
            )
            if finite and candidate_energy <= armijo_rhs:
                free = candidate_free.detach().clone()
                state = candidate
                accepted = True
                break
            alpha *= 0.5
        trace.append(
            {
                "iteration": iteration,
                "energy": energy,
                "gradient_norm": gradient_norm,
                "force_threshold": force_threshold,
                "linear_result": str(solution.result),
                "linear_relative_residual": relative_residual,
                "linear_seconds": linear_seconds,
                "linear_matvec_count": system.matvec_count,
                "slope": slope,
                "ccd_fraction": ccd_fraction,
                "accepted": accepted,
                "accepted_alpha": alpha if accepted else None,
                "line_search_steps": line_step,
                "trials": trials,
            }
        )
        if not accepted:
            fail(
                "armijo",
                "Newton-CG Armijo line search exhausted",
                iteration=iteration,
                trials=trials,
            )

    terminal_gradient = problem.grad(state)
    terminal_gradient_norm = float(torch.linalg.vector_norm(terminal_gradient))
    converged = bool(
        math.isfinite(terminal_gradient_norm)
        and terminal_gradient_norm <= force_threshold
    )
    if not converged:
        fail(
            "nonlinear_step_budget",
            "Newton-CG exhausted its nonlinear step budget",
            terminal_gradient_norm=terminal_gradient_norm,
        )
    forward.state = state
    torch.cuda.synchronize()
    return {
        "success": True,
        "method": "inexact_newton_cg",
        "seconds": time.perf_counter() - started,
        "steps": len(trace),
        "grad_norm": terminal_gradient_norm,
        "result": "primary_success",
        "initial_gradient_norm": initial_gradient_norm,
        "force_threshold": force_threshold,
        "linear_solver": {
            "implementation": "CupyCG",
            "rtol": linear_rtol,
            "atol": 0.0,
            "max_iterations": linear_max_iterations,
            "failure_policy": "fail visibly; no solver fallback",
        },
        "armijo": {
            "coefficient": armijo_coefficient,
            "max_steps": max_line_search_steps,
        },
        "trace": trace,
    }
