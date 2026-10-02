"""Undamped PNCG with a permanent stall-triggered sparse Newton handoff."""

from __future__ import annotations

from typing import Any, Literal

import torch
from accelerated_solvers import newton_ccd_fraction, safeguarded_newton
from hybrid_hessian import HybridHessian
from joint_equilibrium import ForwardConvergenceError
from pncg_first import run_pncg_phase


class SparseNewtonProblem:
    """Use one exact unshifted GPU CSR matrix per Newton configuration."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.model = delegate.model
        self.hessian = HybridHessian(self.model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def update(self, state: Any, free: torch.Tensor) -> None:
        self.delegate.update(state, free)
        self.hessian.invalidate()

    def hess_diag(self, state: Any) -> torch.Tensor:
        self.delegate.counts["newton_sparse_diagonal"] += 1
        return self.hessian.diagonal(state)

    def hess_prod(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.delegate.counts["newton_sparse_hvp"] += 1
        return self.hessian.apply(state, direction)

    def max_step_size(self, state: Any, direction: torch.Tensor) -> float:
        return newton_ccd_fraction(self.delegate, state, direction)


def hybrid_first(
    problem: Any,
    state: Any,
    *,
    atol: float,
    max_step_norm: float,
    linear_rtol: float = 1e-3,
    max_newton_steps: int = 100,
    newton_shift_policy: Literal["reset", "reuse"] = "reset",
    reuse_shift_force_ratio: float = 3.0,
    callback: Any = None,
    adaptive_stiffness: Any = None,
) -> tuple[Any, dict]:
    """Run the agreed one-shot PNCG policy, then exact safeguarded Newton."""
    if adaptive_stiffness is not None:
        adaptive_stiffness.initialize(problem, state)
    state, pncg = run_pncg_phase(
        problem,
        state,
        atol=atol,
        max_step_norm=max_step_norm,
        callback=callback,
        adaptive_stiffness=adaptive_stiffness,
    )
    counts_at_handoff = dict(problem.counts)
    sparse = SparseNewtonProblem(problem)
    problem.newton_sparse_problem = sparse
    try:
        state, newton = safeguarded_newton(
            sparse,
            state,
            atol=atol,
            linear_rtol=linear_rtol,
            linear_max_steps=1000,
            max_steps=max_newton_steps,
            max_step_norm=max_step_norm,
            preconditioner="diag",
            shift_policy=newton_shift_policy,
            reuse_shift_force_ratio=reuse_shift_force_ratio,
            shift_scale_policy="signed_mean",
            post_step=(
                lambda current, step: (
                    adaptive_stiffness.after_update(
                        sparse,
                        current,
                        phase="newton",
                        step=step,
                        hessian=sparse.hessian,
                    )
                    if adaptive_stiffness is not None
                    else None
                )
            ),
        )
    except ForwardConvergenceError as error:
        raise ForwardConvergenceError(
            str(error),
            receipt={
                "pncg": pncg,
                "counts_at_handoff": counts_at_handoff,
                "hessian": dict(sparse.hessian.metadata),
                "newton": error.receipt,
                "adaptive_stiffness": (
                    adaptive_stiffness.receipt()
                    if adaptive_stiffness is not None
                    else None
                ),
            },
        ) from error
    return state, {
        "pncg": pncg,
        "counts_at_handoff": counts_at_handoff,
        "hessian": dict(sparse.hessian.metadata),
        "newton_steps": newton["steps"],
        "trace": newton["trace"],
        "adaptive_stiffness": (
            adaptive_stiffness.receipt() if adaptive_stiffness is not None else None
        ),
    }
