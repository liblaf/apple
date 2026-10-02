# Copyright (c) 2026 liblaf
# ruff: noqa: EM101, TRY003, TRY301
"""Collision-off hybrid equilibrium for the two-expression inverse experiment.

The physical model has no collision object.  The shared hybrid code still calls
``max_step_size`` through a historically named CCD helper; this wrapper returns
the unconstrained unit fraction without querying any collision geometry.
"""

from __future__ import annotations

import copy
import time
from typing import Any, override

import optree
import torch
from accelerated_solvers import CachedProblem, safeguarded_newton
from hybrid_first_solver import SparseNewtonProblem
from joint_equilibrium import Equilibrium, ForwardConvergenceError
from mouthopen_runtime import SparseAdjointSolver, _ShiftedImplicit
from pncg_first import run_pncg_phase

from liblaf.apple.forward._problem import ForwardProblem


class _CollisionOffProblem(ForwardProblem):
    """Physical FEM problem with an unconditional unit trial fraction."""

    @override
    def max_step_size(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        assert self.model.collision is None
        assert state.collision is None
        return torch.ones((), device=direction.device, dtype=direction.dtype)


class CollisionOffHybridEquilibrium(Equilibrium):
    """PNCG then exact sparse Newton at the declared physical force tolerance."""

    def __init__(
        self,
        *args: Any,
        max_step_norm_m: float,
        newton_linear_rtol: float = 1e-3,
        newton_max_steps: int = 100,
        adjoint_relative_shift: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            forward_method="pncg",
            newton_linear_rtol=newton_linear_rtol,
            newton_max_steps=newton_max_steps,
            **kwargs,
        )
        assert self.forward.model.collision is None
        assert max_step_norm_m > 0
        assert adjoint_relative_shift >= 0
        self.max_step_norm_m = max_step_norm_m
        self.adjoint_relative_shift = adjoint_relative_shift
        self.last_problem: CachedProblem | None = None
        self.last_sparse_problem: SparseNewtonProblem | None = None
        self.last_sparse_adjoint: dict[str, Any] = {}
        self.deadline: float | None = None

    @override
    def solve(
        self,
        materials: dict,
        fixed_values: torch.Tensor,
        seed: torch.Tensor,
        *,
        key: str,
    ) -> torch.Tensor:
        assert self.forward.model.collision is None
        if self.adjoint_relative_shift == 0:
            return super().solve(materials, fixed_values, seed, key=key)
        leaves, spec = optree.tree_flatten(materials)
        return _ShiftedImplicit.apply(self, key, spec, fixed_values, seed, *leaves)

    def drop_warm_adjoint(self, key: str) -> None:
        self.warm_adjoints.pop(key, None)

    @override
    def primal(
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        self.last_forward = {}
        self.last_problem = None
        self.last_sparse_problem = None
        model = self.forward.model
        assert model.collision is None
        assert fixed.shape == model.dof_map.fixed_values.shape
        assert seed.shape == self.forward.state.u.shape
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        state = self.forward.state
        state.u = model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        state.collision = None
        remaining = (
            None if self.deadline is None else self.deadline - time.perf_counter()
        )
        if remaining is not None and remaining <= 0:
            raise ForwardConvergenceError("declared forward wall budget exhausted")
        problem = CachedProblem(
            _CollisionOffProblem(model=model),
            exact_curvature=False,
            wall_seconds=remaining,
        )
        self.last_problem = problem
        started = time.perf_counter()
        try:
            state, pncg = run_pncg_phase(
                problem,
                state,
                atol=self.tolerances["atol"],
                max_step_norm=self.max_step_norm_m,
            )
            if pncg["reason"] != "converged":
                sparse = SparseNewtonProblem(problem)
                self.last_sparse_problem = sparse
                state, newton = safeguarded_newton(
                    sparse,
                    state,
                    atol=self.tolerances["atol"],
                    linear_rtol=self.newton_linear_rtol,
                    linear_max_steps=1000,
                    max_steps=self.newton_max_steps,
                    max_step_norm=self.max_step_norm_m,
                    preconditioner="diag",
                    shift_policy="reuse",
                    reuse_shift_force_ratio=0.0,
                    shift_scale_policy="signed_mean",
                )
            else:
                newton = {"steps": 0, "trace": []}
            terminal_force = float(torch.linalg.vector_norm(problem.grad(state)))
            success = terminal_force <= self.tolerances["atol"]
            self.last_forward = {
                "success": success,
                "method": "collision-off-pncg-sparse-newton",
                "collision_enabled": False,
                "ccd_enabled": False,
                "seconds": time.perf_counter() - started,
                "grad_norm": terminal_force,
                "force_threshold": self.tolerances["atol"],
                "pncg": pncg,
                "newton": newton,
                "terminal_gates": {"force": success},
            }
            if not success:
                raise ForwardConvergenceError(
                    "collision-off hybrid force gate failed", receipt=self.last_forward
                )
            self.forward_count += 1
            return state.u.detach().clone()
        except ForwardConvergenceError as error:
            if not self.last_forward:
                self.last_forward = {
                    "success": False,
                    "collision_enabled": False,
                    "ccd_enabled": False,
                    "failure": str(error),
                    "receipt": error.receipt,
                }
            energy = model.fun(state)
            self.last_forward["failure_state"] = {
                "physical_energy": float(energy)
                if bool(torch.isfinite(energy))
                else None,
                "collision_state_is_none": state.collision is None,
            }
            self.last_failed_displacement = state.u.detach().clone()
            error.receipt = copy.deepcopy(self.last_forward)
            raise


def install_collision_off_runtime(
    physics: Any,
    *,
    forward_atol: float = 1e-8,
    adjoint_rtol: float = 1e-7,
    max_steps: int = 5000,
    max_step_norm_m: float,
    newton_linear_rtol: float = 1e-3,
    newton_max_steps: int = 100,
    adjoint_relative_shift: float = 0.0,
) -> CollisionOffHybridEquilibrium:
    """Remove collision from the actual model and install the hybrid solver."""
    assert 0 < forward_atol <= 1e-8
    assert 0 < adjoint_rtol < 1
    assert max_steps > 0
    assert adjoint_relative_shift >= 0
    old = physics.runtime
    old.forward.model.collision = None
    old.forward.state.collision = None
    runtime = CollisionOffHybridEquilibrium(
        old.forward,
        rtol=0.0,
        atol=forward_atol,
        adjoint_rtol=adjoint_rtol,
        max_steps=max_steps,
        max_step_norm_m=max_step_norm_m,
        newton_linear_rtol=newton_linear_rtol,
        newton_max_steps=newton_max_steps,
        adjoint_relative_shift=adjoint_relative_shift,
    )
    runtime.solver = SparseAdjointSolver(runtime.solver, runtime)
    physics.runtime = runtime
    return runtime


install_mouthopen_hybrid_runtime = install_collision_off_runtime


__all__ = [
    "CollisionOffHybridEquilibrium",
    "install_collision_off_runtime",
    "install_mouthopen_hybrid_runtime",
]
