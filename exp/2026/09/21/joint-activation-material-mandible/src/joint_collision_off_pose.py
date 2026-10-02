# ruff: noqa: EM101, TRY003
"""Collision-free, pose-only expression equilibrium.

This runtime is deliberately an exploratory initializer.  It retains the
accepted-force PNCG and implicit-adjoint contracts but removes IPC from the
model energy, gradient, Hessian, and all primal checks.  A caller must restore
the physical collision runtime before any contact-validated joint result is
claimed.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any, override

import torch
from joint_equilibrium import Equilibrium, ForwardConvergenceError, StrictLineSearch
from joint_expression_equilibrium import AcceptedForcePncg

from liblaf.apple.forward._problem import ForwardProblem


class CollisionOffPoseEquilibrium(Equilibrium):
    """Exact-force PNCG without an IPC collision term.

    ``Equilibrium`` supplies the unchanged custom implicit backward and strict
    adjoint solver.  This class only replaces the primal construction so its
    ForwardProblem is the no-contact mechanical energy.
    """

    def __init__(
        self,
        *args: Any,
        line_search_armijo: float,
        hessian_damping_initial: float,
        pncg_restart_interval: int,
        max_step_norm_m: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.line_search_armijo = line_search_armijo
        self.hessian_damping_initial = hessian_damping_initial
        self.pncg_restart_interval = pncg_restart_interval
        self.max_step_norm_m = max_step_norm_m
        self.last_problem: ForwardProblem | None = None

    @staticmethod
    def _contact_receipt() -> dict[str, Any]:
        """State the absence of IPC explicitly; never emulate a contact pass."""
        return {
            "enabled": False,
            "physical_contact_validated": False,
            "contact_numerically_valid": False,
            "minimum_active_distance_m": None,
            "reason": "IPC disabled for exploratory pose-only equilibrium",
        }

    @override
    def primal(
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        forward, model = self.forward, self.forward.model
        assert model.collision is None
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        forward.state.u = (
            model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        forward.state.collision = None
        problem = ForwardProblem(model=model)
        self.last_problem = problem
        initial_force = torch.linalg.vector_norm(problem.grad(forward.state))
        if float(initial_force) <= self.tolerances["atol"]:
            self.forward_count += 1
            self.last_forward = {
                "success": True,
                "method": "strict_pncg_accepted_force_collision_off",
                "seconds": 0.0,
                "steps": 0,
                "grad_norm": float(initial_force),
                "force_threshold": self.tolerances["atol"],
                "result": "initial_equilibrium",
                "rejected_contact_trials": 0,
                "contact": self._contact_receipt(),
                "terminal_gates": {"physical_contact_validated": False},
            }
            return forward.state.u.detach().clone()

        default = forward.default_optimizer(
            max_steps=self.tolerances["max_steps"],
            rtol=0.0,
            atol=self.tolerances["atol"],
        )
        optimizer = AcceptedForcePncg(
            criteria=default.criteria,
            hess_damping=AcceptedForcePncg.HessianDamping(
                initial=self.hessian_damping_initial
            ),
            line_search=StrictLineSearch(
                armijo=self.line_search_armijo,
                max_steps=60,
                max_step_norm=self.max_step_norm_m,
            ),
        )
        optimizer.restart_interval = self.pncg_restart_interval
        forward.problem, forward.optimizer = problem, optimizer
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                solution = optimizer.minimize(problem, forward.state, forward.free)
        except ForwardConvergenceError as error:
            self.last_forward = {
                "success": False,
                "failure": str(error),
                "receipt": error.receipt,
                "contact": self._contact_receipt(),
            }
            raise
        exact_force = torch.linalg.vector_norm(problem.grad(forward.state))
        self.forward_count += 1
        self.last_forward = {
            "success": bool(solution.success)
            and float(exact_force) <= self.tolerances["atol"],
            "method": "strict_pncg_accepted_force_collision_off",
            "seconds": time.perf_counter() - started,
            "steps": int(solution.state.convergence_state.step),
            "grad_norm": float(exact_force),
            "force_threshold": self.tolerances["atol"],
            "result": str(solution.result),
            "rejected_contact_trials": 0,
            "contact": self._contact_receipt(),
            "terminal_gates": {"physical_contact_validated": False},
        }
        if not self.last_forward["success"]:
            raise ForwardConvergenceError(
                "collision-off expression PNCG did not meet accepted-state force tolerance",
                receipt=self.last_forward,
            )
        return forward.state.u.detach().clone()


def install_collision_off_pose_runtime(
    physics: Any,
    *,
    line_search_armijo: float = 0.25,
    hessian_damping_initial: float = 0.001,
    pncg_restart_interval: int = 200,
    max_step_norm_m: float = 0.0005,
) -> CollisionOffPoseEquilibrium:
    """Install the explicit no-IPC pose runtime on an eye-inclusive physics object.

    The original collision object is intentionally not destroyed; callers that
    require physical contact must retain and restore it before their next solve.
    """
    if not 0 < line_search_armijo < 1:
        raise ValueError("line_search_armijo must lie in (0, 1)")
    if not hessian_damping_initial > 0 or not max_step_norm_m > 0:
        raise ValueError("damping and maximum step norm must be positive")
    if pncg_restart_interval <= 0:
        raise ValueError("PNCG restart interval must be positive")
    old = physics.runtime
    model = old.forward.model
    assert model.collision is not None
    collision = model.collision
    model.collision = None
    old.forward.state.collision = None
    runtime = CollisionOffPoseEquilibrium(
        old.forward,
        rtol=0.0,
        atol=old.tolerances["atol"],
        adjoint_rtol=old.tolerances["adjoint_rtol"],
        max_steps=old.tolerances["max_steps"],
        forward_method="pncg",
        newton_linear_rtol=old.forward_solver["newton_linear_rtol"],
        newton_max_steps=old.forward_solver["newton_max_steps"],
        line_search_armijo=line_search_armijo,
        hessian_damping_initial=hessian_damping_initial,
        pncg_restart_interval=pncg_restart_interval,
        max_step_norm_m=max_step_norm_m,
    )
    # Preserve the object for an explicit caller-managed physical-runtime restore.
    runtime.disabled_collision = collision
    physics.runtime = runtime
    return runtime


__all__ = ["CollisionOffPoseEquilibrium", "install_collision_off_pose_runtime"]
