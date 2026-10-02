# ruff: noqa: EM101, TRY003
"""Differentiable fixed-material expression equilibria with rigid eyes.

This is deliberately separate from the historical equilibrium class: its primal
PNCG uses the accepted-state force criterion and the proven conservative CCD
policy from the eye-neutral forward solve.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, override

import ipctk
import numpy as np
import torch
from joint_equilibrium import (
    Equilibrium,
    ForwardConvergenceError,
    StrictPncg,
)
from joint_work_line_search import WorkLineSearch

from liblaf.apple.forward._problem import ForwardProblem

LOG = logging.getLogger(__name__)


class AcceptedForcePncg(StrictPncg):
    """Stop only when the gradient at the accepted configuration is small."""

    @override
    def terminate(self, problem: Any, model_state: Any, opt_state: Any) -> Any:
        gradient = problem.grad(model_state)
        opt_state.convergence_state.grad_norm = torch.linalg.vector_norm(gradient)
        return super().terminate(problem, model_state, opt_state)

    @override
    def step(self, problem: Any, model_state: Any, opt_state: Any) -> None:
        interval = getattr(self, "restart_interval", 0)
        if interval and opt_state.step % interval == 0:
            opt_state.line_search_state.ok = False
        super().step(problem, model_state, opt_state)
        if opt_state.step and opt_state.step % 500 == 0:
            LOG.info(
                "Expression PNCG accepted step %d, force %.6g",
                opt_state.step,
                float(opt_state.convergence_state.grad_norm),
            )


class FeasibleExpressionProblem(ForwardProblem):
    """CCD-safe Armijo objective without changing the physical IPC potential."""

    def __init__(self, *args: Any, collision_step_safety: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.collision_step_safety = collision_step_safety
        self.rejected_contact_trials = 0

    @override
    def max_step_size(self, state: Any, p: torch.Tensor) -> torch.Tensor:
        fraction = super().max_step_size(state, p)
        if float(fraction) < 1.0:
            fraction = fraction * self.collision_step_safety
        return fraction

    @override
    def hess_quad(self, state: Any, p: torch.Tensor) -> torch.Tensor:
        return torch.dot(p, self.hess_prod(state, p))

    @override
    def callback(self, model_state: Any, opt_state: Any) -> None:
        if not hasattr(self, "diagnostic_directory") or opt_state.step % 50:
            return
        directory = self.diagnostic_directory
        directory.mkdir(parents=True, exist_ok=True)
        row = {
            "step": int(opt_state.step),
            "unix_time": time.time(),
            "force_code": float(torch.linalg.vector_norm(self.grad(model_state))),
            "line_search": self.work_line_search_records[-1],
        }
        with (directory / "trace.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if opt_state.step % 500 == 0:
            checkpoint = directory / "accepted-latest.npz"
            temporary = directory / "accepted-latest.tmp.npz"
            np.savez(temporary, full_displacement_m=model_state.u.numpy(force=True))
            temporary.replace(checkpoint)

    @override
    def fun(self, state: Any) -> torch.Tensor:
        value = super().fun(state)
        collision = self.model.collision
        assert collision is not None
        assert state.collision is not None
        if len(state.collision.collisions) == 0:
            return value
        points = (collision.vertices + state.u[collision.indices]).numpy(force=True)
        distance_sq = state.collision.collisions.compute_minimum_distance(
            collision.collision_mesh, points
        )
        if distance_sq <= collision.min_distance**2:
            self.rejected_contact_trials += 1
            return torch.full_like(value, torch.inf)
        return value


class ExpressionEquilibrium(Equilibrium):
    """Exact-force PNCG primal compatible with the inherited implicit adjoint."""

    def __init__(
        self,
        *args: Any,
        line_search_armijo: float,
        hessian_damping_initial: float,
        pncg_restart_interval: int,
        max_step_norm_m: float,
        collision_step_safety: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 < collision_step_safety < 1:
            raise ValueError("collision_step_safety must lie in (0,1)")
        self.line_search_armijo = line_search_armijo
        self.hessian_damping_initial = hessian_damping_initial
        self.pncg_restart_interval = pncg_restart_interval
        self.max_step_norm_m = max_step_norm_m
        self.collision_step_safety = collision_step_safety
        self.last_problem: FeasibleExpressionProblem | None = None

    @override
    def primal(  # noqa: PLR0915
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        forward, model = self.forward, self.forward.model
        collision = model.collision
        assert collision is not None
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        forward.state.u = (
            model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        prior = collision.state_at(seed.detach())
        boundary_change = forward.state.u - seed.detach()
        boundary_fraction = float(
            collision.max_step_size(prior, seed.detach(), boundary_change)
        )
        if boundary_fraction < 1.0:
            self.last_forward = {
                "success": False,
                "failure": "Dirichlet boundary proposal fails CCD",
                "contact": {"ccd_boundary_fraction": boundary_fraction},
            }
            raise ForwardConvergenceError(
                "Dirichlet boundary proposal fails contact CCD",
                receipt=self.last_forward,
            )
        forward.state.collision = collision.state_at(forward.state.u)
        forward.state.collision.boundary_ccd_fraction = boundary_fraction
        receipt = collision.diagnostics(forward.state.collision, forward.state.u)
        if not receipt["contact_numerically_valid"]:
            raise ForwardConvergenceError(
                "expression seed violates contact feasibility", receipt=receipt
            )
        problem = FeasibleExpressionProblem(
            model=model, collision_step_safety=self.collision_step_safety
        )
        if hasattr(self, "diagnostic_directory"):
            problem.diagnostic_directory = Path(self.diagnostic_directory)
        initial_force = torch.linalg.vector_norm(problem.grad(forward.state))
        if float(initial_force) <= self.tolerances["atol"]:
            positions = (collision.vertices + forward.state.u[collision.indices]).numpy(
                force=True
            )
            intersects = bool(
                ipctk.has_intersections(
                    collision.collision_mesh, positions, ipctk.LBVH()
                )
            )
            min_gap = receipt["minimum_active_distance_m"]
            strict_gap = min_gap is None or min_gap >= collision.min_distance
            if intersects or not strict_gap:
                raise ForwardConvergenceError(
                    "initial expression collision gate failed", receipt=receipt
                )
            self.forward_count += 1
            self.last_problem = problem
            self.last_forward = {
                "success": True,
                "method": "strict_pncg_exact_curvature_resolved_work",
                "seconds": 0.0,
                "steps": 0,
                "grad_norm": float(initial_force),
                "force_threshold": self.tolerances["atol"],
                "result": "initial_equilibrium",
                "rejected_contact_trials": 0,
                "contact": receipt,
            }
            return forward.state.u.detach().clone()
        default = forward.default_optimizer(
            max_steps=self.tolerances["max_steps"],
            rtol=0.0,
            atol=self.tolerances["atol"],
        )
        optimizer = AcceptedForcePncg(
            criteria=default.criteria,
            hess_damping=StrictPncg.HessianDamping(
                initial=self.hessian_damping_initial
            ),
            line_search=WorkLineSearch(
                armijo=self.line_search_armijo,
                max_steps=60,
                max_step_norm=self.max_step_norm_m,
            ),
        )
        optimizer.restart_interval = self.pncg_restart_interval
        forward.problem, forward.optimizer = problem, optimizer
        self.last_problem = problem
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                solution = optimizer.minimize(problem, forward.state, forward.free)
        except ForwardConvergenceError as error:
            self.last_forward = {
                "success": False,
                "failure": str(error),
                "receipt": error.receipt,
            }
            raise
        exact_force = torch.linalg.vector_norm(problem.grad(forward.state))
        self.forward_count += 1
        terminal_contact = collision.diagnostics(
            forward.state.collision, forward.state.u
        )
        positions = (collision.vertices + forward.state.u[collision.indices]).numpy(
            force=True
        )
        intersects = bool(
            ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
        )
        min_gap = terminal_contact["minimum_active_distance_m"]
        strict_gap = min_gap is None or min_gap >= collision.min_distance
        self.last_forward = {
            "success": bool(solution.success)
            and float(exact_force) <= self.tolerances["atol"],
            "method": "strict_pncg_exact_curvature_resolved_work",
            "seconds": time.perf_counter() - started,
            "steps": int(solution.state.convergence_state.step),
            "grad_norm": float(exact_force),
            "force_threshold": self.tolerances["atol"],
            "result": str(solution.result),
            "rejected_contact_trials": problem.rejected_contact_trials,
            "stable_work_steps": getattr(problem, "work_line_search_count", 0),
            "last_line_searches": getattr(problem, "work_line_search_records", []),
            "contact": terminal_contact,
            "terminal_gates": {
                "no_intersections": not intersects,
                "minimum_active_gap_at_least_10nm": strict_gap,
            },
        }
        if not self.last_forward["success"]:
            raise ForwardConvergenceError(
                "expression PNCG did not meet accepted-state force tolerance",
                receipt=self.last_forward,
            )
        if not self.last_forward["contact"]["contact_numerically_valid"]:
            raise ForwardConvergenceError(
                "expression PNCG terminal contact invalid", receipt=self.last_forward
            )
        if not all(self.last_forward["terminal_gates"].values()):
            raise ForwardConvergenceError(
                "expression PNCG terminal collision gate failed",
                receipt=self.last_forward,
            )
        return forward.state.u.detach().clone()


def install_precise_expression_runtime(
    physics: Any,
    *,
    line_search_armijo: float = 0.25,
    hessian_damping_initial: float = 0.001,
    pncg_restart_interval: int = 200,
    max_step_norm_m: float = 0.0005,
    collision_step_safety: float = 0.95,
    ccd_tolerance_m: float = 1e-10,
    ccd_max_iterations: int = 100000,
) -> ExpressionEquilibrium:
    """Install the production expression runtime on eye-inclusive physics."""
    if not 0 < ccd_tolerance_m <= 1e-8:
        raise ValueError(
            "CCD tolerance must be positive and no larger than the 10 nm buffer"
        )
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    if collision.min_distance != 1e-8:
        raise ValueError(
            "expression runtime requires the inherited 10 nm numerical CCD buffer"
        )
    collision.narrow_phase_ccd = ipctk.TightInclusionCCD(
        tolerance=ccd_tolerance_m, max_iterations=ccd_max_iterations
    )
    old = physics.runtime
    runtime = ExpressionEquilibrium(
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
        collision_step_safety=collision_step_safety,
    )
    physics.runtime = runtime
    physics.contact_definition["config"].update(
        {"ccd_tolerance_m": ccd_tolerance_m, "ccd_max_iterations": ccd_max_iterations}
    )
    return runtime


__all__ = [
    "ExpressionEquilibrium",
    "FeasibleExpressionProblem",
    "install_precise_expression_runtime",
]
