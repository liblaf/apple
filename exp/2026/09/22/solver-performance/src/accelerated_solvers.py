# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, N818, PLR0912, PLR0915, PT018, SLF001, TRY003, TRY301
"""Explicit forward-solver variants; the physical implicit adjoint is unchanged."""

from __future__ import annotations

import contextlib
import io
import logging
import math
import time
from collections import Counter
from typing import Any, Literal

import torch
from joint_equilibrium import Equilibrium, ForwardConvergenceError, StrictLineSearch
from joint_expression_equilibrium import AcceptedForcePncg, FeasibleExpressionProblem

from liblaf.apple.forward._problem import ForwardProblem

LOG = logging.getLogger(__name__)


class CachedProblem:
    """Cache only a solve-local gradient at an unchanged displacement state.

    Materials and fixed values are held constant for this object's lifetime.
    The tensor mutation version also catches updates made outside ``update``.
    """

    def __init__(
        self,
        delegate: Any,
        *,
        cache_gradient: bool = True,
        exact_curvature: bool = False,
        wall_seconds: float | None = None,
    ) -> None:
        self.delegate = delegate
        self.model = delegate.model
        self.cache_gradient = cache_gradient
        self.exact_curvature = exact_curvature
        assert wall_seconds is None or (
            math.isfinite(wall_seconds) and wall_seconds > 0
        )
        self.deadline = (
            None if wall_seconds is None else time.perf_counter() + wall_seconds
        )
        self.counts: Counter = Counter()
        self._key = None
        self._gradient = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def check_budget(self) -> None:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise ForwardConvergenceError("declared forward wall budget exhausted")

    def update(self, state: Any, free: torch.Tensor) -> None:
        self.check_budget()
        self.counts["update"] += 1
        self._key = None
        self.delegate.update(state, free)

    def invalidate(self) -> None:
        """Drop a derivative cached across an external material transaction."""
        self._key = None
        self._gradient = None

    def fun(self, state: Any) -> torch.Tensor:
        self.check_budget()
        self.counts["fun"] += 1
        return self.delegate.fun(state)

    def grad(self, state: Any) -> torch.Tensor:
        self.check_budget()
        self.counts["grad_requests"] += 1
        key = (id(state), id(state.u), state.u._version)
        if not self.cache_gradient or key != self._key:
            self.counts["grad_evaluations"] += 1
            self._gradient = self.delegate.grad(state)
            self._key = key
        return self._gradient

    def hess_diag(self, state: Any) -> torch.Tensor:
        self.check_budget()
        self.counts["hess_diag"] += 1
        return self.delegate.hess_diag(state)

    def hess_prod(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.check_budget()
        self.counts["hess_prod"] += 1
        return self.delegate.hess_prod(state, direction)

    def hess_quad(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.check_budget()
        self.counts["hess_quad"] += 1
        if self.exact_curvature:
            return torch.dot(direction, self.hess_prod(state, direction))
        return self.delegate.hess_quad(state, direction)

    def max_step_size(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.check_budget()
        self.counts["ccd"] += 1
        return self.delegate.max_step_size(state, direction)


class LinearRejection(RuntimeError):
    pass


def newton_ccd_fraction(problem: Any, state: Any, direction: torch.Tensor) -> float:
    """Apply the Newton CCD margin once, preserving a hybrid's PNCG margin."""
    delegate = problem.delegate if isinstance(problem, CachedProblem) else problem
    if not isinstance(delegate, FeasibleExpressionProblem):
        return float(problem.max_step_size(state, direction))
    original_safety = delegate.collision_step_safety
    delegate.collision_step_safety = 0.9
    try:
        return float(problem.max_step_size(state, direction))
    finally:
        delegate.collision_step_safety = original_safety


def pcg(
    matvec: Any,
    precondition: Any,
    rhs: torch.Tensor,
    *,
    rtol: float,
    max_steps: int = 1000,
) -> tuple[torch.Tensor, dict]:
    """PCG with explicit negative-curvature rejection and a true residual check."""
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    norm = float(torch.linalg.vector_norm(rhs))
    if norm == 0:
        return x, {"steps": 0, "relative_residual": 0.0}
    z = precondition(residual)
    rz = torch.dot(residual, z)
    if not math.isfinite(float(rz)) or float(rz) <= 0:
        raise LinearRejection("preconditioner is not positive definite")
    direction = z.clone()
    for step in range(max_steps):
        hd = matvec(direction)
        curvature = torch.dot(direction, hd)
        if not math.isfinite(float(curvature)) or float(curvature) <= 0:
            raise LinearRejection("nonpositive or nonfinite CG curvature")
        alpha = rz / curvature
        x = x + alpha * direction
        residual = residual - alpha * hd
        if float(torch.linalg.vector_norm(residual)) <= rtol * norm:
            true_residual = rhs - matvec(x)
            relative = float(torch.linalg.vector_norm(true_residual)) / norm
            if not math.isfinite(relative):
                raise LinearRejection("nonfinite true linear residual")
            if relative <= rtol:
                return x, {"steps": step + 1, "relative_residual": relative}
            # Residual replacement restarts conjugacy; it never relaxes tolerance.
            residual = true_residual
            z = precondition(residual)
            rz = torch.dot(residual, z)
            direction = z.clone()
            continue
        z = precondition(residual)
        next_rz = torch.dot(residual, z)
        if not math.isfinite(float(next_rz)) or float(next_rz) <= 0:
            raise LinearRejection("invalid preconditioned residual")
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    raise LinearRejection("CG iteration budget exhausted")


def safeguarded_newton_step(
    problem: Any,
    state: Any,
    *,
    atol: float = 0.0,
    linear_rtol: float = 1e-3,
    linear_max_steps: int = 1000,
    max_step_norm: float,
    armijo: float = 1e-4,
    max_shift_attempts: int = 8,
    max_backtracking_trials: int = 8,
    backtracking_factor: float = 0.5,
    preconditioner: str = "diag",
    initial_shift_scale: float = 0.0,
    initial_shift_ratio: float = 0.0,
    shift_policy: Literal["reset", "reuse"] = "reset",
    reuse_shift_force_ratio: float = 3.0,
    shift_scale_policy: Literal["signed_mean", "mean_abs"] = "signed_mean",
    gradient: torch.Tensor | None = None,
) -> tuple[Any, dict]:
    """Accept one Newton step without imposing a whole-solve force gate.

    Shifts modify the search system only. Energy, force, terminal acceptance,
    and all implicit-adjoint Hessian products retain the original mechanics.
    ``initial_shift_scale`` starts a step at that multiple of the diagonal
    scale; its default preserves the historical unshifted first attempt.
    ``initial_shift_ratio`` is a solve-local diagonal-scale hint; zero resets
    the search to the unshifted system. ``mean_abs`` scales regularization by
    the mean absolute diagonal; ``signed_mean`` preserves earlier experiments.
    ``reuse_shift_force_ratio`` keeps the historical near-tolerance reset at
    three times ``atol`` by default; zero permits reuse through convergence.
    This search-policy option does not change the physical stopping tolerance.
    """
    assert atol >= 0 and 0 < linear_rtol < 1
    assert max_step_norm > 0 and 0 < armijo < 1
    assert (
        linear_max_steps > 0 and max_shift_attempts > 0 and max_backtracking_trials > 0
    )
    assert 0 < backtracking_factor < 1
    assert preconditioner in {"diag", "block"}
    assert initial_shift_scale >= 0
    assert shift_policy in {"reset", "reuse"}
    assert reuse_shift_force_ratio >= 0
    assert shift_scale_policy in {"signed_mean", "mean_abs"}
    model = problem.model
    free = model.dof_map.to_free(state.u).detach().clone()
    gradient = problem.grad(state) if gradient is None else gradient
    force = float(torch.linalg.vector_norm(gradient))
    if not math.isfinite(force):
        raise ForwardConvergenceError("nonfinite Newton force")
    energy = float(problem.fun(state))
    if not math.isfinite(energy):
        raise ForwardConvergenceError("nonfinite Newton energy")
    diagonal = problem.hess_diag(state)
    assert bool(torch.isfinite(diagonal).all())
    scale = float(
        (diagonal.abs() if shift_scale_policy == "mean_abs" else diagonal).mean()
    )
    assert math.isfinite(scale) and scale > 0, "Hessian shift scale must be positive"
    if preconditioner == "block":
        from vertex_blocks import build_vertex_preconditioner

        block = build_vertex_preconditioner(model, state)
    reuse_shift = shift_policy == "reuse" and force > reuse_shift_force_ratio * atol
    shift = (
        initial_shift_scale * scale
        if initial_shift_scale > 0
        else (initial_shift_ratio * scale / 10 if reuse_shift else 0.0)
    )
    if initial_shift_scale == 0 and shift < scale * 1e-6:
        shift = 0.0
    initial_shift = shift
    retries = []
    for _attempt in range(max_shift_attempts):
        started = time.perf_counter()
        shifted_block = block.with_shift(shift) if preconditioner == "block" else None
        apply = (
            (
                lambda vector, diagonal=diagonal, shift=shift: (
                    vector / (diagonal + shift).abs()
                )
            )
            if preconditioner == "diag"
            else shifted_block.apply
        )
        try:
            direction, linear = pcg(
                lambda vector, shift=shift: (
                    problem.hess_prod(state, vector) + shift * vector
                ),
                apply,
                -gradient,
                rtol=linear_rtol,
                max_steps=linear_max_steps,
            )
            slope = float(torch.dot(gradient, direction))
            if not math.isfinite(slope) or slope >= 0:
                raise LinearRejection("Newton direction is not descent")
        except LinearRejection as error:
            retries.append(
                {
                    "shift": shift,
                    "initial_shift": initial_shift,
                    "initial_shift_scale": initial_shift_scale,
                    "shift_policy": shift_policy,
                    "shift_scale_policy": shift_scale_policy,
                    "shift_scale": scale,
                    "reason": str(error),
                    "seconds": time.perf_counter() - started,
                }
            )
            shift = scale if shift == 0 else shift * 10
            continue
        linear["seconds"] = time.perf_counter() - started
        magnitude = float(direction.abs().max())
        alpha = min(1.0, max_step_norm / magnitude)
        ccd = newton_ccd_fraction(problem, state, alpha * direction)
        if not math.isfinite(ccd) or not 0 < ccd <= 1:
            raise ForwardConvergenceError("Newton CCD fraction is invalid")
        alpha *= ccd
        for backtrack in range(max_backtracking_trials):
            problem.update(state, free + alpha * direction)
            value = float(problem.fun(state))
            if math.isfinite(value) and value <= energy + armijo * alpha * slope:
                receipt = {
                    "force": force,
                    "energy": energy,
                    "shift": shift,
                    "initial_shift": initial_shift,
                    "initial_shift_scale": initial_shift_scale,
                    "shift_policy": shift_policy,
                    "shift_scale_policy": shift_scale_policy,
                    "shift_scale": scale,
                    "alpha": alpha,
                    "ccd": ccd,
                    "backtracks": backtrack,
                    "line_search_trials": backtrack + 1,
                    "max_coordinate_displacement": max_step_norm,
                    "linear": linear,
                    "regularization_retries": retries,
                    "shifted_preconditioner": (
                        shifted_block.setup_metadata
                        if shifted_block is not None
                        else None
                    ),
                    "next_shift_ratio": shift / scale,
                }
                return state, receipt
            alpha *= backtracking_factor
        problem.update(state, free)
        retries.append(
            {
                "shift": shift,
                "shift_scale_policy": shift_scale_policy,
                "shift_scale": scale,
                "reason": "Armijo exhausted",
                "line_search_trials": max_backtracking_trials,
            }
        )
        shift = scale if shift == 0 else shift * 10
    raise ForwardConvergenceError(
        "Newton regularization exhausted", receipt={"retries": retries}
    )


def safeguarded_newton(
    problem: Any,
    state: Any,
    *,
    atol: float,
    linear_rtol: float = 1e-3,
    linear_max_steps: int = 1000,
    max_steps: int = 100,
    max_step_norm: float,
    armijo: float = 1e-4,
    max_shift_attempts: int = 8,
    max_backtracking_trials: int = 8,
    backtracking_factor: float = 0.5,
    preconditioner: str = "diag",
    initial_shift_scale: float = 0.0,
    shift_policy: Literal["reset", "reuse"] = "reset",
    reuse_shift_force_ratio: float = 3.0,
    shift_scale_policy: Literal["signed_mean", "mean_abs"] = "signed_mean",
    post_step: Any = None,
) -> tuple[Any, dict]:
    """Run accepted safeguarded Newton steps until the original force gate."""
    assert atol > 0 and 0 < linear_rtol < 1 and max_steps > 0
    assert max_step_norm > 0 and 0 < armijo < 1
    assert preconditioner in {"diag", "block"}
    assert initial_shift_scale >= 0
    assert shift_policy in {"reset", "reuse"}
    assert shift_scale_policy in {"signed_mean", "mean_abs"}
    trace = []
    previous_shift_ratio = 0.0
    for iteration in range(max_steps):
        gradient = problem.grad(state)
        force = float(torch.linalg.vector_norm(gradient))
        if not math.isfinite(force):
            raise ForwardConvergenceError("nonfinite Newton force")
        if force <= atol:
            return state, {"steps": len(trace), "trace": trace}
        try:
            state, receipt = safeguarded_newton_step(
                problem,
                state,
                atol=atol,
                linear_rtol=linear_rtol,
                linear_max_steps=linear_max_steps,
                max_step_norm=max_step_norm,
                armijo=armijo,
                max_shift_attempts=max_shift_attempts,
                max_backtracking_trials=max_backtracking_trials,
                backtracking_factor=backtracking_factor,
                preconditioner=preconditioner,
                initial_shift_scale=initial_shift_scale,
                initial_shift_ratio=previous_shift_ratio,
                shift_policy=shift_policy,
                reuse_shift_force_ratio=reuse_shift_force_ratio,
                shift_scale_policy=shift_scale_policy,
                gradient=gradient,
            )
        except ForwardConvergenceError as error:
            if str(error) != "Newton regularization exhausted":
                raise
            raise ForwardConvergenceError(
                "Newton regularization exhausted",
                receipt={"trace": trace, "retries": error.receipt["retries"]},
            ) from error
        previous_shift_ratio = receipt.pop("next_shift_ratio")
        if post_step is not None:
            observation = post_step(state, iteration + 1)
            if observation is not None:
                receipt["adaptive_stiffness"] = observation
        trace.append({"iteration": iteration, **receipt})
        LOG.info(
            "Newton %d force %.5g, CG %d, shift %.3g, alpha %.3g",
            iteration,
            receipt["force"],
            receipt["linear"]["steps"],
            receipt["shift"],
            receipt["alpha"],
        )
    force = float(torch.linalg.vector_norm(problem.grad(state)))
    if not math.isfinite(force) or force > atol:
        raise ForwardConvergenceError(
            "Newton iteration budget exhausted",
            receipt={"trace": trace, "grad_norm": force},
        )
    return state, {"steps": len(trace), "trace": trace}


def hybrid_pncg_newton(
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
    linear_rtol: float,
    max_newton_steps: int,
    newton_max_step_norm: float,
    preconditioner: str,
    newton_switch_atol: float = 0.0,
    shift_policy: Literal["reset", "reuse"] = "reset",
) -> tuple[Any, dict[str, Any]]:
    """Run cached accepted-force PNCG, then exact safeguarded Newton.

    The coarse tolerance is a solver transition criterion only.  Newton must
    still meet the original accepted-state force threshold; this routine never
    falls back to a different solver after a failure.
    """
    assert math.isfinite(initial_force) and initial_force > atol > 0
    assert math.isfinite(newton_switch_atol) and newton_switch_atol >= 0
    coarse_threshold = max(atol, initial_force * 1e-3, newton_switch_atol)
    if initial_force <= coarse_threshold:
        coarse_steps = 0
        coarse_force = initial_force
    else:
        default = make_default_optimizer(coarse_threshold)
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
        with contextlib.redirect_stdout(io.StringIO()):
            coarse = optimizer.minimize(
                problem, state, problem.model.dof_map.to_free(state.u)
            )
        if not coarse.success:
            raise ForwardConvergenceError("coarse PNCG did not converge")
        coarse_steps = int(coarse.state.convergence_state.step)
        coarse_force = float(torch.linalg.vector_norm(problem.grad(state)))
    if not math.isfinite(coarse_force) or coarse_force > coarse_threshold:
        raise ForwardConvergenceError(
            "coarse PNCG missed its accepted-state force gate"
        )
    state, newton = safeguarded_newton(
        problem,
        state,
        atol=atol,
        linear_rtol=linear_rtol,
        max_steps=max_newton_steps,
        max_step_norm=newton_max_step_norm,
        preconditioner=preconditioner,
        shift_policy=shift_policy,
    )
    return state, {
        "steps": coarse_steps + int(newton["steps"]),
        "coarse_steps": coarse_steps,
        "coarse_threshold": coarse_threshold,
        "newton_switch_atol": newton_switch_atol,
        "coarse_terminal_force": coarse_force,
        "newton_steps": int(newton["steps"]),
        "trace": newton["trace"],
    }


class AcceleratedEquilibrium(Equilibrium):
    """Same owned-snapshot implicit differentiation, alternative primal only."""

    def __init__(
        self,
        reference: Any,
        method: str,
        *,
        rest_points: Any,
        wall_seconds: float | None,
        linear_rtol: float,
        max_newton_steps: int,
        newton_switch_atol: float,
        shift_policy: Literal["reset", "reuse"] = "reset",
    ) -> None:
        self.__dict__.update(reference.__dict__)
        self.warm_adjoints = {}
        self.last_forward = {}
        self.last_adjoint = {}
        self.method = method
        self.wall_seconds = wall_seconds
        self.linear_rtol = linear_rtol
        self.max_newton_steps = max_newton_steps
        self.newton_switch_atol = newton_switch_atol
        self.shift_policy = shift_policy
        self.adaptive_options = {}
        self.trace_callback = None
        self.newton_parameters = None
        if method.startswith(("newton_", "hybrid_")) or method == "adaptive_diag":
            from mesh_step_scale import mean_rest_edge_length

            assert rest_points is not None, (
                "Newton's mesh-scaled cap requires rest_points"
            )
            self.mean_mesh_edge_length = mean_rest_edge_length(
                self.forward.model, rest_points
            )
            self.newton_max_step_norm = 0.5 * self.mean_mesh_edge_length
            self.newton_parameters = {
                "linear_rtol": linear_rtol,
                "linear_max_steps": 1000,
                "preconditioner": "vertex_block"
                if method.endswith("_block")
                else "abs_jacobi",
                "initial_shift": 0.0,
                "shift_policy": shift_policy,
                "shift_scale": "mean(diag(H))",
                "shift_multiplier": 10.0,
                "max_shift_attempts": 8,
                "mean_mesh_edge_length_m": self.mean_mesh_edge_length,
                "max_coordinate_displacement_m": self.newton_max_step_norm,
                "armijo": 1e-4,
                "backtracking_factor": 0.5,
                "max_backtracking_trials": 8,
                "ccd_safety_factor": 0.9,
                "max_steps": max_newton_steps,
                "wall_seconds": wall_seconds,
            }

    def primal(
        self, materials: dict, fixed: torch.Tensor, seed: torch.Tensor
    ) -> torch.Tensor:
        started = time.perf_counter()
        forward, model = self.forward, self.forward.model
        collision = model.collision
        model.set_materials(materials)
        model.dof_map.fixed_values = fixed.detach().clone()
        state = model.State(
            u=model.dof_map.to_full(model.dof_map.to_free(seed)).detach().clone()
        )
        if collision is not None:
            prior = collision.state_at(seed.detach())
            fraction = float(
                collision.max_step_size(prior, seed.detach(), state.u - seed)
            )
            if fraction < 1.0:
                raise ForwardConvergenceError(
                    "Dirichlet boundary proposal fails contact CCD"
                )
            state.collision = collision.state_at(state.u)
            state.collision.boundary_ccd_fraction = fraction
            initial_contact = collision.diagnostics(state.collision, state.u)
            if not initial_contact["contact_numerically_valid"]:
                raise ForwardConvergenceError(
                    "expression seed violates contact feasibility",
                    receipt=initial_contact,
                )
            delegate = FeasibleExpressionProblem(
                model=model, collision_step_safety=self.collision_step_safety
            )
        else:
            delegate = ForwardProblem(model=model)
        problem = CachedProblem(
            delegate,
            cache_gradient=self.method != "baseline",
            exact_curvature=self.method == "exact_pncg",
            wall_seconds=self.wall_seconds,
        )
        forward.state, forward.problem = state, problem
        self.last_problem = problem
        try:
            initial_force = float(torch.linalg.vector_norm(problem.grad(state)))
            if initial_force <= self.tolerances["atol"]:
                result = {"steps": 0, "trace": []}
            elif self.method == "adaptive_diag":
                from adaptive_pncg import adaptive_pncg_newton

                state, result = adaptive_pncg_newton(
                    problem,
                    state,
                    atol=self.tolerances["atol"],
                    make_default_optimizer=lambda threshold: forward.default_optimizer(
                        max_steps=self.tolerances["max_steps"], rtol=0.0, atol=threshold
                    ),
                    hessian_damping_initial=self.hessian_damping_initial,
                    line_search_armijo=self.line_search_armijo,
                    max_step_norm=self.max_step_norm_m,
                    pncg_restart_interval=self.pncg_restart_interval,
                    linear_rtol=self.linear_rtol,
                    max_newton_steps=self.max_newton_steps,
                    newton_max_step_norm=self.newton_max_step_norm,
                    max_pncg_steps=self.tolerances["max_steps"],
                    callback=self.trace_callback,
                    **self.adaptive_options,
                )
            elif self.method.startswith("newton_"):
                state, result = safeguarded_newton(
                    problem,
                    state,
                    atol=self.tolerances["atol"],
                    linear_rtol=self.linear_rtol,
                    max_steps=self.max_newton_steps,
                    max_step_norm=self.newton_max_step_norm,
                    preconditioner="block" if self.method == "newton_block" else "diag",
                    shift_policy=self.shift_policy,
                )
            elif self.method == "hybrid_first":
                from hybrid_first_solver import hybrid_first

                state, result = hybrid_first(
                    problem,
                    state,
                    atol=self.tolerances["atol"],
                    max_step_norm=self.newton_max_step_norm,
                    linear_rtol=self.linear_rtol,
                    max_newton_steps=self.max_newton_steps,
                    callback=self.trace_callback,
                )
            elif self.method.startswith("hybrid_"):
                state, result = hybrid_pncg_newton(
                    problem,
                    state,
                    initial_force=initial_force,
                    atol=self.tolerances["atol"],
                    make_default_optimizer=lambda threshold: forward.default_optimizer(
                        max_steps=self.tolerances["max_steps"],
                        rtol=0.0,
                        atol=threshold,
                    ),
                    hessian_damping_initial=self.hessian_damping_initial,
                    line_search_armijo=self.line_search_armijo,
                    max_step_norm=self.max_step_norm_m,
                    pncg_restart_interval=self.pncg_restart_interval,
                    linear_rtol=self.linear_rtol,
                    max_newton_steps=self.max_newton_steps,
                    newton_max_step_norm=self.newton_max_step_norm,
                    preconditioner="block" if self.method == "hybrid_block" else "diag",
                    newton_switch_atol=self.newton_switch_atol,
                    shift_policy=self.shift_policy,
                )
            else:
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
                with contextlib.redirect_stdout(io.StringIO()):
                    solution = optimizer.minimize(
                        problem, state, model.dof_map.to_free(state.u)
                    )
                if not solution.success:
                    raise ForwardConvergenceError("PNCG did not converge")
                result = {
                    "steps": int(solution.state.convergence_state.step),
                    "trace": [],
                }
            force = float(torch.linalg.vector_norm(problem.grad(state)))
            if not math.isfinite(force) or force > self.tolerances["atol"]:
                raise ForwardConvergenceError(
                    "terminal accepted-state force gate failed"
                )
            if collision is not None:
                import ipctk

                contact = collision.diagnostics(state.collision, state.u)
                points = (collision.vertices + state.u[collision.indices]).numpy(
                    force=True
                )
                intersects = bool(
                    ipctk.has_intersections(
                        collision.collision_mesh, points, ipctk.LBVH()
                    )
                )
                gap = contact["minimum_active_distance_m"]
                if (
                    not contact["contact_numerically_valid"]
                    or intersects
                    or (gap is not None and gap < collision.min_distance)
                ):
                    raise ForwardConvergenceError(
                        "terminal expression contact gate failed", receipt=contact
                    )
            else:
                contact = {"enabled": False, "physical_contact_validated": False}
            if state.u.is_cuda:
                torch.cuda.synchronize(state.u.device)
            self.last_forward = {
                "success": True,
                "method": self.method,
                "newton_parameters": self.newton_parameters,
                "seconds": time.perf_counter() - started,
                "grad_norm": force,
                "force_threshold": self.tolerances["atol"],
                "contact": contact,
                "counts": dict(problem.counts),
                **result,
            }
            self.forward_count += 1
            forward.state = state
            return state.u.detach().clone()
        except ForwardConvergenceError as error:
            self.last_forward = {
                "success": False,
                "method": self.method,
                "newton_parameters": self.newton_parameters,
                "seconds": time.perf_counter() - started,
                "failure": str(error),
                "counts": dict(problem.counts),
                "solver": error.receipt,
            }
            raise


def accelerate_runtime(
    reference_runtime: Any,
    method: str,
    *,
    rest_points: Any = None,
    wall_seconds: float | None = None,
    linear_rtol: float = 1e-3,
    max_newton_steps: int = 100,
    newton_switch_atol: float = 0.0,
    shift_policy: Literal["reset", "reuse"] = "reset",
) -> AcceleratedEquilibrium:
    assert method in {
        "baseline",
        "cached_pncg",
        "exact_pncg",
        "newton_diag",
        "newton_block",
        "hybrid_diag",
        "hybrid_block",
        "hybrid_first",
        "adaptive_diag",
    }
    assert math.isfinite(newton_switch_atol) and newton_switch_atol >= 0
    assert shift_policy in {"reset", "reuse"}
    return AcceleratedEquilibrium(
        reference_runtime,
        method,
        rest_points=rest_points,
        wall_seconds=wall_seconds,
        linear_rtol=linear_rtol,
        max_newton_steps=max_newton_steps,
        newton_switch_atol=newton_switch_atol,
        shift_policy=shift_policy,
    )
