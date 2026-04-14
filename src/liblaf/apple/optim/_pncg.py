from __future__ import annotations

import time
from typing import Literal, Protocol, override

import attrs
import jarp
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Optimizer, Result, Solution
from liblaf.peach.optim.base import SupportsFun
from liblaf.peach.optim.pncg import PNCGObjective

type BooleanNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

_MAX_CONJUGATE_WEIGHT: float = 10.0


class NonlinearStageObjective[X](Protocol):
    def update_state(self, parameters: Vector) -> X: ...

    def objective_value(self, state: X) -> Scalar: ...

    def gradient(self, state: X) -> Vector: ...

    def hessian_diagonal(self, state: X) -> Vector: ...

    def curvature_along(self, state: X, direction: Vector) -> Scalar: ...


class LineSearchObjective[X](
    NonlinearStageObjective[X], PNCGObjective[X], SupportsFun[X], Protocol
): ...


@attrs.frozen
class ConvergenceCriteria:
    target_relative_gradient_norm: Scalar = jnp.asarray(1e-5)
    acceptable_relative_gradient_norm: Scalar = jnp.asarray(1e-3)
    target_absolute_gradient_norm: Scalar = jnp.asarray(1e-10)
    acceptable_absolute_gradient_norm: Scalar = jnp.asarray(1e-10)
    maximum_iterations: Integer[Array, ""] = jnp.asarray(1000, dtype=jnp.int32)


@attrs.frozen
class LineSearchSettings:
    armijo_sufficient_decrease: Scalar = jnp.asarray(0.0)
    backtracking_factor: Scalar = jnp.asarray(0.5)
    maximum_backtracking_steps: Integer[Array, ""] = jnp.asarray(20, dtype=jnp.int32)
    maximum_step_inf_norm: Scalar | None = jnp.asarray(jnp.inf)


@attrs.frozen
class HessianDamping:
    constant_strength: Scalar = jnp.asarray(0.0)


@attrs.frozen
class PNCGOverrides:
    convergence: ConvergenceCriteria | None = None
    line_search: LineSearchSettings | None = None
    damping: HessianDamping | None = None


@attrs.frozen
class PNCGIteration:
    parameters: Vector
    objective_value: Scalar
    best_objective_value: Scalar
    gradient_norm: Scalar
    initial_gradient_norm: Scalar
    relative_gradient_norm: Scalar
    step_length: Scalar
    conjugate_weight: Scalar
    used_steepest_descent_reset: BooleanNumeric
    regularization_strength: Scalar
    backtracking_steps: Integer[Array, ""]
    curvature_along_search_direction: Scalar


@attrs.frozen
class PNCGResult:
    success: bool
    status: Literal[
        "target_converged",
        "acceptable_converged",
        "maximum_iterations_reached",
        "stagnated",
        "numerical_failure",
    ]
    final_iteration: PNCGIteration
    best_parameters: Vector


@jarp.define(kw_only=True)
class PNCGState(Optimizer.State):
    n_steps: Integer[Array, ""] = jarp.array(default=0)

    alpha: Scalar = jarp.array()
    beta: Scalar = jarp.array()
    decrease: Scalar = jarp.array()
    first_decrease: Scalar = jarp.array()
    value: Scalar = jarp.array()

    grad: Vector = jarp.array()
    hess_diag: Vector = jarp.array()
    hess_quad: Scalar = jarp.array()
    params: Vector = jarp.array()
    preconditioner: Vector = jarp.array()
    search_direction: Vector = jarp.array()

    best_decrease: Scalar = jarp.array(default=jnp.inf)
    best_params: Vector = jarp.array()

    stagnation_counter: Integer[Array, ""] = jarp.array(default=0)
    stagnation_restarts: Integer[Array, ""] = jarp.array(default=0)

    first_value: Scalar = jarp.array(default=jnp.inf)
    first_grad_norm: Scalar = jarp.array(default=jnp.inf)
    first_grad_norm_max: Scalar = jarp.array(default=jnp.inf)
    best_value: Scalar = jarp.array(default=jnp.inf)

    backtracking_steps: Integer[Array, ""] = jarp.array(default=0)
    used_steepest_descent_reset: BooleanNumeric = jarp.array(default=False)
    force_steepest_descent_next: BooleanNumeric = jarp.array(default=False)


@jarp.define(kw_only=True)
class PNCGStats(Optimizer.Stats):
    relative_decrease: Scalar = jarp.array(default=jnp.inf)
    grad_norm: Scalar = jarp.array(default=jnp.inf)
    grad_norm_max: Scalar = jarp.array(default=jnp.inf)
    first_value: Scalar = jarp.array(default=jnp.inf)
    best_value: Scalar = jarp.array(default=jnp.inf)
    first_grad_norm: Scalar = jarp.array(default=jnp.inf)
    first_grad_norm_max: Scalar = jarp.array(default=jnp.inf)


@jarp.define(kw_only=True)
class PNCG(Optimizer[PNCGObjective, PNCGState, PNCGStats]):
    State = PNCGState
    Stats = PNCGStats

    type Callback[X] = Optimizer.Callback[X, PNCG.State, PNCG.Stats]
    type Solution = Optimizer.Solution[State, Stats]

    convergence: ConvergenceCriteria = jarp.field(factory=ConvergenceCriteria)
    line_search: LineSearchSettings = jarp.field(factory=LineSearchSettings)
    damping: HessianDamping = jarp.field(factory=HessianDamping)
    stagnation_max_restarts: Integer[Array, ""] = jarp.array(default=40)
    stagnation_patience: Integer[Array, ""] = jarp.array(default=40)
    jit: bool = jarp.static(default=True)

    def __init__(
        self,
        convergence: ConvergenceCriteria | None = None,
        line_search: LineSearchSettings | None = None,
        damping: HessianDamping | None = None,
        *,
        use_jit: bool = True,
        jit: bool | None = None,
        max_steps: Integer[Array, ""] | int | None = None,
        atol: Scalar | float | None = None,
        rtol: Scalar | float | None = None,
        atol_primary: Scalar | float | None = None,
        rtol_primary: Scalar | float | None = None,
        line_search_factor: Scalar | float | None = None,
        line_search_c1: Scalar | float | None = None,
        max_line_search_steps: Integer[Array, ""] | int | None = None,
        max_delta: Scalar | float | None = None,
        stagnation_max_restarts: Integer[Array, ""] | int | None = None,
        stagnation_patience: Integer[Array, ""] | int | None = None,
        beta_reset_threshold: Scalar | float | None = None,  # noqa: ARG002
        beta_non_negative: bool | None = None,  # noqa: ARG002
        reset_beta_on_non_descent: bool | None = None,  # noqa: ARG002
    ) -> None:
        convergence = convergence or ConvergenceCriteria()
        line_search = line_search or LineSearchSettings()
        damping = damping or HessianDamping()
        if max_steps is not None:
            convergence = attrs.evolve(convergence, maximum_iterations=max_steps)
        if atol is not None:
            convergence = attrs.evolve(
                convergence, acceptable_absolute_gradient_norm=atol
            )
        if rtol is not None:
            convergence = attrs.evolve(
                convergence, acceptable_relative_gradient_norm=rtol
            )
        if atol_primary is not None:
            convergence = attrs.evolve(
                convergence, target_absolute_gradient_norm=atol_primary
            )
        if rtol_primary is not None:
            convergence = attrs.evolve(
                convergence, target_relative_gradient_norm=rtol_primary
            )
        if line_search_factor is not None:
            line_search = attrs.evolve(
                line_search, backtracking_factor=line_search_factor
            )
        if line_search_c1 is not None:
            line_search = attrs.evolve(
                line_search, armijo_sufficient_decrease=line_search_c1
            )
        if max_line_search_steps is not None:
            line_search = attrs.evolve(
                line_search, maximum_backtracking_steps=max_line_search_steps
            )
        if max_delta is not None:
            line_search = attrs.evolve(line_search, maximum_step_inf_norm=max_delta)
        convergence = _canonicalize_convergence(convergence)
        line_search = _canonicalize_line_search(line_search)
        damping = _canonicalize_damping(damping)
        use_jit = use_jit if jit is None else jit
        self.__attrs_init__(  # pyright: ignore[reportAttributeAccessIssue]
            convergence=convergence,
            line_search=line_search,
            damping=damping,
            stagnation_max_restarts=jnp.asarray(
                40 if stagnation_max_restarts is None else stagnation_max_restarts,
                dtype=jnp.int32,
            ),
            stagnation_patience=jnp.asarray(
                40 if stagnation_patience is None else stagnation_patience,
                dtype=jnp.int32,
            ),
            jit=use_jit,
        )

    @property
    def use_jit(self) -> bool:
        return self.jit

    @property
    def max_steps(self) -> Integer[Array, ""]:
        return self.convergence.maximum_iterations

    @property
    def atol(self) -> Scalar:
        return self.convergence.acceptable_absolute_gradient_norm

    @property
    def rtol(self) -> Scalar:
        return self.convergence.acceptable_relative_gradient_norm

    @property
    def atol_primary(self) -> Scalar:
        return self.convergence.target_absolute_gradient_norm

    @property
    def rtol_primary(self) -> Scalar:
        return self.convergence.target_relative_gradient_norm

    @property
    def line_search_factor(self) -> Scalar:
        return self.line_search.backtracking_factor

    @property
    def line_search_c1(self) -> Scalar:
        return self.line_search.armijo_sufficient_decrease

    @property
    def max_line_search_steps(self) -> Integer[Array, ""]:
        return self.line_search.maximum_backtracking_steps

    @property
    def max_delta(self) -> Scalar:
        if self.line_search.maximum_step_inf_norm is None:
            return jnp.asarray(jnp.inf)
        return self.line_search.maximum_step_inf_norm

    def solve[X](
        self,
        objective: NonlinearStageObjective[X],
        *,
        initial_parameters: Vector,
        overrides: PNCGOverrides | None = None,
        callback: Callback[X] | None = None,
    ) -> PNCGResult:
        solver: PNCG = self._with_overrides(overrides)
        model_state = solver._initial_model_state(objective, initial_parameters)
        solution: Solution[PNCGState, PNCGStats]
        solution, _ = solver.minimize(
            objective, model_state, initial_parameters, callback=callback
        )
        return solver._to_result(solution)

    def _with_overrides(self, overrides: PNCGOverrides | None) -> PNCG:
        if overrides is None:
            return self
        return attrs.evolve(
            self,
            convergence=overrides.convergence or self.convergence,
            line_search=overrides.line_search or self.line_search,
            damping=overrides.damping or self.damping,
        )

    def _initial_model_state[X](
        self, objective: NonlinearStageObjective[X], initial_parameters: Vector
    ) -> X:
        if hasattr(objective, "update_state"):
            return objective.update_state(initial_parameters)
        return initial_parameters

    @override
    def init[X](
        self, objective: PNCGObjective[X], model_state: X, params: Vector
    ) -> tuple[State, Stats]:
        del objective, model_state
        zeros: Vector = jnp.zeros_like(params)
        state = PNCGState(
            n_steps=jnp.zeros((), jnp.int32),
            alpha=jnp.zeros((), dtype=params.dtype),
            beta=jnp.zeros((), dtype=params.dtype),
            decrease=jnp.asarray(jnp.inf, dtype=params.dtype),
            first_decrease=jnp.asarray(jnp.inf, dtype=params.dtype),
            value=jnp.asarray(jnp.inf, dtype=params.dtype),
            grad=zeros,
            hess_diag=zeros,
            hess_quad=jnp.zeros((), dtype=params.dtype),
            params=params,
            preconditioner=zeros,
            search_direction=zeros,
            best_decrease=jnp.asarray(jnp.inf, dtype=params.dtype),
            best_params=params,
            stagnation_counter=jnp.zeros((), jnp.int32),
            stagnation_restarts=jnp.zeros((), jnp.int32),
            first_value=jnp.asarray(jnp.inf, dtype=params.dtype),
            first_grad_norm=jnp.asarray(jnp.inf, dtype=params.dtype),
            first_grad_norm_max=jnp.asarray(jnp.inf, dtype=params.dtype),
            best_value=jnp.asarray(jnp.inf, dtype=params.dtype),
            backtracking_steps=jnp.zeros((), jnp.int32),
            used_steepest_descent_reset=jnp.asarray(False),
            force_steepest_descent_next=jnp.asarray(False),
        )
        stats = PNCGStats(relative_decrease=jnp.asarray(jnp.inf, dtype=params.dtype))
        return state, stats

    @override
    def update_stats[X](
        self,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> Stats:
        del objective, model_state
        grad_norm = jnp.linalg.norm(opt_state.grad)
        grad_norm_max = jnp.linalg.norm(opt_state.grad, ord=jnp.inf)
        relative_decrease = jnp.where(
            jnp.abs(opt_state.first_decrease) > 0.0,
            opt_state.decrease / opt_state.first_decrease,
            jnp.zeros_like(opt_state.decrease),
        )
        return attrs.evolve(
            opt_stats,
            relative_decrease=relative_decrease,
            grad_norm=grad_norm,
            grad_norm_max=grad_norm_max,
            first_value=opt_state.first_value,
            best_value=opt_state.best_value,
            first_grad_norm=opt_state.first_grad_norm,
            first_grad_norm_max=opt_state.first_grad_norm_max,
        )

    @override
    def step[X](
        self, objective: LineSearchObjective[X], model_state: X, opt_state: State
    ) -> tuple[X, State]:
        model_state = objective.update(model_state, opt_state.params)
        value: Scalar = objective.fun(model_state)
        grad: Vector = objective.grad(model_state)
        hess_diag: Vector = objective.hess_diag(model_state)
        damping_strength: Scalar = jnp.asarray(
            self.damping.constant_strength, dtype=hess_diag.dtype
        )
        hess_diag = hess_diag + damping_strength
        preconditioner: Vector = _make_preconditioner(hess_diag)

        beta: Scalar
        search_direction: Vector
        grad_dot_direction: Scalar
        used_steepest_descent_reset: BooleanNumeric
        (
            beta,
            search_direction,
            grad_dot_direction,
            used_steepest_descent_reset,
            opt_state,
        ) = self._compute_direction(
            grad=grad,
            preconditioner=preconditioner,
            state=opt_state,
        )

        hess_quad: Scalar = objective.hess_quad(model_state, search_direction)
        hess_quad = hess_quad + damping_strength * jnp.vdot(
            search_direction, search_direction
        )
        is_valid: BooleanNumeric = (
            jnp.isfinite(value)
            & jnp.isfinite(grad_dot_direction)
            & jnp.isfinite(hess_quad)
            & jnp.all(jnp.isfinite(grad))
            & jnp.all(jnp.isfinite(hess_diag))
            & jnp.all(jnp.isfinite(preconditioner))
        )
        if self.jit:
            return jax.lax.cond(
                is_valid,
                lambda _: self._step_with_line_search(
                    objective=objective,
                    model_state=model_state,
                    opt_state=opt_state,
                    value=value,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    beta=beta,
                    search_direction=search_direction,
                    grad_dot_direction=grad_dot_direction,
                    hess_quad=hess_quad,
                    used_steepest_descent_reset=used_steepest_descent_reset,
                ),
                lambda _: self._step_nan(
                    model_state=model_state,
                    opt_state=opt_state,
                    value=value,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    search_direction=search_direction,
                    hess_quad=hess_quad,
                    used_steepest_descent_reset=used_steepest_descent_reset,
                ),
                operand=None,
            )
        if bool(np.asarray(is_valid)):
            return self._step_with_line_search(
                objective=objective,
                model_state=model_state,
                opt_state=opt_state,
                value=value,
                grad=grad,
                hess_diag=hess_diag,
                preconditioner=preconditioner,
                beta=beta,
                search_direction=search_direction,
                grad_dot_direction=grad_dot_direction,
                hess_quad=hess_quad,
                used_steepest_descent_reset=used_steepest_descent_reset,
            )
        return self._step_nan(
            model_state=model_state,
            opt_state=opt_state,
            value=value,
            grad=grad,
            hess_diag=hess_diag,
            preconditioner=preconditioner,
            search_direction=search_direction,
            hess_quad=hess_quad,
            used_steepest_descent_reset=used_steepest_descent_reset,
        )

    def _compute_direction(
        self, *, grad: Vector, preconditioner: Vector, state: State
    ) -> tuple[Scalar, Vector, Scalar, BooleanNumeric, State]:
        force_steepest_descent: BooleanNumeric = state.force_steepest_descent_next
        restart_from_stagnation: BooleanNumeric = (
            state.stagnation_counter >= self.stagnation_patience
        )
        restart_to_steepest_descent: BooleanNumeric = (
            (state.n_steps == 0) | force_steepest_descent | restart_from_stagnation
        )
        state = attrs.evolve(
            state,
            stagnation_counter=jax.lax.select(
                restart_from_stagnation,
                jnp.zeros_like(state.stagnation_counter),
                state.stagnation_counter,
            ),
            stagnation_restarts=state.stagnation_restarts
            + restart_from_stagnation.astype(state.stagnation_restarts.dtype),
            force_steepest_descent_next=jnp.asarray(False),
        )
        beta: Scalar = _compute_dai_kou_beta(
            g=grad,
            g_prev=state.grad,
            p_prev=state.search_direction,
            preconditioner=preconditioner,
        )
        beta_is_valid: BooleanNumeric = jnp.isfinite(beta) & (
            jnp.abs(beta) <= _MAX_CONJUGATE_WEIGHT
        )
        reset_beta: BooleanNumeric = restart_to_steepest_descent | (~beta_is_valid)
        beta = jnp.where(reset_beta, jnp.zeros((), dtype=grad.dtype), beta)
        search_direction: Vector = (
            -preconditioner * grad + beta * state.search_direction
        )
        grad_dot_direction: Scalar = jnp.vdot(grad, search_direction)
        is_descent: BooleanNumeric = jnp.isfinite(grad_dot_direction) & (
            grad_dot_direction < 0.0
        )
        if self.jit:
            beta, search_direction, grad_dot_direction = jax.lax.cond(
                is_descent,
                lambda _: (beta, search_direction, grad_dot_direction),
                lambda _: _steepest_descent_direction(grad, preconditioner),
                operand=None,
            )
        elif not bool(np.asarray(is_descent)):
            beta, search_direction, grad_dot_direction = _steepest_descent_direction(
                grad, preconditioner
            )
        used_steepest_descent_reset: BooleanNumeric = reset_beta | (~is_descent)
        return (
            beta,
            search_direction,
            grad_dot_direction,
            used_steepest_descent_reset,
            state,
        )

    def _step_with_line_search[X](
        self,
        *,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        value: Scalar,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        beta: Scalar,
        search_direction: Vector,
        grad_dot_direction: Scalar,
        hess_quad: Scalar,
        used_steepest_descent_reset: BooleanNumeric,
    ) -> tuple[X, State]:
        alpha: Scalar = _initial_alpha(
            grad=grad,
            search_direction=search_direction,
            hess_quad=hess_quad,
            maximum_step_inf_norm=self.line_search.maximum_step_inf_norm,
        )
        trial_state: X
        trial_value: Scalar
        accepted: BooleanNumeric
        backtracking_steps: Integer[Array, ""]
        trial_state, alpha, trial_value, accepted, backtracking_steps = (
            _backtracking_line_search(
                objective=objective,
                model_state=model_state,
                params=opt_state.params,
                search_direction=search_direction,
                value=value,
                grad_dot_direction=grad_dot_direction,
                alpha=alpha,
                factor=self.line_search.backtracking_factor,
                c1=self.line_search.armijo_sufficient_decrease,
                max_steps=self.line_search.maximum_backtracking_steps,
                jit=self.jit,
            )
        )

        if self.jit:
            return jax.lax.cond(
                accepted,
                lambda _: self._accept_step(
                    model_state=trial_state,
                    opt_state=opt_state,
                    value=value,
                    trial_value=trial_value,
                    alpha=alpha,
                    beta=beta,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    search_direction=search_direction,
                    hess_quad=hess_quad,
                    backtracking_steps=backtracking_steps,
                    used_steepest_descent_reset=used_steepest_descent_reset,
                ),
                lambda _: self._reject_step(
                    objective=objective,
                    model_state=model_state,
                    opt_state=opt_state,
                    value=value,
                    params=opt_state.params,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    search_direction=search_direction,
                    hess_quad=hess_quad,
                    backtracking_steps=backtracking_steps,
                    used_steepest_descent_reset=used_steepest_descent_reset,
                ),
                operand=None,
            )
        if bool(np.asarray(accepted)):
            return self._accept_step(
                model_state=trial_state,
                opt_state=opt_state,
                value=value,
                trial_value=trial_value,
                alpha=alpha,
                beta=beta,
                grad=grad,
                hess_diag=hess_diag,
                preconditioner=preconditioner,
                search_direction=search_direction,
                hess_quad=hess_quad,
                backtracking_steps=backtracking_steps,
                used_steepest_descent_reset=used_steepest_descent_reset,
            )
        return self._reject_step(
            objective=objective,
            model_state=model_state,
            opt_state=opt_state,
            value=value,
            params=opt_state.params,
            grad=grad,
            hess_diag=hess_diag,
            preconditioner=preconditioner,
            search_direction=search_direction,
            hess_quad=hess_quad,
            backtracking_steps=backtracking_steps,
            used_steepest_descent_reset=used_steepest_descent_reset,
        )

    def _accept_step[X](
        self,
        *,
        model_state: X,
        opt_state: State,
        value: Scalar,
        trial_value: Scalar,
        alpha: Scalar,
        beta: Scalar,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        search_direction: Vector,
        hess_quad: Scalar,
        backtracking_steps: Integer[Array, ""],
        used_steepest_descent_reset: BooleanNumeric,
    ) -> tuple[X, State]:
        decrease: Scalar = value - trial_value
        grad_norm: Scalar = jnp.linalg.norm(grad)
        grad_norm_max: Scalar = jnp.linalg.norm(grad, ord=jnp.inf)
        first_value: Scalar = jax.lax.select(
            opt_state.n_steps == 0, value, opt_state.first_value
        )
        first_decrease: Scalar = jax.lax.select(
            opt_state.n_steps == 0, decrease, opt_state.first_decrease
        )
        first_grad_norm: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm, opt_state.first_grad_norm
        )
        first_grad_norm_max: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm_max, opt_state.first_grad_norm_max
        )
        opt_state = attrs.evolve(
            opt_state,
            first_value=first_value,
            first_decrease=first_decrease,
            first_grad_norm=first_grad_norm,
            first_grad_norm_max=first_grad_norm_max,
            alpha=alpha,
            beta=beta,
            decrease=decrease,
            value=trial_value,
            grad=grad,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            params=opt_state.params + alpha * search_direction,
            preconditioner=preconditioner,
            search_direction=search_direction,
            backtracking_steps=backtracking_steps,
            used_steepest_descent_reset=used_steepest_descent_reset,
            force_steepest_descent_next=jnp.asarray(False),
        )
        opt_state = self._detect_stagnation(opt_state, trial_value)
        opt_state = attrs.evolve(opt_state, n_steps=opt_state.n_steps + 1)
        return model_state, opt_state

    def _reject_step[X](
        self,
        *,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        value: Scalar,
        params: Vector,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        search_direction: Vector,
        hess_quad: Scalar,
        backtracking_steps: Integer[Array, ""],
        used_steepest_descent_reset: BooleanNumeric,
    ) -> tuple[X, State]:
        grad_norm: Scalar = jnp.linalg.norm(grad)
        grad_norm_max: Scalar = jnp.linalg.norm(grad, ord=jnp.inf)
        first_value: Scalar = jax.lax.select(
            opt_state.n_steps == 0, value, opt_state.first_value
        )
        first_grad_norm: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm, opt_state.first_grad_norm
        )
        first_grad_norm_max: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm_max, opt_state.first_grad_norm_max
        )
        model_state = objective.update(model_state, params)
        opt_state = attrs.evolve(
            opt_state,
            alpha=jnp.zeros_like(opt_state.alpha),
            beta=jnp.zeros_like(opt_state.beta),
            decrease=jnp.asarray(jnp.inf, dtype=hess_quad.dtype),
            value=value,
            first_value=first_value,
            first_grad_norm=first_grad_norm,
            first_grad_norm_max=first_grad_norm_max,
            grad=grad,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            preconditioner=preconditioner,
            search_direction=search_direction,
            stagnation_counter=jnp.maximum(
                opt_state.stagnation_counter, self.stagnation_patience
            ),
            backtracking_steps=backtracking_steps,
            used_steepest_descent_reset=used_steepest_descent_reset,
            force_steepest_descent_next=jnp.asarray(True),
        )
        opt_state = attrs.evolve(opt_state, n_steps=opt_state.n_steps + 1)
        return model_state, opt_state

    def _step_nan[X](
        self,
        *,
        model_state: X,
        opt_state: State,
        value: Scalar,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        search_direction: Vector,
        hess_quad: Scalar,
        used_steepest_descent_reset: BooleanNumeric,
    ) -> tuple[X, State]:
        nan = jnp.asarray(jnp.nan, dtype=hess_quad.dtype)
        grad_norm: Scalar = jnp.linalg.norm(grad)
        grad_norm_max: Scalar = jnp.linalg.norm(grad, ord=jnp.inf)
        first_value: Scalar = jax.lax.select(
            opt_state.n_steps == 0, value, opt_state.first_value
        )
        first_grad_norm: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm, opt_state.first_grad_norm
        )
        first_grad_norm_max: Scalar = jax.lax.select(
            opt_state.n_steps == 0, grad_norm_max, opt_state.first_grad_norm_max
        )
        opt_state = attrs.evolve(
            opt_state,
            alpha=jnp.zeros_like(opt_state.alpha),
            beta=jnp.zeros_like(opt_state.beta),
            decrease=nan,
            value=value,
            first_value=first_value,
            first_grad_norm=first_grad_norm,
            first_grad_norm_max=first_grad_norm_max,
            grad=grad,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            preconditioner=preconditioner,
            search_direction=search_direction,
            backtracking_steps=jnp.zeros_like(opt_state.backtracking_steps),
            used_steepest_descent_reset=used_steepest_descent_reset,
            force_steepest_descent_next=jnp.asarray(True),
            n_steps=opt_state.n_steps + 1,
        )
        return model_state, opt_state

    @override
    def terminate[X](
        self,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> BooleanNumeric:
        del objective, model_state, opt_stats
        grad_norm: Scalar = jnp.linalg.norm(opt_state.grad)
        return (
            jnp.isnan(opt_state.decrease)
            | jnp.any(jnp.isnan(opt_state.grad))
            | jnp.any(jnp.isnan(opt_state.hess_diag))
            | jnp.isnan(opt_state.hess_quad)
            | (
                jnp.isfinite(opt_state.first_grad_norm)
                & (
                    grad_norm
                    <= self.convergence.target_absolute_gradient_norm
                    + self.convergence.target_relative_gradient_norm
                    * opt_state.first_grad_norm
                )
            )
            | (opt_state.n_steps > self.convergence.maximum_iterations)
            | (opt_state.stagnation_restarts > self.stagnation_max_restarts)
        )

    @override
    def postprocess[X](
        self,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        del objective, model_state
        result: Optimizer.Result = Result.UNKNOWN_ERROR
        grad_norm: Scalar = jnp.linalg.norm(opt_state.grad)
        if (
            jnp.isnan(opt_state.decrease)
            | jnp.any(jnp.isnan(opt_state.grad))
            | jnp.any(jnp.isnan(opt_state.hess_diag))
            | jnp.isnan(opt_state.hess_quad)
        ):
            result = Result.NAN
        elif (
            grad_norm
            <= self.convergence.target_absolute_gradient_norm
            + self.convergence.target_relative_gradient_norm * opt_state.first_grad_norm
        ):
            result = Result.PRIMARY_SUCCESS
        elif (
            grad_norm
            <= self.convergence.acceptable_absolute_gradient_norm
            + self.convergence.acceptable_relative_gradient_norm
            * opt_state.first_grad_norm
        ):
            result = Result.SECONDARY_SUCCESS
        elif opt_state.n_steps > self.convergence.maximum_iterations:
            result = Result.MAX_STEPS_REACHED
        elif opt_state.stagnation_restarts > self.stagnation_max_restarts:
            result = Result.STAGNATION
        opt_stats._end_time = time.perf_counter()  # noqa: SLF001
        return Solution(result=result, state=opt_state, stats=opt_stats)

    def _detect_stagnation(self, state: PNCGState, value: Scalar) -> PNCGState:
        def true_fun(state: PNCGState) -> PNCGState:
            return attrs.evolve(state, stagnation_counter=state.stagnation_counter + 1)

        def false_fun(state: PNCGState) -> PNCGState:
            return attrs.evolve(
                state,
                best_value=value,
                best_params=state.params,
                stagnation_counter=jnp.zeros_like(state.stagnation_counter),
            )

        return jax.lax.cond(value >= state.best_value, true_fun, false_fun, state)

    def _to_result(self, solution: Solution[State, Stats]) -> PNCGResult:
        final_iteration = PNCGIteration(
            parameters=solution.state.params,
            objective_value=solution.state.value,
            best_objective_value=solution.state.best_value,
            gradient_norm=jnp.linalg.norm(solution.state.grad),
            initial_gradient_norm=solution.state.first_grad_norm,
            relative_gradient_norm=_relative_gradient_norm(
                jnp.linalg.norm(solution.state.grad), solution.state.first_grad_norm
            ),
            step_length=solution.state.alpha,
            conjugate_weight=solution.state.beta,
            used_steepest_descent_reset=solution.state.used_steepest_descent_reset,
            regularization_strength=jnp.asarray(
                self.damping.constant_strength, dtype=solution.state.alpha.dtype
            ),
            backtracking_steps=solution.state.backtracking_steps,
            curvature_along_search_direction=solution.state.hess_quad,
        )
        status: Literal[
            "target_converged",
            "acceptable_converged",
            "maximum_iterations_reached",
            "stagnated",
            "numerical_failure",
        ]
        match solution.result:
            case Result.PRIMARY_SUCCESS:
                status = "target_converged"
            case Result.SECONDARY_SUCCESS:
                status = "acceptable_converged"
            case Result.MAX_STEPS_REACHED:
                status = "maximum_iterations_reached"
            case Result.STAGNATION:
                status = "stagnated"
            case _:
                status = "numerical_failure"
        return PNCGResult(
            success=bool(solution.success),
            status=status,
            final_iteration=final_iteration,
            best_parameters=solution.state.best_params,
        )


@jarp.jit(inline=True)
def _relative_gradient_norm(grad_norm: Scalar, first_grad_norm: Scalar) -> Scalar:
    return jnp.where(first_grad_norm > 0.0, grad_norm / first_grad_norm, 0.0)


def _canonicalize_scalar(value: Scalar | float | None) -> float | int | None:
    if value is None:
        return None
    if np.isscalar(value):
        return value.item() if hasattr(value, "item") else value
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return value


def _canonicalize_convergence(
    convergence: ConvergenceCriteria,
) -> ConvergenceCriteria:
    return ConvergenceCriteria(
        target_relative_gradient_norm=_canonicalize_scalar(
            convergence.target_relative_gradient_norm
        ),
        acceptable_relative_gradient_norm=_canonicalize_scalar(
            convergence.acceptable_relative_gradient_norm
        ),
        target_absolute_gradient_norm=_canonicalize_scalar(
            convergence.target_absolute_gradient_norm
        ),
        acceptable_absolute_gradient_norm=_canonicalize_scalar(
            convergence.acceptable_absolute_gradient_norm
        ),
        maximum_iterations=_canonicalize_scalar(convergence.maximum_iterations),
    )


def _canonicalize_line_search(line_search: LineSearchSettings) -> LineSearchSettings:
    return LineSearchSettings(
        armijo_sufficient_decrease=_canonicalize_scalar(
            line_search.armijo_sufficient_decrease
        ),
        backtracking_factor=_canonicalize_scalar(line_search.backtracking_factor),
        maximum_backtracking_steps=_canonicalize_scalar(
            line_search.maximum_backtracking_steps
        ),
        maximum_step_inf_norm=_canonicalize_scalar(line_search.maximum_step_inf_norm),
    )


def _canonicalize_damping(damping: HessianDamping) -> HessianDamping:
    return HessianDamping(
        constant_strength=_canonicalize_scalar(damping.constant_strength)
    )


@jarp.jit(inline=True)
def _steepest_descent_direction(
    grad: Vector, preconditioner: Vector
) -> tuple[Scalar, Vector, Scalar]:
    beta: Scalar = jnp.zeros((), dtype=grad.dtype)
    search_direction: Vector = -preconditioner * grad
    grad_dot_direction: Scalar = jnp.vdot(grad, search_direction)
    return beta, search_direction, grad_dot_direction


@jarp.jit(inline=True)
def _initial_alpha(
    *,
    grad: Vector,
    search_direction: Vector,
    hess_quad: Scalar,
    maximum_step_inf_norm: Scalar | None,
) -> Scalar:
    alpha_quadratic: Scalar = _compute_alpha(grad, search_direction, hess_quad)
    alpha_quadratic = jnp.where(
        alpha_quadratic > 0.0,
        alpha_quadratic,
        jnp.asarray(jnp.inf, alpha_quadratic.dtype),
    )
    alpha_delta: Scalar = jnp.asarray(jnp.inf, dtype=alpha_quadratic.dtype)
    if maximum_step_inf_norm is not None:
        direction_norm: Scalar = jnp.linalg.norm(search_direction, ord=jnp.inf)
        alpha_delta = jnp.where(
            direction_norm > 0.0,
            jnp.asarray(maximum_step_inf_norm, dtype=alpha_quadratic.dtype)
            / direction_norm,
            jnp.asarray(jnp.inf, dtype=alpha_quadratic.dtype),
        )
    alpha: Scalar = jnp.minimum(alpha_quadratic, alpha_delta)
    fallback: Scalar = jnp.minimum(jnp.asarray(1.0, dtype=alpha.dtype), alpha_delta)
    alpha = jnp.where(jnp.isfinite(alpha) & (alpha > 0.0), alpha, fallback)
    alpha = jnp.nan_to_num(alpha, nan=0.0, neginf=0.0, posinf=1.0)
    return jnp.maximum(alpha, 0.0)


def _evaluate_trial[X](
    *,
    objective: LineSearchObjective[X],
    model_state: X,
    params: Vector,
    search_direction: Vector,
    value: Scalar,
    grad_dot_direction: Scalar,
    alpha: Scalar,
    c1: Scalar,
) -> tuple[X, Scalar, BooleanNumeric]:
    trial_params: Vector = params + alpha * search_direction
    model_state = objective.update(model_state, trial_params)
    trial_value: Scalar = objective.fun(model_state)
    accepted: BooleanNumeric = jnp.isfinite(trial_value) & (
        trial_value <= value + c1 * alpha * grad_dot_direction
    )
    return model_state, trial_value, accepted


def _backtracking_line_search[X](
    *,
    objective: LineSearchObjective[X],
    model_state: X,
    params: Vector,
    search_direction: Vector,
    value: Scalar,
    grad_dot_direction: Scalar,
    alpha: Scalar,
    factor: Scalar,
    c1: Scalar,
    max_steps: Integer[Array, ""],
    jit: bool,
) -> tuple[X, Scalar, Scalar, BooleanNumeric, Integer[Array, ""]]:
    max_steps = jnp.asarray(max_steps, dtype=jnp.int32)
    trial_value: Scalar
    accepted: BooleanNumeric
    model_state, trial_value, accepted = _evaluate_trial(
        objective=objective,
        model_state=model_state,
        params=params,
        search_direction=search_direction,
        value=value,
        grad_dot_direction=grad_dot_direction,
        alpha=alpha,
        c1=c1,
    )

    def cond_fun(carry: tuple[X, Scalar, Scalar, BooleanNumeric, Integer[Array, ""]]):
        _model_state, alpha, _trial_value, accepted, steps = carry
        return (~accepted) & (steps < max_steps) & (alpha > 0.0)

    def body_fun(
        carry: tuple[X, Scalar, Scalar, BooleanNumeric, Integer[Array, ""]],
    ) -> tuple[X, Scalar, Scalar, BooleanNumeric, Integer[Array, ""]]:
        model_state, alpha, _trial_value, _accepted, steps = carry
        alpha = alpha * factor
        model_state, trial_value, accepted = _evaluate_trial(
            objective=objective,
            model_state=model_state,
            params=params,
            search_direction=search_direction,
            value=value,
            grad_dot_direction=grad_dot_direction,
            alpha=alpha,
            c1=c1,
        )
        return model_state, alpha, trial_value, accepted, steps + 1

    model_state, alpha, trial_value, accepted, steps = jarp.while_loop(
        cond_fun,
        body_fun,
        (
            model_state,
            alpha,
            trial_value,
            accepted,
            jnp.zeros((), dtype=max_steps.dtype),
        ),
        jit=jit,
    )
    return model_state, alpha, trial_value, accepted, steps


@jarp.jit(inline=True)
def _make_preconditioner(hess_diag: Vector) -> Vector:
    hess_diag = jnp.abs(hess_diag)
    hess_diag_mean: Scalar = jnp.mean(hess_diag, where=hess_diag > 0.0)
    hess_diag = jnp.where(hess_diag > 0.0, hess_diag, hess_diag_mean)
    return jnp.reciprocal(hess_diag)


@jarp.jit(inline=True)
def _compute_alpha(g: Vector, p: Vector, pHp: Scalar) -> Scalar:
    alpha: Scalar = -jnp.vdot(g, p) / pHp
    alpha = jnp.nan_to_num(alpha, nan=0.0)
    return alpha


@jarp.jit(inline=True)
def _compute_dai_kou_beta(
    *, g: Vector, g_prev: Vector, p_prev: Vector, preconditioner: Vector
) -> Scalar:
    y: Vector = g - g_prev
    yTp: Scalar = jnp.vdot(y, p_prev)
    safe_yTp: Scalar = jnp.where(
        jnp.abs(yTp) > jnp.asarray(1e-12, dtype=yTp.dtype),
        yTp,
        jnp.asarray(1.0, dtype=yTp.dtype),
    )
    preconditioned_y: Vector = preconditioner * y
    beta: Scalar = jnp.vdot(g, preconditioned_y) / safe_yTp - (
        jnp.vdot(y, preconditioned_y) / safe_yTp
    ) * (jnp.vdot(p_prev, g) / safe_yTp)
    return jnp.where(
        jnp.abs(yTp) > jnp.asarray(1e-12, dtype=yTp.dtype),
        beta,
        jnp.asarray(jnp.inf, dtype=yTp.dtype),
    )
