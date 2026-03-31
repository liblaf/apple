import time
from typing import Protocol, override

import jarp
import jax
import jax.numpy as jnp
import liblaf.peach.optim as peach_optim
import numpy as np
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Optimizer, Result, Solution
from liblaf.peach.optim.base import SupportsFun
from liblaf.peach.optim.pncg import PNCGObjective
from liblaf.peach.optim.pncg._pncg import _compute_alpha, _make_preconditioner

type BooleanNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


class LineSearchObjective[X](PNCGObjective[X], SupportsFun[X], Protocol): ...


@jarp.define(kw_only=True)
class PNCG(peach_optim.PNCG):
    from liblaf.peach.optim.pncg import PNCGState as State
    from liblaf.peach.optim.pncg import PNCGStats as Stats

    type Callback[X] = Optimizer.Callback[X, PNCG.State, PNCG.Stats]
    type Solution = Optimizer.Solution[State, Stats]

    line_search_factor: Scalar = jarp.array(default=0.5)
    line_search_c1: Scalar = jarp.array(default=1e-4)
    max_line_search_steps: Integer[Array, ""] = jarp.array(default=20)
    reset_beta_on_non_descent: bool = jarp.static(default=True)

    @override
    def step[X](
        self, objective: LineSearchObjective[X], model_state: X, opt_state: State
    ) -> tuple[X, State]:
        model_state = objective.update(model_state, opt_state.params)
        value: Scalar = objective.fun(model_state)
        grad: Vector = objective.grad(model_state)
        hess_diag: Vector = objective.hess_diag(model_state)
        preconditioner: Vector = _make_preconditioner(hess_diag)

        beta: Scalar
        beta, opt_state = self._compute_beta(
            grad=grad, preconditioner=preconditioner, state=opt_state
        )
        search_direction: Vector = (
            -preconditioner * grad + beta * opt_state.search_direction
        )
        grad_dot_direction: Scalar = jnp.vdot(grad, search_direction)
        if self.reset_beta_on_non_descent:
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
                beta, search_direction, grad_dot_direction = (
                    _steepest_descent_direction(grad, preconditioner)
                )

        hess_quad: Scalar = objective.hess_quad(model_state, search_direction)
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
                ),
                lambda _: self._step_nan(
                    model_state=model_state,
                    opt_state=opt_state,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    search_direction=search_direction,
                    hess_quad=hess_quad,
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
            )
        return self._step_nan(
            model_state=model_state,
            opt_state=opt_state,
            grad=grad,
            hess_diag=hess_diag,
            preconditioner=preconditioner,
            search_direction=search_direction,
            hess_quad=hess_quad,
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
    ) -> tuple[X, State]:
        alpha: Scalar = _initial_alpha(
            grad=grad,
            search_direction=search_direction,
            hess_quad=hess_quad,
            max_delta=self.max_delta,
        )
        trial_state: X
        trial_value: Scalar
        accepted: BooleanNumeric
        trial_state, alpha, trial_value, accepted = _backtracking_line_search(
            objective=objective,
            model_state=model_state,
            params=opt_state.params,
            search_direction=search_direction,
            value=value,
            grad_dot_direction=grad_dot_direction,
            alpha=alpha,
            factor=self.line_search_factor,
            c1=self.line_search_c1,
            max_steps=self.max_line_search_steps,
            jit=self.jit,
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
                ),
                lambda _: self._reject_step(
                    objective=objective,
                    model_state=trial_state,
                    opt_state=opt_state,
                    params=opt_state.params,
                    grad=grad,
                    hess_diag=hess_diag,
                    preconditioner=preconditioner,
                    search_direction=search_direction,
                    hess_quad=hess_quad,
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
            )
        return self._reject_step(
            objective=objective,
            model_state=trial_state,
            opt_state=opt_state,
            params=opt_state.params,
            grad=grad,
            hess_diag=hess_diag,
            preconditioner=preconditioner,
            search_direction=search_direction,
            hess_quad=hess_quad,
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
    ) -> tuple[X, State]:
        decrease: Scalar = value - trial_value
        opt_state.first_decrease = jax.lax.select(
            opt_state.n_steps == 0, decrease, opt_state.first_decrease
        )
        opt_state.alpha = alpha
        opt_state.beta = beta
        opt_state.decrease = decrease
        opt_state.grad = grad
        opt_state.hess_diag = hess_diag
        opt_state.hess_quad = hess_quad
        opt_state.params += alpha * search_direction
        opt_state.preconditioner = preconditioner
        opt_state.search_direction = search_direction
        opt_state = self._detect_stagnation(opt_state)
        opt_state.n_steps += 1
        return model_state, opt_state

    def _reject_step[X](
        self,
        *,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        params: Vector,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        search_direction: Vector,
        hess_quad: Scalar,
    ) -> tuple[X, State]:
        model_state = objective.update(model_state, params)
        opt_state.alpha = jnp.zeros_like(opt_state.alpha)
        opt_state.beta = jnp.zeros_like(opt_state.beta)
        opt_state.decrease = jnp.asarray(jnp.inf, dtype=hess_quad.dtype)
        opt_state.grad = grad
        opt_state.hess_diag = hess_diag
        opt_state.hess_quad = hess_quad
        opt_state.preconditioner = preconditioner
        opt_state.search_direction = search_direction
        opt_state.stagnation_counter = jnp.maximum(
            opt_state.stagnation_counter, self.stagnation_patience
        )
        opt_state.n_steps += 1
        return model_state, opt_state

    def _step_nan[X](
        self,
        *,
        model_state: X,
        opt_state: State,
        grad: Vector,
        hess_diag: Vector,
        preconditioner: Vector,
        search_direction: Vector,
        hess_quad: Scalar,
    ) -> tuple[X, State]:
        nan = jnp.asarray(jnp.nan, dtype=hess_quad.dtype)
        opt_state.alpha = jnp.zeros_like(opt_state.alpha)
        opt_state.beta = jnp.zeros_like(opt_state.beta)
        opt_state.decrease = nan
        opt_state.grad = grad
        opt_state.hess_diag = hess_diag
        opt_state.hess_quad = hess_quad
        opt_state.preconditioner = preconditioner
        opt_state.search_direction = search_direction
        opt_state.n_steps += 1
        return model_state, opt_state

    @override
    def terminate[X](
        self,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> BooleanNumeric:
        return (
            jnp.isnan(opt_state.decrease)
            | jnp.any(jnp.isnan(opt_state.grad))
            | jnp.any(jnp.isnan(opt_state.hess_diag))
            | jnp.isnan(opt_state.hess_quad)
            | super().terminate(objective, model_state, opt_state, opt_stats)
        )

    @override
    def postprocess[X](
        self,
        objective: LineSearchObjective[X],
        model_state: X,
        opt_state: State,
        opt_stats: Stats,
    ) -> Solution:
        result: Optimizer.Result = Result.UNKNOWN_ERROR
        if (
            jnp.isnan(opt_state.decrease)
            | jnp.any(jnp.isnan(opt_state.grad))
            | jnp.any(jnp.isnan(opt_state.hess_diag))
            | jnp.isnan(opt_state.hess_quad)
        ):
            result = Result.NAN
        elif (
            opt_state.best_decrease
            <= self.atol_primary + self.rtol_primary * opt_state.first_decrease
        ):
            result = Result.PRIMARY_SUCCESS
        elif opt_state.decrease <= self.atol + self.rtol * opt_state.first_decrease:
            result = Result.SECONDARY_SUCCESS
        elif opt_state.n_steps > self.max_steps:
            result = Result.MAX_STEPS_REACHED
        elif opt_state.stagnation_restarts > self.stagnation_max_restarts:
            result = Result.STAGNATION
        opt_stats._end_time = time.perf_counter()  # noqa: SLF001
        return Solution(result=result, state=opt_state, stats=opt_stats)


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
    max_delta: Scalar,
) -> Scalar:
    alpha_quadratic: Scalar = _compute_alpha(grad, search_direction, hess_quad)
    alpha_quadratic = jnp.where(
        alpha_quadratic > 0.0,
        alpha_quadratic,
        jnp.asarray(jnp.inf, alpha_quadratic.dtype),
    )
    direction_norm: Scalar = jnp.linalg.norm(search_direction, ord=jnp.inf)
    alpha_delta: Scalar = jnp.where(
        direction_norm > 0.0,
        max_delta / direction_norm,
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
) -> tuple[X, Scalar, Scalar, BooleanNumeric]:
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

    model_state, alpha, trial_value, accepted, _ = jarp.while_loop(
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
    return model_state, alpha, trial_value, accepted
