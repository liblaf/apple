import math
from collections.abc import Callable
from typing import override

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf import grapes
from liblaf.apple import utils
from liblaf.apple.optim._abc import Callback, OptimizeResult

from ._abc import Optimizer


class State(flax.struct.PyTreeNode):
    alpha: Float[jax.Array, ""] = flax.struct.field(default=None)
    Delta_E: Float[jax.Array, ""] = flax.struct.field(default=None)
    g: Float[jax.Array, " N"] = flax.struct.field(default=None)
    hess_diag: Float[jax.Array, " N"] = flax.struct.field(default=None)
    hess_quad: Float[jax.Array, ""] = flax.struct.field(default=None)
    p: Float[jax.Array, " N"] = flax.struct.field(default=None)
    P: Float[jax.Array, " N"] = flax.struct.field(default=None)
    x: Float[jax.Array, " N"] = flax.struct.field(default=None)

    first: bool = flax.struct.field(pytree_node=False, default=True)


class PNCG(Optimizer):
    d_hat: float = flax.struct.field(pytree_node=False, default=math.inf)
    maxiter: int = flax.struct.field(pytree_node=False, default=150)
    tol: float = flax.struct.field(pytree_node=False, default=1e-5)

    @override
    def _minimize(
        self,
        fun: Callable[..., Float[jax.Array, ""]],
        x0: Float[ArrayLike, " N"],
        *,
        args: tuple = (),
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        callback: Callback | None = None,
        **kwargs,
    ) -> OptimizeResult:
        assert callable(jac_and_hess_diag)
        assert callable(hess_quad)

        x: Float[jax.Array, " N"] = jnp.asarray(x0)
        state: State = State(x=x, first=True)
        result = OptimizeResult(
            {
                "n_iter": 0,
                "success": False,
                "x": x,
            }
        )
        if callable(callback):
            callback(result)

        Delta_E0: Float[jax.Array, ""] = None  # pyright: ignore[reportAssignmentType]
        timer: grapes.TimedIterable = grapes.timer(range(self.maxiter), name="PNCG")
        for it in timer:
            state = self.step(
                state,
                jac_and_hess_diag=jac_and_hess_diag,
                hess_quad=hess_quad,
                args=args,
            )
            if it == 0:
                Delta_E0 = state.Delta_E
                result["Delta_E0"] = Delta_E0
            result.update(
                {
                    "alpha": state.alpha,
                    "Delta_E": state.Delta_E,
                    "hess_diag": state.hess_diag,
                    "hess_quad": state.hess_quad,
                    "jac": state.g,
                    "n_iter": it + 1,
                    "p": state.p,
                    "P": state.P,
                    "x": state.x,
                }
            )
            if callable(callback):
                callback(result)
            if state.Delta_E < self.tol * Delta_E0:
                result["success"] = True
                break
        timer.timing.finish()
        return result

    @utils.jit
    def calc_beta(
        self,
        g_next: Float[jax.Array, " N"],
        g: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
    ) -> Float[jax.Array, ""]:
        y: Float[jax.Array, " N"] = g_next - g
        yT_p: Float[jax.Array, ""] = jnp.dot(y, p)
        beta: Float[jax.Array, ""] = jnp.dot(g_next, y) / yT_p - (
            jnp.dot(y, y) / yT_p
        ) * (jnp.dot(p, g_next) / yT_p)
        return beta

    @utils.jit
    def calc_Delta_E(
        self,
        g: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        pHp: Float[jax.Array, " N"],
    ) -> Float[jax.Array, ""]:
        return -jnp.dot(g, p) - 0.5 * jnp.dot(p, pHp)

    @utils.jit
    def calc_DK_direction(
        self,
        g: Float[jax.Array, " N"],
        g_prev: Float[jax.Array, " N"],
        P: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
    ) -> Float[jax.Array, " N"]:
        beta: Float[jax.Array, ""] = self.calc_beta(g_next=g, g=g_prev, p=p)
        return -P * g + beta * p

    @utils.jit
    def calc_init_p(
        self, g: Float[jax.Array, " N"], P: Float[jax.Array, " N"]
    ) -> Float[jax.Array, " N"]:
        return -P * g

    @utils.jit
    def calc_p_inf_norm(self, p: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        return jnp.linalg.norm(p, ord=jnp.inf)

    # @utils.jit(static_argnames=("jac_and_hess_diag", "hess_quad", "args"))
    def step(
        self,
        state: State,
        *,
        jac_and_hess_diag: Callable,
        hess_quad: Callable,
        args: tuple,
    ) -> State:
        x: Float[jax.Array, " N"] = state.x
        g: Float[jax.Array, " N"]
        hess_diag: Float[jax.Array, " N"]
        g, hess_diag = jac_and_hess_diag(x, *args)
        P: Float[jax.Array, " N"] = 1.0 / hess_diag
        p: Float[jax.Array, " N"]
        if state.first:
            p = self.calc_init_p(g=g, P=P)
        else:
            p = self.calc_DK_direction(g=g, g_prev=state.g, P=P, p=state.p)
        pHp: Float[jax.Array, ""] = hess_quad(x, p, *args)
        alpha: Float[jax.Array, ""] = jnp.minimum(
            self.d_hat / (2.0 * self.calc_p_inf_norm(p)), -jnp.dot(g, p) / pHp
        )
        x = x + alpha * p
        Delta_E: Float[jax.Array, ""] = -alpha * jnp.dot(g, p) - 0.5 * alpha**2 * pHp
        return state.replace(
            alpha=alpha,
            Delta_E=Delta_E,
            g=g,
            hess_diag=hess_diag,
            hess_quad=pHp,
            p=p,
            P=P,
            x=x,
            first=False,
        )
