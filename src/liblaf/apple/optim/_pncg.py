import math
from collections.abc import Callable
from typing import override

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import utils
from liblaf.apple.optim._abc import Callback, OptimizeResult

from ._abc import Optimizer


class PNCG(flax.struct.PyTreeNode, Optimizer):
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

        result = OptimizeResult(success=False)

        x: Float[jax.Array, " N"] = jnp.asarray(x0)
        g: Float[jax.Array, " N"] = jnp.zeros_like(x)
        p: Float[jax.Array, " N"] = jnp.zeros_like(x)
        Delta_E0: Float[jax.Array, ""] = jnp.asarray(0.0)
        P_next: Float[jax.Array, " N"] = jnp.zeros_like(x)

        result.update(
            {
                "Delta_E": jnp.asarray(0.0),
                "hess_diag": P_next,
                "hess_quad": jnp.asarray(0.0),
                "jac": g,
                "n_iter": 0,
                "p": p,
                "x": x,
            }
        )
        if callable(callback):
            callback(result)

        for it in range(self.maxiter):
            g_next, P_next = jac_and_hess_diag(x, *args)
            if it == 0:
                p = self.calc_init_p(g=g_next, P=P_next)
            else:
                p = self.calc_DK_direction(g=g_next, g_prev=g, P=P_next, p=p)
            pHp: Float[jax.Array, ""] = hess_quad(x, p, *args)
            alpha: Float[jax.Array, ""] = jnp.minimum(
                self.d_hat / (2.0 * self.calc_p_inf_norm(p)),
                -jnp.dot(g_next, p) / pHp,
            )
            x: Float[jax.Array, " N"] = x + alpha * p
            Delta_E: Float[jax.Array, ""] = (
                -alpha * jnp.dot(g_next, p) - 0.5 * alpha**2 * pHp
            )
            if it == 0:
                Delta_E0: Float[jax.Array, ""] = Delta_E
            result.update(
                {
                    "Delta_E": Delta_E,
                    "hess_diag": P_next,
                    "hess_quad": pHp,
                    "jac": g_next,
                    "n_iter": it + 1,
                    "p": p,
                    "x": x,
                }
            )
            if callable(callback):
                callback(result)
            if Delta_E < self.tol * Delta_E0:
                result.update({"success": True})
                break
            g = g_next
        return result

    @utils.jit
    def calc_beta(
        self,
        g_next: Float[jax.Array, " N"],
        g: Float[jax.Array, " N"],
        # p_next: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
    ) -> Float[jax.Array, ""]:
        y: Float[jax.Array, " N"] = g_next - g
        yT_p: Float[jax.Array, ""] = jnp.dot(y, p)
        beta: Float[jax.Array, ""] = jnp.dot(g_next, y) / yT_p - (
            jnp.dot(y, y) / yT_p
        ) * (jnp.dot(p, g_next) / yT_p)
        return beta

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
