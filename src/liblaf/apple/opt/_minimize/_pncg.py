from collections.abc import Callable, Mapping, Sequence
from typing import override

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float
from loguru import logger

from liblaf import apple

from . import MinimizeAlgorithm, MinimizeResult


@apple.register_dataclass()
@attrs.frozen
class MinimizePNCG(MinimizeAlgorithm):
    d_hat: float = jnp.inf  # 1.5
    eps: float = 5e-5
    iter_max: int = 150

    @override
    def _minimize(
        self,
        fun: Callable,
        x0: Float[jax.Array, " N"],
        *,
        args: Sequence | None = None,
        kwargs: Mapping | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        bounds: Sequence | None = None,
        callback: Callable,
    ) -> MinimizeResult:
        # assert fun is not None
        assert jac is not None
        # assert hess_diag is not None
        assert hess_quad is not None
        assert jac_and_hess_diag is not None
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        result = MinimizeResult()
        x: Float[jax.Array, " N"] = x0
        Delta_E0: Float[jax.Array, ""] = jnp.zeros(())
        g: Float[jax.Array, " N"] = jac(x, *args, **kwargs)
        p: Float[jax.Array, " N"] = jnp.zeros_like(x)
        for k in range(self.iter_max):
            # g_next: Float[jax.Array, " N"] = jac(x, *args, **kwargs)
            # H_diag: Float[jax.Array, " N"] = hess_diag(x, *args, **kwargs)
            g_next, H_diag = jac_and_hess_diag(x, *args, **kwargs)
            P_diag: Float[jax.Array, " N"] = 1.0 / H_diag
            if k == 0:
                p = self.compute_initial_p(g, P_diag)
            else:
                p = self.compute_DK_direction(g, g_next, p, P_diag)
            g = g_next
            pHp: Float[jax.Array, ""] = hess_quad(x, p, *args, **kwargs)
            alpha: Float[jax.Array, ""]
            gp: Float[jax.Array, ""]
            alpha, gp = self.line_search_newton(g, p, pHp)
            p_max: Float[jax.Array, ""] = self.compute_p_inf_norm(p)
            if alpha * p_max > 0.5 * self.d_hat:
                alpha = 0.5 * self.d_hat / p_max
            x = self.update_x(alpha, p, x)
            result["x"] = x
            result["jac"] = g
            result["hess_diag"] = H_diag
            result["hess_quad"] = pHp
            Delta_E: Float[jax.Array, ""] = -alpha * gp - 0.5 * alpha**2 * pHp
            result["Delta_E"] = Delta_E
            if Delta_E < 0:
                logger.warning(f"Delta_E = {Delta_E}")
                Delta_E = jnp.zeros(())  # TODO: fix this workaround
            callback(result)
            if k == 0:
                Delta_E0 = Delta_E
            elif Delta_E < self.eps * Delta_E0:
                break
        return result

    @apple.jit(static_argnames=["self"])
    def compute_beta(
        self,
        g_next: Float[jax.Array, " N"],
        g: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        P_next_diag: Float[jax.Array, " N"],
    ) -> Float[jax.Array, ""]:
        y: Float[jax.Array, " N"] = g_next - g
        Py: Float[jax.Array, " N"] = P_next_diag * y
        yp: Float[jax.Array, ""] = jnp.dot(y, p)
        yp += self.eps
        beta: Float[jax.Array, ""] = jnp.dot(g_next, Py) / yp - (
            jnp.dot(y, Py) / yp
        ) * (jnp.dot(p, g_next) / yp)
        return beta

    @apple.jit(static_argnames=["self"])
    def compute_initial_p(
        self, g: Float[jax.Array, " N"], P_diag: Float[jax.Array, " N"]
    ) -> Float[jax.Array, " N"]:
        return -P_diag * g

    @apple.jit(static_argnames=["self"])
    def compute_DK_direction(
        self,
        g: Float[jax.Array, " N"],
        g_next: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        P_diag: Float[jax.Array, " N"],
    ) -> Float[jax.Array, " N"]:
        beta: Float[jax.Array, ""] = self.compute_beta(g_next, g, p, P_diag)
        g: Float[jax.Array, " N"] = g_next
        Pg: Float[jax.Array, " N"] = P_diag * g
        p: Float[jax.Array, " N"] = -Pg + beta * p
        return p

    @apple.jit(static_argnames=["self"])
    def line_search_newton(
        self,
        g: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        pHp: Float[jax.Array, ""],
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, ""]]:
        gp: Float[jax.Array, ""] = jnp.dot(g, p)
        alpha: Float[jax.Array, ""] = -gp / (pHp + self.eps)
        return alpha, gp

    @apple.jit(static_argnames=["self"])
    def compute_p_inf_norm(self, p: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        p = p.reshape(-1, 3)
        p = jnp.linalg.norm(p, axis=1)
        p_inf: Float[jax.Array, ""] = jnp.max(jnp.abs(p))
        return p_inf

    @apple.jit(static_argnames=["self"])
    def update_x(
        self,
        alpha: Float[jax.Array, ""],
        p: Float[jax.Array, " N"],
        x: Float[jax.Array, " N"],
    ) -> Float[jax.Array, " N"]:
        return x + alpha * p
