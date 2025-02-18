from collections.abc import Callable
from typing import Protocol

import attrs
import jax
import jax.numpy as jnp
import scipy.optimize
from jaxtyping import Float

from . import MinimizeAlgorithm


class AbstractHessian(Protocol):
    def __matmul__(self, other: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def __rmatmul__(self, other: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        raise NotImplementedError

    def diagonal(self) -> Float[jax.Array, " N"]:
        raise NotImplementedError


@attrs.frozen
class MinimizePNCG(MinimizeAlgorithm):
    eps: float = 1e-3
    iter_max: int = 100

    def _minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        *,
        callback: Callable,
    ) -> scipy.optimize.OptimizeResult:
        assert jac
        assert hess
        result = scipy.optimize.OptimizeResult()
        x: Float[jax.Array, " N"] = x0
        Delta_E0: Float[jax.Array, ""] = jnp.asarray(0.0)
        g: Float[jax.Array, " N"] = jac(x)
        p: Float[jax.Array, " N"] = jnp.zeros_like(x)
        for k in range(self.iter_max):
            g_next: Float[jax.Array, " N"] = jac(x)
            H: AbstractHessian = hess(x)
            P_diag: Float[jax.Array, " N"] | None = self.preconditioning(x)
            beta: Float[jax.Array, ""] = (
                jnp.asarray(0.0) if k == 0 else self.compute_beta(g_next, g, p, P_diag)
            )
            g: Float[jax.Array, " N"] = g_next
            Pg: Float[jax.Array, " N"] = g if P_diag is None else P_diag * g
            p: Float[jax.Array, " N"] = -Pg + beta * p
            gp: Float[jax.Array, ""] = jnp.dot(g, p)
            pHp: Float[jax.Array, ""] = jnp.dot(p, H @ p)
            alpha: Float[jax.Array, ""] = -gp / pHp
            x = x + alpha * p
            Delta_E: Float[jax.Array, ""] = -alpha * gp - 0.5 * alpha**2 * pHp
            callback(result)
            if k == 0:
                Delta_E0 = Delta_E
            elif Delta_E < self.eps * Delta_E0:
                break
        return result

    def compute_beta(
        self,
        g_next: Float[jax.Array, " N"],
        g: Float[jax.Array, " N"],
        p: Float[jax.Array, " N"],
        P_next_diag: Float[jax.Array, " N"] | None = None,
    ) -> Float[jax.Array, ""]:
        y: Float[jax.Array, " N"] = g_next - g
        Py: Float[jax.Array, " N"] = y if P_next_diag is None else P_next_diag * y
        yp: Float[jax.Array, ""] = jnp.dot(y, p)
        beta: Float[jax.Array, ""] = jnp.dot(g_next, Py) / yp - (
            jnp.dot(y, Py) / yp
        ) * (jnp.dot(p, g_next) / yp)
        return beta

    def preconditioning(self, H: AbstractHessian) -> Float[jax.Array, " N"] | None:
        try:
            return 1.0 / H.diagonal()
        except NotImplementedError:
            return None
