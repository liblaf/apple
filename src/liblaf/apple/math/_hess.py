import abc
from collections.abc import Callable

import beartype
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pylops
import scipy.sparse.linalg
from jaxtyping import Float

from liblaf import apple


class AbstractHessian:
    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype: ...

    @property
    @abc.abstractmethod
    def n(self) -> int: ...

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def as_scipy(self) -> scipy.sparse.linalg.LinearOperator:
        return scipy.sparse.linalg.LinearOperator(
            shape=self.shape,
            matvec=self.matvec,  # pyright: ignore[reportCallIssue]
            rmatvec=self.rmatvec,  # pyright: ignore[reportCallIssue]
            dtype=jnp.float64,
        )

    def as_pylops(self) -> pylops.LinearOperator:
        return pylops.JaxOperator(
            pylops.FunctionOperator(self.matvec, self.matvec, self.n, self.n)
        )

    def matvec(self, v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        v = jnp.asarray(v, dtype=self.dtype)
        return self._matvec(v)

    def rmatvec(self, v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        v = jnp.asarray(v, dtype=self.dtype)
        return self._rmatvec(v)

    @abc.abstractmethod
    def _matvec(self, v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]: ...

    def _rmatvec(self, v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        return self._matvec(v)


@apple.jit(static_argnames=["fun"])
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def hess_diag(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    vs: Float[jax.Array, "N N"] = jnp.identity(x.shape[0], dtype=x.dtype)
    f_hvp: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]] = hvp_fun(fun, x)

    @jaxtyping.jaxtyped(typechecker=beartype.beartype)
    def comp(v: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        return jnp.vdot(v, f_hvp(v))

    return jax.vmap(comp)(vs)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def hess_as_operator(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
) -> pylops.LinearOperator:
    hvp: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]] = hvp_fun(fun, x)
    return pylops.JaxOperator(
        pylops.FunctionOperator(hvp, hvp, x.size, x.size, dtype=x.dtype)
    )


@apple.jit(static_argnames=["fun"])
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def hvp(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    tangents_out: Float[jax.Array, " N"]
    _primals_out, tangents_out = jax.jvp(jax.grad(fun), (x,), (v,))
    return tangents_out


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def hvp_fun(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]]:
    f_hvp: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]]
    _y, f_hvp = jax.linearize(jax.grad(fun), x)
    f_hvp = jax.jit(f_hvp)

    def hvp(v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        v = jnp.asarray(v, dtype=x.dtype)
        return f_hvp(v)

    return hvp
