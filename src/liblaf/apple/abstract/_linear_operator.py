from collections.abc import Callable

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float


@attrs.frozen
class LinearOperator:
    dtype: np.dtype
    shape: tuple[int, ...]
    _matvec: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " M"]] = attrs.field(
        alias="matvec"
    )
    _rmatvec: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]] | None = (
        attrs.field(default=None, alias="rmatvec")
    )
    _matmat: Callable[[Float[jax.Array, "N K"]], Float[jax.Array, "M K"]] | None = (
        attrs.field(default=None, alias="matmat")
    )
    _rmatmat: Callable[[Float[jax.Array, "M K"]], Float[jax.Array, "N K"]] | None = (
        attrs.field(default=None, alias="rmatmat")
    )

    def matvec(self, v: Float[jax.Array, " N"]) -> Float[jax.Array, " M"]:
        v = jnp.asarray(v, self.dtype)
        return self._matvec(v)

    def rmatvec(self, v: Float[jax.Array, " M"]) -> Float[jax.Array, " N"]:
        v = jnp.asarray(v, self.dtype)
        if self._rmatvec is not None:
            return self._rmatvec(v)
        return jax.linear_transpose(
            self.matvec, jax.ShapeDtypeStruct((self.shape[1],), self.dtype)
        )(v)[0]

    def matmat(self, V: Float[jax.Array, "N K"]) -> Float[jax.Array, "M K"]:
        V = jnp.asarray(V, self.dtype)
        if self._matmat is not None:
            return self._matmat(V)
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(V)

    def rmatmat(self, V: Float[jax.Array, "M K"]) -> Float[jax.Array, "N K"]:
        V = jnp.asarray(V, self.dtype)
        if self._rmatmat is not None:
            return self._rmatmat(V)
        return jax.vmap(self.rmatvec, in_axes=1, out_axes=1)(V)


def as_linear_operator(A: Float[jax.Array, "M N"]) -> LinearOperator:
    return LinearOperator(dtype=A.dtype, shape=A.shape, matvec=lambda v: A @ v)
