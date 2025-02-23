from collections.abc import Callable

import jax
from jaxtyping import Float

from liblaf import apple


def jvp_fun(
    fun: Callable[[Float[jax.Array, "..."]], Float[jax.Array, "..."]],
    x: Float[jax.Array, "..."],
) -> Callable[[Float[jax.Array, "..."]], Float[jax.Array, "..."]]:
    lin_fun: Callable[[Float[jax.Array, " ..."]], Float[jax.Array, " ..."]]
    _primals_out, lin_fun = jax.linearize(fun, x)
    return lin_fun


def jac_as_operator(
    fun: Callable[[Float[jax.Array, "..."]], Float[jax.Array, "..."]],
    x: Float[jax.Array, "..."],
) -> apple.LinearOperator:
    lin_fun: Callable[[Float[jax.Array, " ..."]], Float[jax.Array, " ..."]] = jvp_fun(
        fun, x
    )
    y: jax.ShapeDtypeStruct = jax.eval_shape(lin_fun, x)
    return apple.LinearOperator(
        dtype=y.dtype, shape=(*y.shape, *x.shape), matvec=lin_fun
    )
