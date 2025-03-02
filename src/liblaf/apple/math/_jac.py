from collections.abc import Callable

import jax
import pylops
from jaxtyping import Float


def jac_as_operator(
    fun: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]],
    x: Float[jax.Array, " M"],
) -> pylops.LinearOperator:
    jvp: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]] = jvp_fun(fun, x)
    vjp: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " M"]] = vjp_fun(fun, x)
    y: Float[jax.ShapeDtypeStruct, " N"] = jax.eval_shape(jvp, x)
    return pylops.JaxOperator(
        pylops.FunctionOperator(jvp, vjp, y.size, x.size, dtype=y.dtype)
    )


def jvp_fun(
    fun: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]],
    x: Float[jax.Array, " M"],
) -> Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]]:
    lin_fun: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]]
    _primals_out, lin_fun = jax.linearize(fun, x)
    return lin_fun


def vjp_fun(
    fun: Callable[[Float[jax.Array, " M"]], Float[jax.Array, " N"]],
    x: Float[jax.Array, " M"],
) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, " M"]]:
    vjpfun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " M"]]
    _primals_out, vjpfun = jax.vjp(fun, x)
    return vjpfun
