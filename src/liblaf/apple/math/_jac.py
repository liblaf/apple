from collections.abc import Callable, Mapping, Sequence

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


def jvp(
    fun: Callable[..., Float[jax.Array, " N"]],
    x: Float[jax.Array, " M"],
    v: Float[jax.Array, " M"],
    *,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> Float[jax.Array, " N"]:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    tangents_out: Float[jax.Array, " N"]
    _primals_out, tangents_out = jax.jvp(lambda x: fun(x, *args, **kwargs), (x,), (v,))
    return tangents_out


def vjp(
    fun: Callable[..., Float[jax.Array, " N"]],
    x: Float[jax.Array, " M"],
    v: Float[jax.Array, " N"],
    *,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> Float[jax.Array, " M"]:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    vjpfun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " M"]]
    _primals_out, vjpfun = jax.vjp(lambda x: fun(x, *args, **kwargs), (x,))
    vjp: Float[jax.Array, " M"]
    (vjp,) = vjpfun(v)
    return vjp


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

    def vjpfun_(y: Float[jax.Array, " N"]) -> Float[jax.Array, " M"]:
        return vjpfun(y)[0]

    return vjpfun_
