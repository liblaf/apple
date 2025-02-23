from collections.abc import Callable

import beartype
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf import apple


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
def hess_as_linear_operator(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
) -> apple.LinearOperator:
    f_hvp: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]] = hvp_fun(fun, x)
    y: jax.ShapeDtypeStruct = jax.eval_shape(f_hvp, x)
    return apple.LinearOperator(dtype=y.dtype, shape=(*y.shape, *x.shape), matvec=f_hvp)


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
    return f_hvp


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def vhp(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    return hvp_fun(fun, x)(v)


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def vhp_fun(
    fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]],
    x: Float[jax.Array, " N"],
) -> Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]]:
    vhpfun: Callable[[Float[jax.Array, " N"]], tuple[Float[jax.Array, " N"]]]
    _primals_out, vhpfun = jax.vjp(jax.grad(fun), x)

    def vhp_fun(v: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        result: Float[jax.Array, " N"]
        (result,) = vhpfun(v)
        return result

    return vhp_fun
