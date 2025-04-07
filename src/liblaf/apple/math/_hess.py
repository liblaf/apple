from collections.abc import Callable, Mapping, Sequence

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.jit()
def vdot_tree[T: PyTree](a: T, b: T) -> Float[jax.Array, ""]:
    a_flat: Float[jax.Array, " N"]
    a_flat, _ = jax.flatten_util.ravel_pytree(a)
    b_flat: Float[jax.Array, " N"]
    b_flat, _ = jax.flatten_util.ravel_pytree(b)
    return jnp.vdot(a_flat, b_flat)


@apple.jit(static_argnames=["fun"])
def hess_diag[T: PyTree](
    fun: Callable[..., Float[jax.Array, ""]],
    x: T,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> T:
    hess_quad_op_: Callable[[T], Float[jax.Array, ""]] = hess_quad_op(
        fun, x, args=args, kwargs=kwargs
    )
    x_flat, unravel_x = jax.flatten_util.ravel_pytree(x)

    def hess_quad_flat(v_flat: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        v = unravel_x(v_flat)
        return hess_quad_op_(v)

    V: Float[jax.Array, "N N"] = jnp.identity(x_flat.size)
    diag: Float[jax.Array, " N"] = jax.vmap(hess_quad_flat)(V)
    return unravel_x(diag)


@apple.jit(static_argnames=["fun"])
def hess_quad[T](
    fun: Callable[..., Float[jax.Array, ""]],
    x: T,
    v: T,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> Float[jax.Array, ""]:
    Hv: T = hvp(fun, x, v, args=args, kwargs=kwargs)
    return vdot_tree(v, Hv)


def hess_quad_op[T: PyTree](
    func: Callable[..., Float[jax.Array, ""]],
    x: T,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> Callable[[T], Float[jax.Array, ""]]:
    hvp_op_: Callable[[T], T] = hvp_op(func, x, args=args, kwargs=kwargs)

    @apple.jit()
    def op(v: T) -> Float[jax.Array, ""]:
        return vdot_tree(v, hvp_op_(v))

    return op


@apple.jit(static_argnames=["fun"])
def hvp[T: PyTree](
    fun: Callable[..., Float[jax.Array, ""]],
    x: T,
    v: T,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> T:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    def f(x: T) -> Float[jax.Array, ""]:
        return fun(x, *args, **kwargs)

    tangents_out: T
    _primals_out, tangents_out = jax.jvp(jax.grad(f), (x,), (v,))
    return tangents_out


def hvp_op[T](
    fun: Callable[..., Float[jax.Array, ""]],
    x: T,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
) -> Callable[[T], T]:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}

    def f(x: T) -> Float[jax.Array, ""]:
        return fun(x, *args, **kwargs)

    op: Callable[[T], T]
    y, op = jax.linearize(jax.grad(f), x)
    op = jax.jit(op)
    return op
