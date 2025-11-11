from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from liblaf.peach import tree

type Vector = Float[Array, " I"]
type Scalar = Float[Array, ""]


def hess_diag(
    func: Callable[..., Scalar],
    x: PyTree,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = {},
) -> PyTree:
    x_flat: Vector
    unflatten: tree.Unflatten[PyTree]
    x_flat, unflatten = tree.flatten(x)

    def compute(v_flat: Vector) -> Vector:
        v: PyTree = unflatten(v_flat)
        Hv: PyTree = hess_prod(func, x, v, args, kwargs)
        return jnp.vdot(v, Hv)

    vs: Float[Array, "I I"] = jnp.identity(x_flat.size)
    diag_flat: Vector = jax.vmap(compute)(vs)
    diag: PyTree = unflatten(diag_flat)
    return diag


def hess_prod(
    func: Callable[..., Scalar],
    x: PyTree,
    p: PyTree,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = {},
) -> PyTree:
    x_flat: Vector
    unflatten: tree.Unflatten[PyTree]
    x_flat, unflatten = tree.flatten(x)
    p_flat: Vector
    p_flat, _ = tree.flatten(p)

    def fun(x: Vector) -> Scalar:
        return func(unflatten(x), *args, **kwargs)

    prod_flat: Vector
    _, prod_flat = jax.jvp(jax.grad(fun), (x_flat,), (p_flat,))
    prod: PyTree = unflatten(prod_flat)
    return prod
