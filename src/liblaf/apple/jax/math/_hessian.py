from collections.abc import Callable, Mapping

import jax
from jaxtyping import Float, PyTree


def hessp[T: PyTree](
    fun: Callable[..., Float[jax.Array, ""]],
    x: T,
    v: T,
    args: tuple = (),
    kwargs: Mapping = {},
) -> T:
    """.

    References:
        [1]: [Advanced automatic differentiation — JAX documentation](https://docs.jax.dev/en/latest/advanced-autodiff.html#hessian-vector-products-using-both-forward-and-reverse-mode)
    """

    def f(x: T) -> Float[jax.Array, ""]:
        return fun(x, *args, **kwargs)

    tangents_out: T
    _primals_out, tangents_out = jax.jvp(jax.grad(f), (x,), (v,))
    return tangents_out
