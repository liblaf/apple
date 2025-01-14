import enum
from collections.abc import Callable

import jax
import jax.numpy as jnp


class HVPMethod(enum.StrEnum):
    GRAD_OF_GRAD = "grad-of-grad"
    FORWARD_OVER_REVERSE = "forward-over-reverse"
    REVERSE_OVER_FORWARD = "reverse-over-forward"
    REVERSE_OVER_REVERSE = "reverse-over-reverse"


def hvp(
    fun: Callable,
    x: jax.Array,
    v: jax.Array,
    *,
    method: HVPMethod = HVPMethod.FORWARD_OVER_REVERSE,
) -> jax.Array:
    """Hessian-vector products.

    References:
        [1] [The Autodiff Cookbook — JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
    """
    match method:
        case HVPMethod.GRAD_OF_GRAD:
            return jax.grad(lambda x: jnp.vdot(jax.grad(fun)(x), v))(x)
        case HVPMethod.FORWARD_OVER_REVERSE:
            return jax.jvp(jax.grad(fun), (x,), (v,))[1]
        case HVPMethod.REVERSE_OVER_FORWARD:
            g = lambda primals: jax.jvp(fun, primals, (v,))[1]  # noqa: E731
            return jax.grad(g)((x,))
        case HVPMethod.REVERSE_OVER_REVERSE:
            return jax.grad(lambda x: jnp.vdot(jax.grad(fun)(x), v))(x)
        case _:
            msg = f"Unknown method '{method}' for Hessians-vector products"
            raise ValueError(msg)
