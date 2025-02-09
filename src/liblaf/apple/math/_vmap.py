from collections.abc import Callable

import jax


def double_vmap(fun: Callable) -> Callable:
    return jax.vmap(jax.vmap(fun))
