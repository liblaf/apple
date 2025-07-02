from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from typing_extensions import TypeIs


def is_array(x: Any, /) -> TypeIs[Array]:
    return eqx.is_array(x)


def is_array_like(x: Any, /) -> TypeIs[ArrayLike]:
    return eqx.is_array_like(x)


def is_scalar(x: Any, /) -> TypeIs[ArrayLike]:
    return is_array_like(x) and jnp.size(x) == 1
