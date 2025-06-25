from typing import Any

import equinox as eqx
import jax
from jaxtyping import ArrayLike
from typing_extensions import TypeIs


def is_array(x: Any, /) -> TypeIs[jax.Array]:
    return eqx.is_array(x)


def is_array_like(x: Any, /) -> TypeIs[ArrayLike]:
    return eqx.is_array_like(x)
