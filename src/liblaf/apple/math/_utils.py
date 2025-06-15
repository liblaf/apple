import enum
from collections.abc import Sequence
from typing import Any, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from liblaf.apple import utils


class BroadcastMode(enum.StrEnum):
    LEADING = enum.auto()
    TRAILING = enum.auto()


class EnumError(ValueError):
    cls: type[enum.Enum]
    value: Any

    def __init__(self, value: Any, cls: type[enum.Enum] = enum.Enum) -> None:
        super().__init__(f"{value!r} is not a valid {cls.__qualname__}")
        self.cls = cls
        self.value = value


@utils.jit(static_argnames=("shape", "mode"))
def broadcast_to(
    arr: ArrayLike, shape: Sequence[int], mode: BroadcastMode = BroadcastMode.LEADING
) -> jax.Array:
    arr = jnp.asarray(arr)
    if arr.shape == shape:
        return arr
    if arr.size == np.prod(shape):
        return arr.reshape(shape)
    match mode:
        case BroadcastMode.LEADING:
            return jnp.broadcast_to(arr, shape)
        case BroadcastMode.TRAILING:
            arr = jnp.reshape(arr, arr.shape + (1,) * (len(shape) - arr.ndim))
            return jnp.broadcast_to(arr, shape)
        case _ as unreachable:
            assert_never(unreachable)
