from collections.abc import Sequence
from typing import overload

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Integer

from ._dof_map import DofMap
from ._integer import DofMapInteger
from ._slice import DofMapSlice


@overload
def as_dof_map(dof_map: None) -> None: ...
@overload
def as_dof_map(dof_map: slice | ArrayLike | DofMap) -> DofMap: ...
def as_dof_map(dof_map: slice | ArrayLike | DofMap | None) -> DofMap | None:
    if dof_map is None:
        return None
    if isinstance(dof_map, DofMap):
        return dof_map
    if isinstance(dof_map, slice):
        return DofMapSlice(dof_map)
    return DofMapInteger(jnp.asarray(dof_map))


def make_dof_map(shape: int | Sequence[int], *, offset: int = 0) -> DofMapInteger:
    integers: Integer[jax.Array, " ..."] = jnp.arange(
        offset, np.prod(shape) + offset, dtype=jnp.int32
    ).reshape(shape)
    return DofMapInteger(integers)
