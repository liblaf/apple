from collections.abc import Mapping
from typing import TYPE_CHECKING, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from liblaf.apple.struct.dof_map import DofMap
from liblaf.apple.struct.mapping import KeyLike, MappingLike, PyTreeDict


class ArrayDict(PyTreeDict[jax.Array]):
    if TYPE_CHECKING:

        def __init__(self, data: MappingLike | None = None, /) -> None: ...

    @override
    def __setitem__(self, key: KeyLike, value: ArrayLike) -> None:
        value = jnp.asarray(value)
        return super().__setitem__(key, value)

    def __add__(self, other: MappingLike, /) -> Self:
        other = ArrayDict(other)
        result: ArrayDict = type(self)()
        for key in self:
            if key in other:
                result[key] = self[key] + other[key]
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                continue
            result[key] = other[key]
        return result

    def sum(self, dof_map: Mapping[str, DofMap], /, n_dof: int) -> jax.Array:
        x: jax.Array = jnp.zeros((n_dof,))
        for key, value in self.items():
            idx: DofMap = dof_map[key]
            x = idx.add(x, value)
        return x
