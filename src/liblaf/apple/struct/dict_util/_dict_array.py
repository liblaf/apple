from typing import Self, override

import jax
import jax.numpy as jnp

from liblaf.apple.struct import tree_util
from liblaf.apple.struct._utils import MappingLike, as_dict

from ._frozen_dict import FrozenDict


def _as_dict_array(mapping: MappingLike = None) -> dict[str, jax.Array]:
    mapping = as_dict(mapping)
    mapping = {k: jnp.asarray(v) for k, v in mapping.items()}
    return mapping


class DictArray(FrozenDict[jax.Array]):
    _data: dict[str, jax.Array] = tree_util.mapping(factory=_as_dict_array)

    @override
    def copy(self, add_or_replace: MappingLike = None, /) -> Self:
        add_or_replace = _as_dict_array(add_or_replace)
        return super().copy(add_or_replace)

    def __add__(self, other: MappingLike) -> Self:
        other: dict[str, jax.Array] = _as_dict_array(other)
        result: Self = self.copy()
        for key, value in other.items():
            if key in self._data:
                result._data[key] += value
            else:
                result._data[key] = value
        return result
