from collections.abc import Iterator, Mapping
from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct

from ._field import Field


class FieldCollection(Mapping[str, Field], struct.PyTree):
    _data: Mapping[str, Field] = struct.data(default={})

    # region Mapping[str, T]

    @override
    def __getitem__(self, key: struct.KeyLike) -> Field:
        key = struct.as_key(key)
        return self._data[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._data

    @override
    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, T]

    def __add__(self, other: Mapping[str, Field], /) -> Self:
        result: dict[str, Field] = {}
        for key in self:
            if key in other:
                result[key] = self[key].with_values(self[key] + other[key])
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                continue
            result[key] = other[key]
        return self.evolve(_data=result)

    def add(self, key: struct.KeyLike, value: Field) -> Self:
        key = struct.as_key(key)
        new_data: dict[str, Field] = {**self._data, key: value}
        return self.evolve(_data=new_data)

    def select(self, keys: struct.KeysLike, /) -> Self:
        keys = struct.as_keys(keys)
        selected: dict[str, Field] = {key: self._data[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, other: Mapping[str, Field], /) -> Self:
        new_data: dict[str, Field] = {**self._data, **other}
        return self.evolve(_data=new_data)

    def gather(
        self, indices: Mapping[str, struct.Index], /, *, n_dof: int
    ) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, " DoF"] = jnp.zeros((n_dof,))
        for key, field in self.items():
            idx: struct.Index = indices[key]
            x = idx.add(x, field.values)
        return x
