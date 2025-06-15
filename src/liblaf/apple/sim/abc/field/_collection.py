from collections.abc import Mapping
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct

from ._field import Field


class FieldCollection(struct.FrozenDict[str, Field]):
    def __add__(self, other: Mapping[str, Field], /) -> Self:
        result: dict[str, Field] = {}
        for key in self:
            if key in other:
                result[key] = self[key] + other[key]
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                continue
            result[key] = other[key]
        return self.evolve(_fields=result)

    def gather(
        self, index: Mapping[str, struct.Index], /, *, n_dof: int
    ) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, " DoF"] = jnp.zeros((n_dof,))
        for key, field in self.items():
            idx: struct.Index = index[key]
            x = idx.add(x, field.values)
        return x
