from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct

from ._field import Field


class FieldCollection(struct.PyTreeDict[Field]):
    def __add__(self, other: Mapping[str, Field], /) -> Self:
        result: Self = type(self)()
        for key in self:
            if key in other:
                result[key] = self[key].with_values(self[key] + other[key])
            else:
                result[key] = self[key]
        for key in other:
            if key in self:
                continue
            result[key] = other[key]
        return result

    @eqx.filter_jit
    def sum(
        self, indices: Mapping[str, struct.DofMap], /, *, n_dof: int
    ) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, " DoF"] = jnp.zeros((n_dof,))
        for key, field in self.items():
            idx: struct.DofMap = indices[key]
            x = idx.add(x, field.values)
        return x
