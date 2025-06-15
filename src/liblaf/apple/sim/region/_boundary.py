from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Integer

from liblaf.apple import struct

from ._region import Region


class RegionBoundary(Region):
    origin: Region = struct.data(default=None)

    @classmethod
    def from_region(cls, region: Region) -> Self:
        self: Self = cls(origin=region, _geometry=region.geometry.boundary)
        return self

    @property
    def original_cell_id(self) -> Integer[jax.Array, " cells"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.geometry.cell_data["cell-id"])

    @property
    def original_point_id(self) -> Integer[jax.Array, " points"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.geometry.point_data["point-id"])
