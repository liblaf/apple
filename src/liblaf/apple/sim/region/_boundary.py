from typing import ClassVar, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Integer

from liblaf.apple import struct

from ._region import Region


class RegionBoundary(Region):
    is_view: ClassVar[bool] = True
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

    @property
    @override
    def refs(self) -> struct.NodeCollection["Region"]:
        return struct.NodeCollection(self.origin)

    @override
    def with_deps(self, deps: struct.CollectionLike) -> Self:
        origin: Region = self.origin.with_deps(deps)
        return self.from_region(origin)
