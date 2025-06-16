from typing import Self, override

import jax
from jaxtyping import Integer

from liblaf.apple import struct
from liblaf.apple.sim.abc import Object, Operator, Region


class OperatorBoundary(Operator):
    original_point_id: Integer[jax.Array, " points"] = struct.array(default=None)
    source: Object = struct.data(default=None)

    @classmethod
    def apply(cls, obj: Object) -> Object:
        region: Region = obj.region.boundary
        self: Self = cls(original_point_id=region.original_point_id, source=obj)
        result: Object = Object.from_region(region)
        result = result.evolve(dof_index=obj.dof_index[self.original_point_id], op=self)
        result = self.update(result)
        return result

    @property
    @override
    def deps(self) -> struct.NodeCollection[Object]:
        return struct.NodeCollection(self.source)

    @override
    def update(self, result: Object) -> Object:
        result = result.evolve(op=self)
        result = result.update(
            displacement=self.source.displacement[self.original_point_id],
            velocity=self.source.velocity[self.original_point_id],
            force=self.source.force[self.original_point_id],
        )
        return result

    @override
    def with_deps(self, deps: struct.NodesLike) -> Self:
        deps = struct.NodeCollection(deps)
        return self.evolve(source=deps[self.source])
