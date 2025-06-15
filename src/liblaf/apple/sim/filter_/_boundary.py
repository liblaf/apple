from typing import Self, override

from liblaf.apple import struct
from liblaf.apple.sim.abc import Field, Object
from liblaf.apple.sim.obj import ObjectTriangle

from ._filter import Filter


class FilterBoundary(Filter):
    _object: Object = struct.data(default=None)
    _result: Object = struct.data(default=None)

    @classmethod
    def from_object(cls, obj: Object) -> Self:
        displacement: Field = obj.displacement.boundary
        velocity: Field | None = None
        if obj.velocity is not None:
            displacement.with_values(
                obj.velocity[displacement.geometry.original_point_id]
            )
        force: Field | None = None
        if obj.force is not None:
            force = displacement.with_values(
                obj.force[displacement.geometry.original_point_id]
            )
        self: Self = cls(_object=obj)
        result: Object = ObjectTriangle(
            displacement=displacement,
            velocity=velocity,
            force=force,
            dof_index=obj.dof_index[obj.displacement.geometry.original_point_id],
            origin=self,
        )
        self = self.evolve(_result=result)
        return self

    @property
    def obj(self) -> Object:
        return self._object

    @property
    @override
    def refs(self) -> struct.NodeCollection:
        return struct.NodeCollection(self._object)

    @property
    @override
    def result(self) -> Object:
        raise self._object.boundary

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        refs = struct.NodeCollection(refs)
        return self.evolve(_object=refs[self.obj.id])
