from typing import Self, override

from liblaf.apple import struct
from liblaf.apple.sim import field as _f


class Object(struct.Node):
    displacement: _f.Field = struct.data(default=None)
    velocity: _f.Field = struct.data(default=None)
    force: _f.Field = struct.data(default=None)

    origin: "Filter" = struct.data(default=None)

    @property
    @override
    def refs(self) -> struct.NodeCollection["Filter"]:
        return struct.NodeCollection(self.origin)

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        refs = struct.NodeCollection(refs)
        if self.origin is None:
            return refs[self.id].result
        return refs[self.id]


class Filter(struct.Node):
    @property
    def result(self) -> Object:
        raise NotImplementedError

    @property
    @override
    def refs(self) -> struct.NodeCollection[Object]:
        raise NotImplementedError

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        refs = struct.NodeCollection(refs)
        if self.result is None:
            return refs[self.id]
        return refs[self.id].origin
