from typing import TYPE_CHECKING, Self, override

from liblaf.apple import struct

if TYPE_CHECKING:
    from ._object import Object


class Filter(struct.Node):
    @property
    @override
    def networkx_attrs(self) -> struct.NetworkxNodeAttrs:
        attrs: struct.NetworkxNodeAttrs = super().networkx_attrs
        attrs["shape"] = ">"
        return attrs

    @property
    def refs(self) -> struct.NodeCollection["Object"]:
        raise NotImplementedError

    @property
    def result(self) -> "Object":
        raise NotImplementedError

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        raise NotImplementedError
