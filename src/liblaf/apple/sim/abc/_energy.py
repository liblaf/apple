from typing import Self, override

from liblaf.apple import math, struct

from ._object import Object


class Energy(struct.DerivativeMixin, struct.Node):
    dof_index: math.Index = struct.data(default=math.make_index())

    # region Graph

    @property
    @override
    def networkx_attrs(self) -> struct.NetworkxNodeAttrs:
        attrs: struct.NetworkxNodeAttrs = super().networkx_attrs
        attrs["shape"] = "s"
        return attrs

    @property
    @override
    def refs(self) -> struct.NodeCollection[Object]:
        raise NotImplementedError

    @override
    def update(self, refs: struct.CollectionLike, /) -> Self:
        raise NotImplementedError

    # endregion Graph
