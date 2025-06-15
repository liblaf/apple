from typing import TYPE_CHECKING, override

from liblaf.apple import struct

if TYPE_CHECKING:
    from liblaf.apple.sim import obj as _o


class Filter(struct.Node):
    @property
    @override
    def networkx_attrs(self) -> struct.NetworkxNodeAttrs:
        attrs: struct.NetworkxNodeAttrs = super().networkx_attrs
        attrs["shape"] = ">"
        return attrs

    @property
    @override
    def refs(self) -> struct.NodeCollection["_o.Object"]:
        raise NotImplementedError

    @property
    def result(self) -> "_o.Object":
        raise NotImplementedError
