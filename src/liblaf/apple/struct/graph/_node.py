import collections
from typing import TYPE_CHECKING, Any, Self, TypedDict

import attrs

from liblaf.apple.struct._pytree import PyTree, static

from ._utils import NodesLike

if TYPE_CHECKING:
    from ._collection import NodeCollection


counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: Any) -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    return id_


class DisplayAttrs(TypedDict, total=False):
    label: str
    shape: str
    size: int


class Node(PyTree):
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True), kw_only=True)

    @property
    def deps(self) -> "NodeCollection":
        raise NotImplementedError

    @property
    def display_attrs(self) -> DisplayAttrs:
        return DisplayAttrs(label=self.id, shape="o", size=len(self.id) ** 2 * 60)

    def with_deps(self, deps: NodesLike, /) -> Self:
        raise NotImplementedError
