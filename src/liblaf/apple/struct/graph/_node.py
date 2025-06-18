import abc
import collections
from typing import Any, Self, TypedDict

import attrs

from liblaf.apple.struct._utils import MappingLike
from liblaf.apple.struct.dict_util import FrozenDict
from liblaf.apple.struct.tree_util import PyTree, static


class NodeAttrs(TypedDict, total=False):
    label: str
    shape: str
    size: int


counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: Any) -> str:
    name: str = type(self).__name__
    id_: str = f"{name}-{counter[name]:03d}"
    counter[name] += 1
    return id_


class GraphNode(PyTree):
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True), kw_only=True)

    @property
    @abc.abstractmethod
    def deps(self) -> FrozenDict["GraphNode"]:
        raise NotImplementedError

    @property
    def node_attrs(self) -> NodeAttrs:
        return NodeAttrs(label=self.id, shape="o", size=len(self.id) ** 2 * 60)

    def prepare(self) -> Self:
        return self

    @abc.abstractmethod
    def with_deps(self, deps: MappingLike["GraphNode"], /) -> Self:
        raise NotImplementedError
