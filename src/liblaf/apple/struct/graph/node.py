import abc
import collections
from typing import Self, TypedDict

from liblaf.apple.struct.mapping import MappingLike, PyTreeDict
from liblaf.apple.struct.pytree import PyTree, static


class NodeAttrs(TypedDict, total=False):
    label: str
    shape: str
    size: int


class GraphNode(PyTree):
    id: str = static(default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.id is None:
            prefix: str = type(self).__qualname__
            self.id = f"{prefix}-{counter[prefix]:03d}"
            counter[prefix] += 1

    @property
    @abc.abstractmethod
    def deps(self) -> PyTreeDict["GraphNode"]:
        raise NotImplementedError

    @property
    def node_attrs(self) -> NodeAttrs:
        return NodeAttrs(label=self.id, shape="o", size=len(self.id) ** 2 * 60)

    def prepare(self) -> Self:
        return self

    @abc.abstractmethod
    def with_deps(self, deps: MappingLike["GraphNode"], /) -> Self:
        raise NotImplementedError


counter: collections.Counter[str] = collections.Counter()
