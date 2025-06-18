from collections.abc import Generator, Iterator
from typing import Self, override

import attrs
import networkx as nx

from liblaf.apple.struct._utils import KeyLike, as_key
from liblaf.apple.struct.dict_util import MappingMixin

from ._node import GraphNode


@attrs.define
class Graph(MappingMixin[GraphNode]):
    _graph: nx.DiGraph = attrs.field(factory=nx.DiGraph)

    # region MappingMixin[GraphNode]

    @override
    def __getitem__(self, key: KeyLike) -> GraphNode:
        key = as_key(key)
        return self._graph.nodes[key]["node"]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._graph.nodes

    @override
    def __len__(self) -> int:
        return len(self._graph.nodes)

    # endregion MappingMixin[GraphNode]

    @property
    def bases(self) -> Generator[str]:
        for key in self.keys():
            if self._graph.in_degree(key) == 0:
                yield key

    @property
    def topological(self) -> Generator[str]:
        return nx.topological_sort(self._graph)

    def add(self, node: GraphNode, /) -> Self:
        self._graph.add_node(node.id, node=node, **node.node_attrs)
        for dep in node.deps.values():
            dep: GraphNode
            self.add(dep)
            self._graph.add_edge(dep.id, node.id)
        return self
