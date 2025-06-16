from collections.abc import Generator
from typing import override

import attrs
import networkx as nx

from ._collection import NodeCollection
from ._mixin import NodeCollectionMixin
from ._node import Node
from ._utils import KeyLike, as_key


@attrs.define
class Graph(NodeCollectionMixin):
    _graph: nx.DiGraph = attrs.field(factory=nx.DiGraph)

    # region Mapping[str, Node]

    @override
    def __getitem__(self, key: KeyLike) -> Node:
        key = as_key(key)
        return self._graph.nodes[key]["node"]

    @override
    def __iter__(self) -> Generator[str]:
        yield from self._graph

    @override
    def __len__(self) -> int:
        return len(self._graph)

    # endregion Mapping[str, Node]

    @property
    def bases(self) -> NodeCollection:
        return NodeCollection(
            self[key] for key, in_degree in self._graph.in_degree if in_degree == 0
        )

    @property
    def topological(self) -> Generator[str]:
        yield from nx.topological_sort(self._graph)

    def add(self, node: Node, /) -> None:
        self._graph.add_node(node.id, node=node, **node.display_attrs)
        for dep in node.deps.values():
            dep: Node
            self.add(dep)
            self._graph.add_edge(dep.id, node.id)
