import collections
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Self, TypedDict, cast, overload

import attrs
import networkx as nx

from ._field import static
from ._pytree import PyTree

type CollectionLike[T: "Node"] = T | Iterable[T | None] | Mapping[str, T] | None


class NetworkxNodeAttrs(TypedDict, total=False):
    label: str
    shape: str
    size: int


counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: "Node") -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    return id_


class Node(PyTree):
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True), kw_only=True)

    @property
    def networkx_attrs(self) -> NetworkxNodeAttrs:
        return NetworkxNodeAttrs(label=self.id, shape="o", size=len(self.id) ** 2 * 60)

    @property
    def refs(self) -> "NodeCollection":
        return NodeCollection()

    @property
    def refs_recursive(self) -> "NodeCollection":
        refs: NodeCollection = NodeCollection(self)
        for ref in self.refs:
            ref: Node
            refs = refs.update(ref.refs_recursive)
        return refs

    def add_to_graph(self, graph: nx.DiGraph | None = None, /) -> nx.DiGraph:
        if graph is None:
            graph = nx.DiGraph()
        graph.add_node(self.id, **self.networkx_attrs)
        for ref in self.refs:
            ref: Node
            graph = ref.add_to_graph(graph)
            graph.add_edge(ref.id, self.id)
        return graph

    def update(self, refs: CollectionLike, /) -> Self:
        refs = NodeCollection(refs)
        return cast("Self", refs.get(self.id, self))


def _as_mapping[T: Node](nodes: CollectionLike[T], /) -> Mapping[str, T]:
    if nodes is None:
        return {}
    if isinstance(nodes, Mapping):
        nodes = cast("Mapping[str, T]", nodes)
        return nodes
    if isinstance(nodes, Iterable):
        return {node.id: node for node in nodes if node}
    return {nodes.id: nodes}


class NodeCollection[T: Node](PyTree):
    _data: dict[str, T] = attrs.field(converter=_as_mapping, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: CollectionLike[T] = None, /) -> None: ...

    def __contains__(self, key: str | T) -> bool:
        key = _as_id(key)
        return key in self._data

    def __iter__(self) -> Iterator[T]:
        yield from sorted(self._data.values(), key=lambda node: node.id)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str | Node, /) -> T:
        key = _as_id(key)
        return self._data[key]

    def __or__(self, other: CollectionLike, /) -> Self:
        return self.update(other)

    def add(self, node: T) -> Self:
        data: dict[str, T] = self._data.copy()
        data[node.id] = node
        return self.evolve(_data=data)

    @overload
    def get(self, key: str | Node, /) -> T | None: ...
    @overload
    def get(self, key: str | Node, default: T, /) -> T: ...
    def get(self, key: str | Node, default: T | None = None, /) -> T | None:
        key = _as_id(key)
        return self._data.get(key, default)

    def select(self, keys: Iterable[str | Node], /) -> Self:
        keys = (_as_id(key) for key in keys)
        selected: dict[str, T] = {key: self[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, nodes: CollectionLike, /, **kwargs: T) -> Self:
        data: dict[str, T] = self._data.copy()
        data.update(_as_mapping(nodes))
        data.update(kwargs)
        return self.evolve(_data=data)

    # region Graph

    @property
    def nodes(self) -> Self:
        nodes: Self = self
        for node in self:
            nodes = nodes.update(node.refs_recursive)
        return nodes

    def add_to_graph(self, graph: nx.DiGraph | None = None, /) -> nx.DiGraph:
        if graph is None:
            graph = nx.DiGraph()
        for node in self:
            graph = node.add_to_graph(graph)
        return graph

    def rebuild(self, nodes: CollectionLike) -> Self:
        nodes: Self = self.evolve(_data=nodes)
        graph: nx.DiGraph = self.add_to_graph()
        for n_id in nx.lexicographical_topological_sort(graph):
            node: T = nodes[n_id]
            nodes.add(node.update(nodes))
        return self.evolve(_data=nodes)

    # endregion Graph


def _as_id(key: str | Node) -> str:
    if isinstance(key, Node):
        return key.id
    return key
