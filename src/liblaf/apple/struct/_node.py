import collections
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Self, cast, overload

import attrs

from ._pytree import PyTree, static

type CollectionLike[T: "Node"] = T | Iterable[T | None] | Mapping[str, T] | None


counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: "Node") -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    return id_


class Node(PyTree):
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True), kw_only=True)


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

    def keys(self) -> Iterator[str]:
        yield from self._data.keys()

    def select(self, keys: Iterable[str | Node], /) -> Self:
        keys = (_as_id(key) for key in keys)
        selected: dict[str, T] = {key: self[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, nodes: CollectionLike, /, **kwargs: T) -> Self:
        data: dict[str, T] = self._data.copy()
        data.update(_as_mapping(nodes))
        data.update(kwargs)
        return self.evolve(_data=data)


def _as_id(key: str | Node) -> str:
    if isinstance(key, Node):
        return key.id
    return key
