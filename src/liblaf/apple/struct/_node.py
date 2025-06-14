import collections
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, ClassVar, Self, cast

import attrs

from ._field import static
from ._pytree import PyTree

type CollectionLike[T: "Node"] = T | Iterable[T] | Mapping[str, T]


counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: "Node") -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    if type(self).is_view:
        refs_repr: str = ", ".join(ref.id for ref in self.refs)
        id_ += f" -> {refs_repr}"
    return id_


class Node(PyTree):
    id: str = static(default=None)
    is_view: ClassVar[bool | None] = None

    def __attrs_post_init__(self) -> None:
        if self.id is None:
            object.__setattr__(self, "id", uniq_id(self))

    @property
    def deps(self) -> "NodeCollection":
        if self.is_view:
            deps: NodeCollection = NodeCollection()
            for ref in self.refs:
                ref: Node
                deps = deps.update(ref.deps)
            return deps
        return NodeCollection(self)

    @property
    def refs(self) -> "NodeCollection":
        if self.is_view:
            raise NotImplementedError
        return NodeCollection()

    def with_deps(self, deps: CollectionLike["Node"], /) -> Self:
        if self.is_view:
            raise NotImplementedError
        deps = _as_mapping(deps)
        return cast("Self", deps[self.id])


def _as_mapping[T: Node](nodes: CollectionLike[T], /) -> Mapping[str, T]:
    if isinstance(nodes, Mapping):
        nodes = cast("Mapping[str, T]", nodes)
        return nodes
    if isinstance(nodes, Iterable):
        return {node.id: node for node in nodes}
    return {nodes.id: nodes}


class NodeCollection[T: Node](PyTree):
    _data: dict[str, T] = attrs.field(converter=_as_mapping, factory=dict)

    if TYPE_CHECKING:

        def __init__(
            self, data: T | Iterable[T] | Mapping[str, T] | None = None, /
        ) -> None: ...

    def __contains__(self, key: str | T) -> bool:
        key = _as_id(key)
        return key in self._data

    def __iter__(self) -> Iterator[T]:
        yield from self._data.values()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str | Node, /) -> T:
        key = _as_id(key)
        return self._data[key]

    def add(self, node: T) -> Self:
        data: dict[str, T] = self._data.copy()
        data[node.id] = node
        return self.evolve(_data=data)

    def select(self, keys: Iterable[str | Node], /) -> Self:
        keys = (_as_id(key) for key in keys)
        selected: dict[str, T] = {key: self[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, nodes: T | Iterable[T] | Mapping[str, T], /, **kwargs: T) -> Self:
        data: dict[str, T] = self._data.copy()
        data.update(_as_mapping(nodes))
        data.update(kwargs)
        return self.evolve(_data=data)


def _as_id(key: str | Node) -> str:
    if isinstance(key, Node):
        return key.id
    return key
