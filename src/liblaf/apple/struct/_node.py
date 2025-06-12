import collections
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self, cast

import attrs

from ._pytree import PyTree, class_var, data, static

counter = collections.Counter()


def uniq_id(self: "Node") -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    if self.refs:
        refs_repr: str = ", ".join(ref.id for ref in self.refs)
        id_ += f" -> {refs_repr}"
    return id_


def validate_refs(self: "Node", attribute: attrs.Attribute, value: Any) -> None:
    validator = (
        attrs.validators.min_len(1) if self.is_view else attrs.validators.max_len(0)
    )
    validator(self, attribute, value)


class Node(PyTree):
    is_view: bool = class_var(default=False, init=False)
    refs: tuple["Node", ...] = data(default=(), validator=validate_refs)
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True))

    @property
    def deps(self) -> "NodeCollection":
        if self.is_view:
            deps = NodeCollection()
            for ref in self.refs:
                deps.update(ref.deps)
            return deps
        return NodeCollection(self)

    @property
    def ref(self) -> "Node":
        return self.refs[0]

    def with_deps(self, deps: Iterable["Node"] | Mapping[str, "Node"], /) -> Self:
        deps = _as_mapping(deps)
        if not self.is_view:
            return cast("Self", deps[self.id])
        refs: Sequence[Node] = tuple(ref.with_deps(deps) for ref in self.refs)
        return self.evolve(refs=refs)


def _as_mapping[T: Node](nodes: T | Iterable[T] | Mapping[str, T]) -> Mapping[str, T]:
    if isinstance(nodes, Mapping):
        nodes = cast("Mapping[str, T]", nodes)
        return nodes
    if isinstance(nodes, Iterable):
        return {node.id: node for node in nodes}
    return {nodes.id: nodes}


@attrs.frozen
class NodeCollection[T: Node](PyTree):
    data: dict[str, T] = attrs.field(converter=_as_mapping, factory=dict)

    if TYPE_CHECKING:

        def __init__(
            self, data: T | Iterable[T] | Mapping[str, T] | None = None, /
        ) -> None: ...

    def __contains__(self, key: str | T) -> bool:
        key = _as_id(key)
        return key in self.data

    def __iter__(self) -> Iterator[T]:
        yield from self.data.values()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key: str | Node, /) -> T:
        key = _as_id(key)
        return self.data[key]

    def add(self, node: T) -> Self:
        data: dict[str, T] = self.data.copy()
        data[node.id] = node
        return type(self)(data)

    def select(self, keys: Iterable[str | Node], /) -> Self:
        keys = (_as_id(key) for key in keys)
        selected: dict[str, T] = {key: self[key] for key in keys}
        return type(self)(selected)

    def update(self, nodes: T | Iterable[T] | Mapping[str, T], /, **kwargs: T) -> Self:
        data: dict[str, T] = self.data.copy()
        data.update(_as_mapping(nodes))
        data.update(kwargs)
        return type(self)(data)


def _as_id(key: str | Node) -> str:
    if isinstance(key, Node):
        return key.id
    return key
