from collections.abc import Iterable, Mapping
from typing import Protocol, cast, runtime_checkable


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...


type KeyLike = str | Node
type KeysLike = KeyLike | Iterable[KeyLike | None] | None
type NodesLike[T: Node] = T | Iterable[T | None] | Mapping[str, T] | None


def as_key(key: KeyLike) -> str:
    if isinstance(key, Node):
        return key.id
    return key


def as_keys(keys: KeysLike) -> Iterable[str]:
    if keys is None:
        return ()
    if isinstance(keys, Iterable) and not isinstance(keys, str):
        return (as_key(key) for key in keys if key is not None)
    return (as_key(keys),)


def as_mapping[T: Node](nodes: NodesLike[T], /) -> Mapping[str, T]:
    if nodes is None:
        return {}
    if isinstance(nodes, Mapping):
        nodes = cast("Mapping", nodes)
        return {**nodes}
    if isinstance(nodes, Iterable) and not isinstance(nodes, str):
        return {as_key(node): node for node in nodes if node is not None}
    nodes = cast("T", nodes)
    return {as_key(nodes): nodes}
