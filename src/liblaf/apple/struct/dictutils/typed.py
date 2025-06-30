from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...


@runtime_checkable
class SupportsKeysAndGetItem[KT, VT](Protocol):
    def keys(self) -> Iterable[KT]: ...
    def __getitem__(self, key: KT, /) -> VT: ...


type KeyLike = str | tuple[str, Any] | Node
type KeysLike = Iterable[str | Node] | None
type MappingLike = (
    SupportsKeysAndGetItem[str, Any]
    | SupportsKeysAndGetItem[str | Node, Any]
    | Iterable[tuple[str | Node, Any]]
    | Iterable[Node]
    | None
)
