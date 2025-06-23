from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...

def as_key(key: str | tuple[str, Any] | Node, /) -> str: ...
