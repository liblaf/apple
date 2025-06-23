import functools
from typing import Any, Protocol, runtime_checkable

from liblaf import grapes


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...


@functools.singledispatch
def as_key(*args, **kwargs) -> str:
    raise grapes.error.DispatchLookupError(as_key, args, kwargs)


@as_key.register(str)
def _(key: str) -> str:
    return key


@as_key.register(tuple)
def _(pair: tuple[str, Any]) -> str:
    key: str
    key, _ = pair
    return key


@as_key.register(Node)
def _(node: Node) -> str:
    return node.id
