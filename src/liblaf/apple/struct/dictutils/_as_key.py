import functools
import types
from collections.abc import Iterable
from typing import Any

from liblaf import grapes

from .typed import KeyLike, Node


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


@functools.singledispatch
def as_keys(*args, **kwargs) -> list[str]:
    raise grapes.error.DispatchLookupError(as_keys, args, kwargs)


@as_keys.register(Iterable)
def _(keys: Iterable[KeyLike], /) -> list[str]:
    return [as_key(key) for key in keys]


@as_keys.register(types.NoneType)
def _(_: None) -> list[str]:
    return []
