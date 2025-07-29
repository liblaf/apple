import functools
import types
from collections.abc import Iterable
from typing import Any

from liblaf import grapes

from .typed import KeyLike


def as_key(key: Any, /) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        return key[0]
    if (id_ := getattr(key, "id", None)) is not None:
        return id_
    raise grapes.error.DispatchLookupError(as_key, (key,), {})


@functools.singledispatch
def as_keys(*args, **kwargs) -> list[str]:
    raise grapes.error.DispatchLookupError(as_keys, args, kwargs)


@as_keys.register(Iterable)
def _(keys: Iterable[KeyLike], /) -> list[str]:
    return [as_key(key) for key in keys]


@as_keys.register(types.NoneType)
def _(_: None) -> list[str]:
    return []
