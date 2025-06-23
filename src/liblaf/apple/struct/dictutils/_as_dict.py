import functools
import types
from collections.abc import Iterable, Mapping
from typing import Any

import toolz

from liblaf import grapes

from .typed import Node, SupportsKeysAndGetItem


@functools.singledispatch
def as_dict(*args, **kwargs) -> dict[Any, Any]:
    raise grapes.error.DispatchLookupError(as_dict, args, kwargs)


@as_dict.register(Mapping)
@as_dict.register(SupportsKeysAndGetItem)
def _[KT, VT](data: SupportsKeysAndGetItem[KT, VT]) -> dict[KT, VT]:
    return dict(data)


@as_dict.register(Iterable)
def _[KT, VT](data: Iterable[tuple[KT, VT] | Node]) -> dict[KT, VT]:
    first: tuple[KT, VT] | Node
    try:
        first, data = toolz.peek(data)
    except StopIteration:
        return {}
    if isinstance(first, tuple) and len(first) == 2:
        return dict(data)
    return {item.id: item for item in data}


@as_dict.register(types.NoneType)
def _(_data: None) -> dict[Any, Any]:
    return {}
