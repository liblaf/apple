"""Decorators for dtype-specialized Warp structs."""

import functools
from typing import Any, cast

import warp as wp

from ._dtype import warp_default_dtype


def warp_struct[T: type](cls: T) -> T:
    if not hasattr(cls, "__annotations_factory__"):
        return cast("T", wp.struct(cls))

    @functools.cache
    def __class_getitem__(cls: T, key: Any) -> T:  # noqa: N807
        c: type = type(
            cls.__name__,
            (cls,),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__annotations__": cls.__annotations_factory__(key),  # ty:ignore[unresolved-attribute]
            },
        )
        return cast("T", wp.struct(c, module="unique"))

    def __new__(owner: type) -> object:  # noqa: N807
        if owner is cls:
            dtype: Any = warp_default_dtype()
            return __class_getitem__(cls, dtype)()
        return object.__new__(owner)

    cls.__class_getitem__ = classmethod(__class_getitem__)  # ty:ignore[invalid-assignment]
    cls.__new__ = staticmethod(__new__)  # ty:ignore[invalid-assignment]
    return cls
