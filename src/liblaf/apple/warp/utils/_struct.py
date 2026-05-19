"""Decorators for dtype-specialized Warp structs."""

from typing import Any, cast

import attrs
import warp as wp
from warp._src.codegen import Struct, StructInstance

from ._dtype import warp_default_dtype


@attrs.define
class _WarpStruct:
    __wrapped__: type
    _cache: dict[Any, Struct] = attrs.field(factory=dict, init=False)

    def __call__(self, *args, **kwargs) -> StructInstance:
        dtype: Any = warp_default_dtype()
        return self[dtype](*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        dtype: Any = warp_default_dtype()
        return getattr(self[dtype], name)

    def __getitem__(self, key: Any) -> Struct:
        if key not in self._cache:
            c: type = type(
                self.__wrapped__.__name__,
                (self.__wrapped__,),
                {
                    "__module__": self.__wrapped__.__module__,
                    "__qualname__": self.__wrapped__.__qualname__,
                    "__annotations__": self.__wrapped__.__annotations_factory__(key),  # ty:ignore[unresolved-attribute]
                },
            )
            self._cache[key] = wp.struct(c, module="unique")
        return self._cache[key]


def warp_struct[T: type](cls: T) -> T:
    if not hasattr(cls, "__annotations_factory__"):
        return cast("T", wp.struct(cls))
    return cast("T", _WarpStruct(cls))
