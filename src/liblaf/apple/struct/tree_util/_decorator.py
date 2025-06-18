import functools
from collections.abc import Callable
from typing import Protocol, TypedDict, Unpack, dataclass_transform, overload

import attrs

from ._field import array, data, mapping, static
from ._register import register_attrs


class ClassDecorator(Protocol):
    def __call__[T: type](self, cls: T, /) -> T: ...


class DefineKwargs(TypedDict):
    pass


@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, mapping, static)
)
@overload
def pytree[T: type](**kwargs: Unpack[DefineKwargs]) -> ClassDecorator: ...
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, mapping, static)
)
@overload
def pytree[T: type](cls: T, /, **kwargs: Unpack[DefineKwargs]) -> T: ...
def pytree[T: type](maybe_cls: T | None = None, /, **kwargs) -> T | Callable:
    if maybe_cls is None:
        return functools.partial(pytree, **kwargs)
    kwargs.setdefault("field_transformer", _dataclass_names)
    cls: T = attrs.frozen(maybe_cls, **kwargs)
    cls = register_attrs(cls)
    return cls


def _dataclass_names(
    _cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    return [field.evolve(alias=field.name) for field in fields]
