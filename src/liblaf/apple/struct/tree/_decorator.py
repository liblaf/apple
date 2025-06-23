import functools
from collections.abc import Callable
from typing import dataclass_transform, overload

import attrs

from ._field_specifiers import array, container, data, static
from ._register_attrs import register_attrs


@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, container, static)
)
def pytree[T: type](cls: T, /, **kwargs) -> T: ...
@overload
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, container, static)
)
def pytree[T: type](**kwargs) -> Callable[[T], T]: ...
@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, container, static)
)
def pytree[T: type](cls: T | None = None, /, **kwargs) -> Callable | T:
    if cls is None:
        return functools.partial(pytree, **kwargs)
    kwargs.setdefault("field_transformer", _dataclass_names)
    if "repr" not in kwargs and cls.__repr__ is not object.__repr__:
        kwargs["repr"] = False
    cls: T = attrs.frozen(cls, **kwargs)
    cls = register_attrs(cls)
    return cls


def _dataclass_names(
    _cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    """...

    References:
        1. <https://www.attrs.org/en/stable/extending.html#automatic-field-transformation-and-modification>
    """
    return [field.evolve(alias=field.name) for field in fields]
