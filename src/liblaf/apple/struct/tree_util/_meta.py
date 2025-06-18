import abc
from typing import Any, dataclass_transform

import attrs

from ._decorator import pytree
from ._field import array, data, mapping, static


@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, mapping, static)
)
class PyTreeMeta(abc.ABCMeta):
    def __new__[T: type](
        mcs: type[T],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> T:
        cls: T = super().__new__(mcs, name, bases, namespace, **kwargs)
        if "__attrs_attrs__" in namespace:
            return cls
        kwargs.setdefault("repr", False)
        cls = pytree(cls, **kwargs)
        return cls
