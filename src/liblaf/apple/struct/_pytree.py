import abc
import functools
from collections.abc import Callable, Sequence
from typing import Any, Self, dataclass_transform

import attrs
import jax

from ._field import array, data, static
from ._utils import clone_signature


@clone_signature(attrs.frozen)
def pytree[C: type](maybe_cls: C | None = None, **kwargs) -> Callable | C:
    if maybe_cls is None:
        return functools.partial(pytree, **kwargs)
    cls: C = attrs.frozen(maybe_cls, **kwargs)
    cls = register_attrs(cls)
    return cls


def _dataclass_names(
    _cls: type, fields: list[attrs.Attribute]
) -> list[attrs.Attribute]:
    """.

    References:
        1. <https://www.attrs.org/en/stable/extending.html#automatic-field-transformation-and-modification>
    """
    fields = [
        field.evolve(alias=field.name) if not field.alias else field for field in fields
    ]
    return fields


@dataclass_transform(
    frozen_default=True, field_specifiers=(attrs.field, array, data, static)
)
class PyTreeMeta(abc.ABCMeta):
    def __new__[C: type](
        cls: type[C],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs,
    ) -> C:
        c: C = super().__new__(cls, name, bases, namespace, **kwargs)
        if "__attrs_attrs__" not in namespace:
            kwargs.setdefault("field_transformer", _dataclass_names)
            c = attrs.frozen(c, **kwargs)
        return c


class PyTree(metaclass=PyTreeMeta):
    @classmethod
    def __attrs_init_subclass__(cls, **kwargs) -> None:
        register_attrs(cls)

    def evolve(self, **changes) -> Self:
        return attrs.evolve(self, **changes)


def register_attrs[C: type](
    cls: C,
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> C:
    if data_fields is None:
        data_fields = _collect_fields(cls, static=False)
    if meta_fields is None:
        meta_fields = _collect_fields(cls, static=True)
    return jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields, drop_fields=drop_fields
    )


def _collect_fields(cls: type, *, static: bool) -> Sequence[str]:
    fields: list[str] = []
    for field in attrs.fields(cls):
        field: attrs.Attribute
        if not field.init:
            continue
        if field.metadata.get("static", False) == static:
            fields.append(field.name)
    return fields
