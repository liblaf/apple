import dataclasses
from collections.abc import Generator, Iterable, Sequence
from typing import Any, Self

import attrs
import jax.tree_util as jtu

type AuxData = Sequence[Any]
type Children = Iterable[Any]
type ChildrenWithKeys = Iterable[tuple[Any, Any]]


@attrs.define
class Flattener[T]:
    cls: type[T]
    data_fields: Iterable[str] = attrs.field(default=())
    meta_fields: Iterable[str] = attrs.field(default=())

    @classmethod
    def from_cls(
        cls,
        nodetype: type[T],
        data_fields: Iterable[str] | None = None,
        meta_fields: Iterable[str] | None = None,
    ) -> Self:
        if data_fields is None:
            data_fields = _filter_fields(nodetype, static=False)
        if meta_fields is None:
            meta_fields = _filter_fields(nodetype, static=True)
        return cls(
            cls=nodetype, data_fields=tuple(data_fields), meta_fields=tuple(meta_fields)
        )

    def flatten_with_keys(self, obj: Any) -> tuple[ChildrenWithKeys, AuxData]:
        children: ChildrenWithKeys = tuple(
            (jtu.GetAttrKey(name), getattr(obj, name)) for name in self.data_fields
        )
        aux_data: AuxData = tuple(getattr(obj, name) for name in self.meta_fields)
        return children, aux_data

    def flatten(self, obj: Any) -> tuple[Children, AuxData]:
        children: Children = tuple(getattr(obj, name) for name in self.data_fields)
        aux_data: AuxData = tuple(getattr(obj, name) for name in self.meta_fields)
        return children, aux_data

    def unflatten(self, aux_data: AuxData, children: Children) -> T:
        obj: T = object.__new__(self.cls)
        for name, value in zip(self.meta_fields, aux_data, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in zip(self.data_fields, children, strict=True):
            object.__setattr__(obj, name, value)
        return obj


def register_attrs[T: type](
    cls: T,
    data_fields: Iterable[str] | None = None,
    meta_fields: Iterable[str] | None = None,
) -> T:
    flattener: Flattener[T] = Flattener.from_cls(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )
    jtu.register_pytree_with_keys(
        cls,
        flatten_with_keys=flattener.flatten_with_keys,
        unflatten_func=flattener.unflatten,
        flatten_func=flattener.flatten,
    )
    return cls


def _filter_fields(cls: type, *, static: bool) -> Generator[str]:
    if attrs.has(cls):
        for field in attrs.fields(cls):
            field: attrs.Attribute
            if field.metadata.get("static", False) is static:
                yield field.name
    elif dataclasses.is_dataclass(cls):
        for field in dataclasses.fields(cls):
            field: dataclasses.Field
            if field.metadata.get("static", False) is static:
                yield field.name
