from collections.abc import Iterable, Sequence
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
    def from_cls(cls, nodetype: type[T]) -> Self:
        data_fields: list[str] = []
        meta_fields: list[str] = []
        for field in attrs.fields(nodetype):
            field: attrs.Attribute
            if field.metadata.get("static", False):
                meta_fields.append(field.name)
            else:
                data_fields.append(field.name)
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
        for key, value in zip(self.meta_fields, aux_data, strict=True):
            object.__setattr__(obj, key, value)
        for key, value in zip(self.data_fields, children, strict=True):
            object.__setattr__(obj, key, value)
        return obj


def register_attrs[T: type](cls: T) -> T:
    flattener: Flattener[T] = Flattener.from_cls(cls)
    jtu.register_pytree_with_keys(
        cls,
        flatten_with_keys=flattener.flatten_with_keys,
        unflatten_func=flattener.unflatten,
        flatten_func=flattener.flatten,
    )
    return cls
