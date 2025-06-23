import itertools
from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import attrs
import jax

KeyEntry = TypeVar("KeyEntry", bound=Any)
KeyLeafPair = tuple[KeyEntry, Any]
_FLATTEN_SENTINEL = object()


@attrs.frozen
class FlattenedData:
    """Used to provide a pretty repr when doing `jax.tree.structure(...)`.

    References:
        1. <https://github.com/patrick-kidger/equinox/blob/f8ca3458d85c178a2addaff7c50ef6f2eb250ced/equinox/_module.py#L907-L915>
    """

    dynamic_field_names: Sequence[str] = attrs.field(converter=tuple)  # pyright: ignore[reportGeneralTypeIssues]
    static_fields: Sequence[tuple[str, Any]] = attrs.field(converter=tuple)  # pyright: ignore[reportGeneralTypeIssues]
    wrapper_fields: Sequence[tuple[str, Any]] = attrs.field(default=(), converter=tuple)  # pyright: ignore[reportGeneralTypeIssues]

    def __repr__(self) -> str:
        return repr((self.dynamic_field_names, self.static_fields))[1:-1]


def register_attrs[T](cls: type[T]) -> type[T]:
    data_fields: list[str] = []
    meta_fields: list[str] = []
    for field in attrs.fields(cls):
        field: attrs.Attribute
        if not field.init:
            continue
        if field.metadata.get("static", False):
            meta_fields.append(field.name)
        else:
            data_fields.append(field.name)

    def flatten_with_keys(self: T) -> tuple[Iterable[KeyLeafPair], FlattenedData]:
        children: list[KeyLeafPair] = [
            (jax.tree_util.GetAttrKey(name), getattr(self, name))
            for name in data_fields
        ]
        static_fields: list[tuple[str, Any]] = [
            (name, getattr(self, name, _FLATTEN_SENTINEL)) for name in meta_fields
        ]
        aux_data: FlattenedData = FlattenedData(
            dynamic_field_names=data_fields, static_fields=tuple(static_fields)
        )
        return children, aux_data

    def unflatten_func(aux: FlattenedData, dynamic_field_values: Iterable[Any]) -> T:
        """...

        This doesn't go via `__init__`. A user may have done something
        nontrivial there, and the field values may be dummy values as used in
        various places throughout JAX. See also
        https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization,
        which was (I believe) inspired by Equinox's approach here.

        References:
            1. <https://github.com/patrick-kidger/equinox/blob/f8ca3458d85c178a2addaff7c50ef6f2eb250ced/equinox/_module.py#L918-L930>
        """
        self: T = object.__new__(cls)
        for name, value in zip(
            aux.dynamic_field_names, dynamic_field_values, strict=True
        ):
            object.__setattr__(self, name, value)
        for name, value in itertools.chain(aux.static_fields, aux.wrapper_fields):
            if value is not _FLATTEN_SENTINEL:
                object.__setattr__(self, name, value)
        return self

    def flatten_func(self: T) -> tuple[Iterable[Any], FlattenedData]:
        children: list[Any] = [
            getattr(self, name, _FLATTEN_SENTINEL) for name in data_fields
        ]
        static_fields: list[tuple[str, Any]] = [
            (name, getattr(self, name, _FLATTEN_SENTINEL)) for name in meta_fields
        ]
        aux_data: FlattenedData = FlattenedData(
            dynamic_field_names=data_fields, static_fields=static_fields
        )
        return children, aux_data

    jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys=flatten_with_keys,
        unflatten_func=unflatten_func,  # pyright: ignore[reportArgumentType]
        flatten_func=flatten_func,
    )
    return cls
