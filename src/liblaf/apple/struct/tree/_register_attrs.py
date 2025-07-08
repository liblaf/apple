import functools
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final, overload

import attrs
import jax
from jaxtyping import PyTree
from typing_extensions import Sentinel

type KeyLeafPair = tuple[Any, PyTree]


FLATTEN_SENTINEL = Sentinel("FLATTEN_SENTINEL")
WRAPPER_FIELD_NAMES: Final = {
    "__annotations__",
    "__doc__",
    "__module__",
    "__name__",
    "__qualname__",
}


@attrs.frozen
class FlattenedData:
    """Used to provide a pretty repr when doing `jtu.tree_structure(SomeModule(...))`.

    References:
        1. <https://github.com/patrick-kidger/equinox/blob/9a21f7c84dc65e1fae076679cdabe967dbf65d9f/equinox/_module/_module.py#L47C1-L55>
    """

    data_field_names: Sequence[str] = attrs.field()
    meta_fields: Sequence[tuple[str, Any]] = attrs.field()
    wrapper_fields: Sequence[tuple[str, Any]] = attrs.field()

    def __repr__(self) -> str:
        return repr((self.data_field_names, self.meta_fields))[1:-1]


@attrs.frozen
class PyTreeFlattener:
    """.

    References:
        1. <https://github.com/patrick-kidger/equinox/blob/9a21f7c84dc65e1fae076679cdabe967dbf65d9f/equinox/_module/_module.py#L58-L127>
    """

    meta_fields: Iterable[str] = attrs.field()
    data_fields: Iterable[str] = attrs.field()

    def flatten(self, obj: Any) -> tuple[Sequence[PyTree], FlattenedData]:
        data_fields: list[str] = []
        data_values: list[PyTree] = []
        for k in self.data_fields:
            v: Any = getattr(obj, k, FLATTEN_SENTINEL)
            if v is FLATTEN_SENTINEL:
                continue
            data_fields.append(k)
            data_values.append(v)
        aux = FlattenedData(
            data_field_names=data_fields,
            meta_fields=_get_attrs(obj, self.meta_fields),
            wrapper_fields=_get_attrs(obj, WRAPPER_FIELD_NAMES),
        )
        return data_values, aux

    def flatten_with_keys(
        self, obj: Any
    ) -> tuple[Sequence[KeyLeafPair], FlattenedData]:
        data_fields: list[str] = []
        data_values: list[KeyLeafPair] = []
        for k in self.data_fields:
            v: Any = getattr(obj, k, FLATTEN_SENTINEL)
            if v is FLATTEN_SENTINEL:
                continue
            data_fields.append(k)
            data_values.append((jax.tree_util.GetAttrKey(k), v))
        aux = FlattenedData(
            data_field_names=data_fields,
            meta_fields=_get_attrs(obj, self.meta_fields),
            wrapper_fields=_get_attrs(obj, WRAPPER_FIELD_NAMES),
        )
        return data_values, aux

    @staticmethod
    def unflatten[T](
        attr_cls: type[T], aux: FlattenedData, values: Sequence[PyTree], /
    ) -> T:
        # This doesn't go via `__init__`. A user may have done something
        # nontrivial there, and the field values may be dummy values as used in
        # various places throughout JAX. See also
        # https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization,
        # which was (I believe) inspired by Equinox's approach here.
        obj: T = object.__new__(attr_cls)
        for name, value in zip(aux.data_field_names, values, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in itertools.chain(aux.meta_fields, aux.wrapper_fields):
            if value is not FLATTEN_SENTINEL:
                object.__setattr__(obj, name, value)
        return obj


@overload
def register_attrs[T: type](
    *,
    meta_fields: Iterable[str] | None = None,
    data_fields: Iterable[str] | None = None,
) -> Callable[[T], T]: ...
@overload
def register_attrs[T: type](
    cls: T,
    *,
    meta_fields: Iterable[str] | None = None,
    data_fields: Iterable[str] | None = None,
) -> T: ...
def register_attrs(
    cls: type | None = None,
    *,
    meta_fields: Iterable[str] | None = None,
    data_fields: Iterable[str] | None = None,
) -> Any:
    if cls is None:
        return functools.partial(
            register_attrs, meta_fields=meta_fields, data_fields=data_fields
        )
    if meta_fields is None:
        meta_fields = _filter_fields(cls, static=True)
    if data_fields is None:
        data_fields = _filter_fields(cls, static=False)
    flattener = PyTreeFlattener(data_fields=data_fields, meta_fields=meta_fields)
    jax.tree_util.register_pytree_with_keys(
        cls,
        flatten_with_keys=flattener.flatten_with_keys,
        unflatten_func=functools.partial(PyTreeFlattener.unflatten, cls),  # pyright: ignore[reportArgumentType]
        flatten_func=flattener.flatten,
    )
    return cls


def _get_attrs(obj: Any, names: Iterable[str]) -> list[tuple[str, Any]]:
    return [(name, getattr(obj, name, FLATTEN_SENTINEL)) for name in names]


def _filter_fields(cls: type, *, static: bool) -> list[str]:
    fields: list[str] = []
    for field in attrs.fields(cls):
        field: attrs.Attribute
        if field.metadata.get("static", False) == static:
            fields.append(field.name)
    return fields
