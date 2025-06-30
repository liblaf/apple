from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self, override

import attrs
import cytoolz as toolz
import jax
import jax.numpy as jnp
import wadler_lindig as wl
from jaxtyping import ArrayLike

from liblaf.apple.struct import tree

from ._as_dict import as_dict
from ._as_key import as_key, as_keys
from .typed import KeyLike, KeysLike, MappingLike


def as_array_dict(data: MappingLike, /) -> dict[str, jax.Array]:
    data: dict[str, ArrayLike] = as_dict(data)  # pyright: ignore[reportAssignmentType]
    return {k: jnp.asarray(v) for k, v in data.items()}


@tree.pytree
class ArrayDict(tree.PyTreeMixin, Mapping[str, jax.Array]):
    _data: Mapping[str, jax.Array] = tree.container(
        converter=as_array_dict, factory=dict
    )

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike = None, /) -> None: ...

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        cls_kwargs: dict[str, Any] = kwargs.copy()
        cls_kwargs["show_type_module"] = cls_kwargs["show_dataclass_module"]
        return wl.pdoc(type(self), **cls_kwargs) + wl.pdoc(self._data, **kwargs)

    # region Mapping[str, jax.Array]

    @override
    def __getitem__(self, key: KeyLike, /) -> jax.Array:
        key: str = as_key(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data

    @override
    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, jax.Array]

    def __add__(self, other: MappingLike, /) -> Self:
        other: dict[str, jax.Array] = as_array_dict(other)  # pyright: ignore[reportAssignmentType]
        data: dict[str, jax.Array] = dict(self)
        for key, value in other.items():
            if key in data:
                data[key] += value
            else:
                data[key] = value
        return attrs.evolve(self, _data=data)

    def clear(self) -> Self:
        return attrs.evolve(self, _data={})

    def key_filter(self, keys: KeysLike, /) -> Self:
        keys: list[str] = as_keys(keys)
        data: Mapping[str, jax.Array] = {k: self._data[k] for k in keys}
        return attrs.evolve(self, _data=data)

    def pop(self, key: KeyLike, /) -> Self:
        key: str = as_key(key)
        data: Mapping[str, jax.Array] = toolz.dissoc(self._data, key)
        return attrs.evolve(self, _data=data)

    def set(self, key: KeyLike, value: ArrayLike, /) -> Self:
        key: str = as_key(key)
        value: jax.Array = jnp.asarray(value)
        data: Mapping[str, jax.Array] = toolz.assoc(self._data, key, value)
        return attrs.evolve(self, _data=data)

    def update(self, updates: MappingLike | None = None, /, **kwargs) -> Self:
        updates = as_dict(updates)
        updates = toolz.valmap(jnp.asarray, updates)
        data: Mapping[str, jax.Array] = toolz.merge(self._data, updates, kwargs)
        return attrs.evolve(self, _data=data)
