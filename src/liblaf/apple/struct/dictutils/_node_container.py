from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self, override

import cytoolz as toolz
import wadler_lindig as wl

from liblaf.apple.struct import tree

from ._as_dict import as_dict
from ._as_key import as_key, as_keys
from .typed import KeyLike, KeysLike, MappingLike, Node


@tree.pytree
class NodeContainer[T: Node](tree.PyTreeMixin, Mapping[str, T]):
    _data: Mapping[str, T] = tree.container(converter=as_dict, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike = None, /) -> None: ...

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        cls_kwargs: dict[str, Any] = kwargs.copy()
        cls_kwargs["show_type_module"] = cls_kwargs["show_dataclass_module"]
        return wl.pdoc(type(self), **cls_kwargs) + wl.pdoc(
            list(self.values()), **kwargs
        )

    # region Mapping[str, T]

    @override
    def __getitem__(self, key: KeyLike, /) -> T:
        key: str = as_key(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data

    @override
    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, T]

    def add(self, value: T, /) -> Self:
        data: Mapping[str, T] = toolz.assoc(self._data, value.id, value)
        return type(self)(data)

    def clear(self) -> Self:
        return type(self)()

    def key_filter(self, keys: KeysLike, /) -> Self:
        keys: list[str] = as_keys(keys)
        data: Mapping[str, T] = {k: self[k] for k in keys}
        return type(self)(data)

    def pop(self, key: KeyLike, /) -> Self:
        key: str = as_key(key)
        data: Mapping[str, T] = toolz.dissoc(self._data, key)
        return type(self)(data)

    def update(self, updates: MappingLike, /, **kwargs) -> Self:
        updates = as_dict(updates)
        data: Mapping[str, T] = toolz.merge(self._data, updates, kwargs)
        return type(self)(data)
