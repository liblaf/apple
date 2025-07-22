from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Self, override

import cytoolz as toolz
import wadler_lindig as wl

from liblaf.apple.struct import tree

from ._as_dict import as_dict
from ._as_key import as_key, as_keys
from .typed import KeyLike, KeysLike, MappingLike, Node


class NodeContainer[T: Node](tree.PyTree, Mapping[str, T]):
    data: Mapping[str, T] = tree.container(converter=as_dict, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike = None, /) -> None: ...

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        cls_kwargs: dict[str, Any] = kwargs.copy()
        cls_kwargs["show_type_module"] = cls_kwargs["show_dataclass_module"]
        return wl.pdoc(type(self), **cls_kwargs) + wl.pdoc(
            list(self.values()), **kwargs
        )

    # region impl Mapping[str, T]

    @override
    def __getitem__(self, key: KeyLike, /) -> T:
        key: str = as_key(key)
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.data

    @override
    def __len__(self) -> int:
        return len(self.data)

    # endregion impl Mapping[str, T]

    def add(self, value: T, /) -> Self:
        data: Mapping[str, T] = toolz.assoc(self.data, value.id, value)
        return self.replace(data=data)

    def clear(self) -> Self:
        return self.replace(data={})

    def key_filter(self, keys: KeysLike, /) -> Self:
        keys: list[str] = as_keys(keys)
        data: Mapping[str, T] = {k: self[k] for k in keys}
        return self.replace(data=data)

    def pop(self, key: KeyLike, /) -> Self:
        key: str = as_key(key)
        data: Mapping[str, T] = toolz.dissoc(self.data, key)
        return self.replace(data=data)

    def update(self, updates: MappingLike, /, **kwargs) -> Self:
        updates = as_dict(updates)
        data: Mapping[str, T] = toolz.merge(self.data, updates, kwargs)
        return self.replace(data=data)
