from collections.abc import Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Self, override

import wadler_lindig as wl

from liblaf.apple.struct import tree

from ._as_dict import as_dict
from ._as_key import as_key, as_keys
from .typed import KeyLike, KeysLike, MappingLike, Node


class NodeContainer[T: Node](tree.PyTreeMutable, MutableMapping[str, T]):
    data: dict[str, T] = tree.container(converter=as_dict, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike = None, /) -> None: ...

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        cls_kwargs: dict[str, Any] = kwargs.copy()
        cls_kwargs["show_type_module"] = cls_kwargs["show_dataclass_module"]
        return wl.pdoc(type(self), **cls_kwargs) + wl.pdoc(
            list(self.values()), **kwargs
        )

    # region impl MutableMapping[str, T]

    @override
    def __getitem__(self, key: KeyLike, /) -> T:
        key: str = as_key(key)
        return self.data[key]

    @override
    def __setitem__(self, key: KeyLike, value: T, /) -> None:
        key: str = as_key(key)
        self.data[key] = value

    @override
    def __delitem__(self, key: KeyLike, /) -> None:
        key: str = as_key(key)
        del self.data[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.data

    @override
    def __len__(self) -> int:
        return len(self.data)

    # endregion impl MutableMapping[str, T]

    def add(self, value: T, /) -> None:
        key: str = as_key(value)
        self[key] = value

    def key_filter(self, keys: KeysLike, /) -> Self:
        keys: list[str] = as_keys(keys)
        data: Mapping[str, T] = {k: self[k] for k in keys}
        return self.replace(data=data)
