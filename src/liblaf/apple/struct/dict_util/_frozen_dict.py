from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Self, override

import wadler_lindig as wl

from liblaf.apple.struct import tree_util
from liblaf.apple.struct._utils import (
    KeyLike,
    KeysLike,
    MappingLike,
    as_dict,
    as_key,
    as_keys,
)


class FrozenDict[T](tree_util.PyTree, Mapping[str, T]):
    _data: dict[str, T] = tree_util.mapping()

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        # TODO: implement better `__repr__()`
        return super().__pdoc__(**kwargs)

    # region Mapping[str, T]

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike | None = None, /) -> None: ...

    @override
    def __getitem__(self, key: KeyLike) -> T:
        key = as_key(key)
        return self._data[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    @override
    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, T]

    def copy(self, add_or_replace: MappingLike = None, /) -> Self:
        return type(self)({**self, **as_dict(add_or_replace)})

    def select(self, keys: KeysLike, /) -> Self:
        return type(self)({key: self[key] for key in as_keys(keys)})

    def filter(self, predicate: Callable[[str, T], bool], /) -> Self:
        return type(self)(
            {key: value for key, value in self.items() if predicate(key, value)}
        )

    def filter_instance(self, cls: type[T], /) -> Self:
        return self.filter(lambda _, value: isinstance(value, cls))
