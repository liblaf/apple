from collections.abc import Callable, Iterator, Mapping
from typing import Self

from liblaf.apple.struct._pytree import PyTree, data

from ._node import Node
from ._utils import KeyLike, KeysLike, NodesLike, as_key, as_keys, as_mapping


class NodeCollection[T: Node](Mapping[str, T], PyTree):
    _data: Mapping[str, T] = data(default={}, converter=as_mapping)

    # region Mapping[str, T]

    def __getitem__(self, key: KeyLike, /) -> T:
        key = as_key(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, T]

    def add(self, node: T, /) -> Self:
        key: str = as_key(node)
        data: Mapping[str, T] = {**self._data, key: node}
        return type(self)(data)

    def filter(self, predicate: Callable[[T], bool], /) -> Self:
        nodes: Mapping[str, T] = {
            key: node for key, node in self._data.items() if predicate(node)
        }
        return type(self)(nodes)

    def filter_instance[VT: Node](self, cls: type[VT], /) -> "NodeCollection[VT]":
        nodes: Mapping[str, VT] = {
            key: node for key, node in self._data.items() if isinstance(node, cls)
        }
        return type(self)(nodes)  # pyright: ignore[reportReturnType, reportArgumentType]

    def select(self, keys: KeysLike, /) -> Self:
        keys = as_keys(keys)
        selected: Mapping[str, T] = {key: self[key] for key in keys}
        return type(self)(selected)

    def update(self, nodes: NodesLike[T], /, **kwargs: T) -> Self:
        data: Mapping[str, T] = {**self._data, **as_mapping(nodes), **kwargs}
        return type(self)(data)
