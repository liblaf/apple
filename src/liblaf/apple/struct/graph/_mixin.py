import abc
from collections.abc import Callable, Iterator, Mapping

from typing_extensions import TypeIs

from ._collection import NodeCollection
from ._node import Node
from ._utils import KeysLike, as_keys


class NodeCollectionMixin[T: Node](Mapping[str, T]):
    # region Mapping[str, T]

    @abc.abstractmethod
    def __getitem__(self, key: str, /) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    # endregion Mapping[str, T]

    def filter(self, predicate: Callable[[Node], TypeIs[T]], /) -> NodeCollection[T]:
        filtered: Mapping[str, T] = {
            key: node for key, node in self.items() if predicate(node)
        }
        return NodeCollection(filtered)

    def filter_instance[VT: Node](self, cls: type[VT], /) -> NodeCollection[VT]:
        filtered: Mapping[str, VT] = {
            key: node for key, node in self.items() if isinstance(node, cls)
        }
        return NodeCollection(filtered)

    def select(self, keys: KeysLike, /) -> NodeCollection[T]:
        keys = as_keys(keys)
        selected: Mapping[str, T] = {key: self[key] for key in keys}
        return NodeCollection(selected)
