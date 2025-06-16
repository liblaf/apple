import abc
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Self

from ._pytree import PyTree, data


class CollectionTrait[KT, VT](Mapping[KT, VT], PyTree):
    if TYPE_CHECKING:

        def __init__(self, data: Mapping[KT, VT], /) -> None: ...

    # region Mapping[str, T]

    @abc.abstractmethod
    def __getitem__(self, key: KT) -> VT:
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    # endregion Mapping[str, T]

    def select(self, keys: Iterable[KT], /) -> Self:
        selected: dict[KT, VT] = {key: self[key] for key in keys}
        return self.evolve(data=selected)


class FrozenDict[KT, VT](Mapping[KT, VT], PyTree):
    _data: Mapping[KT, VT] = data(default={})

    # region Mapping[str, T]

    def __getitem__(self, key: KT) -> VT:
        return self._data[key]

    def __iter__(self) -> Iterator[KT]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    # endregion Mapping[str, T]

    def add(self, key: KT, value: VT) -> Self:
        new_data: dict[KT, VT] = {**self._data, key: value}
        return self.evolve(_data=new_data)

    def select(self, keys: Iterable[KT], /) -> Self:
        selected: dict[KT, VT] = {key: self._data[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, other: Mapping[KT, VT], /) -> Self:
        new_data: dict[KT, VT] = {**self._data, **other}
        return self.evolve(_data=new_data)
