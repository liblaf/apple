from collections.abc import Iterator, Mapping
from typing import Self

from ._pytree import PyTree, data


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
        key = self._as_key(key)
        value = self._as_value(value)
        new_data: dict[KT, VT] = {**self._data, key: value}
        return self.evolve(_data=new_data)

    def select(self, keys: Iterator[KT], /) -> Self:
        keys = (self._as_key(key) for key in keys)
        selected: dict[KT, VT] = {key: self._data[key] for key in keys}
        return self.evolve(_data=selected)

    def update(self, other: Mapping[KT, VT], /) -> Self:
        other = {self._as_key(k): self._as_value(v) for k, v in other.items()}
        new_data: dict[KT, VT] = {**self._data, **other}
        return self.evolve(_data=new_data)

    def _as_key(self, key: KT) -> KT:
        return key

    def _as_value(self, value: VT) -> VT:
        return value
