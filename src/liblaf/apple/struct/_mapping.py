import abc
from collections.abc import Mapping
from typing import Self


class MappingTrait[KT, VT](Mapping[KT, VT]):
    def add(self, key: KT, value: VT) -> Self:
        key = self.as_key(key)
        items: Mapping[KT, VT] = {**self, key: value}
        return self.with_items(items)

    def select(self, keys: Mapping[KT, VT]) -> Self:
        selected: Mapping[KT, VT] = {key: self[key] for key in keys}
        return self.with_items(selected)

    @abc.abstractmethod
    def with_items(self, items: Mapping[KT, VT]) -> Self:
        raise NotImplementedError

    def as_key(self, key: KT) -> KT:
        return key


class UserMapping[KT, VT](Mapping[KT, VT]):
    pass
