import abc
from collections.abc import Iterable, Mapping
from typing import Self

import attrs
import toolz

from liblaf.apple import tree

from ._as_dict import SupportsKeysAndGetItem, as_dict


@tree.pytree
class FrozenDict[KT, VT](Mapping[KT, VT]):
    _data: Mapping[KT, VT] = tree.container(converter=as_dict, factory=dict)

    def set(self, key: KT, value: VT) -> Self:
        data: dict[KT, VT] = toolz.assoc(self._data, key, value)
        return attrs.evolve(self, _data=data)

    def update(
        self,
        changes: SupportsKeysAndGetItem[KT, VT] | Iterable[tuple[KT, VT]] | None = None,
        /,
        **kwargs: VT,
    ) -> Self:
        data: dict[KT, VT] = toolz.merge(self._data, changes or {}, kwargs)
        return attrs.evolve(self, _data=data)


class MappingMixin[DT, KT, VT](Mapping[KT, VT]):
    @abc.abstractmethod
    def _mapping_factory(self, data: Mapping[KT, VT]) -> DT:
        raise NotImplementedError

    def select(self, keys: Iterable[KT]) -> DT:
        return self._mapping_factory({k: self[k] for k in keys})

    def set(self, key: KT, value: VT) -> DT:
        data: dict[KT, VT] = toolz.assoc(self, key, value)
        return self._mapping_factory(data)

    def update(
        self,
        updates: SupportsKeysAndGetItem[KT, VT] | Iterable[tuple[KT, VT]] | None = None,
        /,
        **kwargs: VT,
    ) -> DT:
        data: dict[KT, VT] = toolz.merge(self, updates or {}, kwargs)
        return self._mapping_factory(data)
