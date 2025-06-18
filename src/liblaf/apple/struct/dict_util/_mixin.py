from collections.abc import Callable, Mapping

from liblaf.apple.struct._utils import KeysLike, MappingLike, as_dict, as_keys

from ._frozen_dict import FrozenDict


class MappingMixin[T](Mapping[str, T]):
    def copy(self, add_or_replace: MappingLike = None, /) -> FrozenDict[T]:
        return FrozenDict({**self, **as_dict(add_or_replace)})

    def select(self, keys: KeysLike, /) -> FrozenDict[T]:
        return FrozenDict({key: self[key] for key in as_keys(keys)})

    def filter(self, predicate: Callable[[str, T], bool], /) -> FrozenDict[T]:
        return FrozenDict(
            {key: value for key, value in self.items() if predicate(key, value)}
        )

    def filter_instance[VT](self, cls: type[VT], /) -> FrozenDict[VT]:
        return FrozenDict(
            {key: value for key, value in self.items() if isinstance(value, cls)}
        )
