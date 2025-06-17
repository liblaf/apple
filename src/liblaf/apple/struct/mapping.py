from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, override, runtime_checkable

from .pytree import PyTree, field


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...


type KeyLike = str | Node
type KeysLike = KeyLike | Iterable[KeyLike] | None
type MappingLike[T] = T | Iterable[T] | Mapping[str, T] | Mapping[KeyLike, T] | None


def as_dict[T](mapping: MappingLike[T]) -> dict[str, T]:
    if mapping is None:
        return {}
    if isinstance(mapping, Mapping):
        return {as_key(k): v for k, v in mapping.items()}
    if isinstance(mapping, Iterable) and not isinstance(mapping, str):
        return {as_key(v): v for v in mapping}
    mapping = cast("T", mapping)
    return {as_key(mapping): mapping}


class PyTreeDict[T](PyTree, MutableMapping[str, T]):
    _data: dict[str, T] = field(converter=as_dict, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike[T] = None, /) -> None: ...

    @classmethod
    def merge(cls, *mappings: MappingLike[T]) -> Self:
        self: Self = cls()
        for d in mappings:
            self.update(d)
        return self

    # region MutableMapping[str, T]

    @override
    def __getitem__(self, key: KeyLike) -> T:
        key = as_key(key)
        return self._data[key]

    @override
    def __setitem__(self, key: KeyLike, value: T) -> None:
        key = as_key(key)
        self._data[key] = value

    @override
    def __delitem__(self, key: KeyLike) -> None:
        key = as_key(key)
        del self._data[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._data

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def update(self, mapping: MappingLike[T] | None = None, /, **kwargs: T) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        if mapping:
            self._data.update(as_dict(mapping))
        if kwargs:
            self._data.update(kwargs)
        return self

    # endregion MutableMapping[str, T]

    def add(self, key: KeyLike | T, value: T | None = None, /) -> None:
        self[key] = value if value is not None else key  # pyright: ignore[reportArgumentType]

    def filter(self, predicate: Callable[[str, T], bool], /) -> Self:
        return type(self)({k: v for k, v in self.items() if predicate(k, v)})

    def filter_instance(self, cls: type[T], /) -> Self:
        return type(self)({k: v for k, v in self.items() if isinstance(v, cls)})

    def select(self, keys: KeysLike, /) -> Self:
        keys = as_keys(keys)
        return type(self)({k: self[k] for k in keys})


class MappingTrait[T: PyTree](Mapping[str, T]):
    def filter(self, predicate: Callable[[str, T], bool], /) -> PyTreeDict[T]:
        return PyTreeDict({k: v for k, v in self.items() if predicate(k, v)})

    def filter_instance(self, cls: type[T], /) -> PyTreeDict[T]:
        return PyTreeDict({k: v for k, v in self.items() if isinstance(v, cls)})

    def select(self, keys: KeysLike, /) -> PyTreeDict[T]:
        keys = as_keys(keys)
        return PyTreeDict({k: self[k] for k in keys})


def as_key(key: Any) -> str:
    if isinstance(key, Node):
        return key.id
    return key


def as_keys(keys: KeysLike) -> Iterable[str]:
    if keys is None:
        return ()
    if isinstance(keys, Iterable) and not isinstance(keys, str):
        return (as_key(k) for k in keys)
    return (as_key(keys),)
