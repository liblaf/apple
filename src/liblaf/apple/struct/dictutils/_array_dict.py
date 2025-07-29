from collections.abc import Iterator, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Self, override

import wadler_lindig as wl
from jaxtyping import Array, ArrayLike

from liblaf.apple import utils
from liblaf.apple.struct import tree

from ._as_dict import as_dict
from ._as_key import as_key, as_keys
from .typed import KeyLike, KeysLike, MappingLike


def as_array_dict(data: MappingLike, /) -> dict[str, Array]:
    data: dict[str, ArrayLike] = as_dict(data)  # pyright: ignore[reportAssignmentType]
    return {k: utils.asarray(v) for k, v in data.items()}


class ArrayDict(tree.PyTreeMutable, MutableMapping[str, Array]):
    data: dict[str, Array] = tree.container(converter=as_array_dict, factory=dict)

    if TYPE_CHECKING:

        def __init__(self, data: MappingLike = None, /) -> None: ...

    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        cls_kwargs: dict[str, Any] = kwargs.copy()
        cls_kwargs["show_type_module"] = cls_kwargs["show_dataclass_module"]
        return wl.pdoc(type(self), **cls_kwargs) + wl.pdoc(self.data, **kwargs)

    # region impl MutableMapping[str, Array]

    @override
    def __getitem__(self, key: KeyLike, /) -> Array:
        key: str = as_key(key)
        return self.data[key]

    @override
    def __setitem__(self, key: KeyLike, value: ArrayLike, /) -> None:
        key: str = as_key(key)
        value: Array = utils.asarray(value)
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

    # endregion impl MutableMapping[str, Array]

    def __add__(self, other: MappingLike, /) -> Self:
        other: dict[str, Array] = as_array_dict(other)  # pyright: ignore[reportAssignmentType]
        data: dict[str, Array] = dict(self)
        for key, value in other.items():
            if key in data:
                data[key] += value
            else:
                data[key] = value
        return self.replace(data=data)

    @override
    def update(self, m: MappingLike | None = None, /, **kwargs: ArrayLike) -> None:
        updates: dict[str, Array] = as_array_dict(m)
        updates.update(as_array_dict(kwargs))
        self.data.update(updates)

    def key_filter(self, keys: KeysLike, /) -> Self:
        keys: list[str] = as_keys(keys)
        data: Mapping[str, Array] = {k: self.data[k] for k in keys}
        return self.replace(data=data)
