from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import pyvista as pv


@runtime_checkable
class Node(Protocol):
    @property
    def id(self) -> str: ...


type KeyLike = str | Node
type KeysLike = KeyLike | Iterable[KeyLike] | None
type MappingLike[T] = T | Iterable[T] | Mapping[str, T] | Mapping[KeyLike, T] | None


def as_dict(mapping: MappingLike) -> dict[str, Any]:
    if not mapping:
        return {}
    if isinstance(mapping, (Mapping, pv.DataSetAttributes)):
        return {as_key(k): v for k, v in mapping.items()}
    if isinstance(mapping, Iterable) and not isinstance(mapping, str):
        return {as_key(v): v for v in mapping}
    return {as_key(mapping): mapping}


def as_key(key: Any) -> str:
    if hasattr(key, "id"):
        return key.id
    return str(key)


def as_keys(keys: Any) -> Iterable[str]:
    if not keys:
        return ()
    if isinstance(keys, Iterable) and not isinstance(keys, str):
        return (as_key(k) for k in keys if k)
    return (as_key(keys),)
