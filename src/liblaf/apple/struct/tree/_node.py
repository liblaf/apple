import collections
from typing import Any

from ._field import field
from ._pytree import PyTree

_counter: collections.Counter[str] = collections.Counter()


class PyTreeNode(PyTree):
    id: str = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.id is None:
            object.__setattr__(self, "id", uniq_id(self))


def uniq_id(obj: Any) -> str:
    name: str = type(obj).__qualname__
    id_: str = f"{name}_{_counter[name]:03d}"
    _counter[name] += 1
    return id_
