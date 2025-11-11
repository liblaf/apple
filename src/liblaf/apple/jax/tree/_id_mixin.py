import collections

import attrs
from liblaf.peach import tree

_counter: collections.Counter[str] = collections.Counter()


def _default_id(self: "IdMixin") -> str:
    name: str = type(self).__qualname__
    count: int = _counter[name]
    _counter[name] += 1
    return f"{name}-{count:03d}"


@tree.define
class IdMixin:
    id: str = attrs.field(
        default=attrs.Factory(_default_id, takes_self=True), kw_only=True
    )
