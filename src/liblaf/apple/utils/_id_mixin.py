from collections import Counter
from typing import Any

import attrs
from liblaf.peach import tree

counter: Counter[str] = Counter()


def _default_id(self: Any) -> str:
    name: str = type(self).__name__
    count: int = counter[name]
    counter[name] += 1
    return f"{name}{count:03d}"


@tree.define
class IdMixin:
    id: str = tree.field(
        default=attrs.Factory(_default_id, takes_self=True), kw_only=True
    )
