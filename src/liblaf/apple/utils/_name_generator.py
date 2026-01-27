import collections
from typing import Any

import attrs

counter: collections.Counter[str] = collections.Counter()


def make_name(obj: Any) -> str:
    cls: type = type(obj)
    cls_name: str = cls.__qualname__
    count: int = counter[cls_name]
    counter[cls_name] += 1
    return f"{cls_name}{count:03d}"


name_factory = attrs.Factory(make_name, takes_self=True)
