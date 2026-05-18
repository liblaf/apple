from collections import Counter
from typing import Any

import attrs

counter: Counter[str] = Counter()


def _default_name(self: Any) -> str:
    cls_name: str = type(self).__name__
    count: int = counter[cls_name]
    counter[cls_name] += 1
    return f"{cls_name}{count}"


DEFAULT_POTENTIAL_NAME = attrs.Factory(_default_name, takes_self=True)
