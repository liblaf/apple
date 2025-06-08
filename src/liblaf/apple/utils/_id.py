import collections
import functools
from collections.abc import Callable

counter = collections.Counter()


def uniq_id(prefix: str = "") -> str:
    id_: str = f"{prefix}-{counter[prefix]:03d}"
    counter[prefix] += 1
    return id_


def uniq_id_factory(prefix: str = "") -> Callable[[], str]:
    return functools.partial(uniq_id, prefix=prefix)
