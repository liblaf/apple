from collections.abc import Callable
from typing import Any


def clone_signature[C](_ref: C) -> Callable[..., C]:
    def wrapper(func: Any) -> C:
        return func

    return wrapper
