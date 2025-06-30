from typing import Any


def is_implemented(fn: Any) -> bool:
    return getattr(fn, "implemented", True)


def not_implemented[C](fn: C) -> C:
    object.__setattr__(fn, "implemented", False)  # bypass frozen instance
    return fn
