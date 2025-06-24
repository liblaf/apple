from typing import Any


def implemented(fn: Any) -> bool:
    return not getattr(fn, "not_implemented", False)


def not_implemented[C](fn: C) -> C:
    fn.not_implemented = True  # pyright: ignore[reportAttributeAccessIssue]
    return fn
