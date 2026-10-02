from collections.abc import Callable
from typing import Any


def not_implemented[F](func: F) -> F:
    """Mark a protocol stub as intentionally unimplemented.

    The marker lets
    [`is_implemented`][liblaf.apple.solvers.utils.is_implemented] distinguish inherited
    protocol placeholders from concrete hook implementations.
    """
    func.__not_implemented__ = True  # ty:ignore[unresolved-attribute]
    return func


def is_implemented(obj: Any, method: str | Callable) -> bool:
    """Return whether `obj` provides a concrete method implementation.

    Args:
        obj: Object to inspect.
        method: Method name or callable whose `__name__` is used.

    Returns:
        `False` for missing attributes, `None`, or methods decorated with
        [`not_implemented`][liblaf.apple.solvers.utils.not_implemented]; otherwise
        `True`.

    Examples:
        >>> from liblaf.apple.solvers.utils import is_implemented, not_implemented
        >>> class Hooks:
        ...     @not_implemented
        ...     def callback(self): ...
        >>> is_implemented(Hooks(), "callback")
        False
        >>> class Concrete:
        ...     def callback(self): ...
        >>> is_implemented(Concrete(), "callback")
        True
    """
    if not isinstance(method, str):
        method: str = method.__name__  # ty:ignore[unresolved-attribute]
    try:
        method: Any = getattr(obj, method)
    except AttributeError:
        return False
    return not (method is None or getattr(method, "__not_implemented__", False))
