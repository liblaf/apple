from collections.abc import Callable, Sequence
from typing import Any

type Node = Any

def tree_at[T](
    where: Callable[[T], Node | Sequence[Node]],
    pytree: T,
    replace: Any | Sequence[Any] = ...,
    replace_fn: Callable[[Node], Any] = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> T: ...
