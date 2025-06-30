from typing import Any

import equinox as eqx


def tree_at(*args, **kwargs) -> Any:
    return eqx.tree_at(*args, **kwargs)
