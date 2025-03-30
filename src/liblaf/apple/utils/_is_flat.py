from typing import Any

import jax


def is_flat(a: Any) -> bool:
    if isinstance(a, jax.Array):
        return a.ndim == 1
    return False
