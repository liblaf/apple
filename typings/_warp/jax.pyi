import jax

from .types import Array

def from_jax(jax_array: jax.Array, dtype: type | None = None) -> Array: ...
def to_jax(warp_array: Array) -> jax.Array: ...

__all__ = ["from_jax", "to_jax"]
