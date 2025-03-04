from typing import Protocol

import jax
from jaxtyping import Float, PyTree


class Unraveler(Protocol):
    def __call__(self, flat: Float[jax.Array, " N"]) -> PyTree: ...
