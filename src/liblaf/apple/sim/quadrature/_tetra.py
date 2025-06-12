from typing import Self

import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct

from ._scheme import Scheme


class QuadratureTetra(Scheme):
    points: Float[jax.Array, "a J"] = struct.array(default=jnp.ones((1, 3)) / 4)
    weights: Float[jax.Array, " q"] = struct.array(default=jnp.ones((1,)) / 6)

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
