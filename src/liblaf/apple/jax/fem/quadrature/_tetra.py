from typing import Self

import felupe.quadrature
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp

from ._scheme import Scheme


@jarp.define
class QuadratureTetra(Scheme):
    points: Float[Array, "quadrature dim"] = jarp.array(default=jnp.ones((1, 3)) / 4.0)
    weights: Float[Array, " quadrature"] = jarp.array(default=jnp.ones((1,)) / 6.0)

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
