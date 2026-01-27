from typing import Self

import felupe.quadrature
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._scheme import Scheme


@jarp.define
class QuadratureTetra(Scheme):
    points: Float[Array, "quadrature dim"] = jarp.array(
        factory=lambda: jnp.ones((1, 3)) / 4.0
    )
    weights: Float[Array, " quadrature"] = jarp.array(
        factory=lambda: jnp.ones((1,)) / 6.0
    )

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
