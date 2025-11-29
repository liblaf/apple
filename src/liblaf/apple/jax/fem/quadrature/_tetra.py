from typing import Self

import felupe.quadrature
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach import tree

from ._scheme import Scheme


def _default_points() -> Float[Array, "q=1 J=3"]:
    return jnp.ones((1, 3)) / 4.0


def _default_weights() -> Float[Array, "q=1"]:
    return jnp.ones((1,)) / 6.0


@tree.define
class QuadratureTetra(Scheme):
    points: Float[Array, "q=1 J=3"] = tree.array(factory=_default_points)
    weights: Float[Array, "q=1"] = tree.array(factory=_default_weights)

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
