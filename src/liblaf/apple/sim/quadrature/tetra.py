import felupe
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import struct

from .scheme import Scheme


def _default_points() -> Float[Array, "q=1 J=3"]:
    return jnp.ones((1, 3)) / 4


def _default_weights() -> Float[Array, "q=1"]:
    return jnp.ones((1,)) / 6


@struct.pytree
class QuadratureTetra(Scheme):
    points: Float[Array, "q=1 J=3"] = struct.array(factory=_default_points)
    weights: Float[Array, "q=1"] = struct.array(factory=_default_weights)

    @classmethod
    def from_order(cls, order: int = 1, /) -> "QuadratureTetra":
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
