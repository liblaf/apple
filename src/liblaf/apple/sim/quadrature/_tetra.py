import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct
from liblaf.apple.sim.abc import Scheme


def _default_points() -> Float[jax.Array, "q=1 J=3"]:
    return jnp.ones((1, 3)) / 4


def _default_weights() -> Float[jax.Array, "q=1"]:
    return jnp.ones((1,)) / 6


class QuadratureTetra(Scheme):
    _points: Float[jax.Array, "q=1 J=3"] = struct.array(factory=_default_points)
    _weights: Float[jax.Array, "q=1"] = struct.array(factory=_default_weights)

    @classmethod
    def from_order(cls, order: int = 1, /) -> "QuadratureTetra":
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order))
