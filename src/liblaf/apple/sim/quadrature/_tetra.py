from typing import Self

import felupe
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct

from ._scheme import Scheme


def _default_points() -> Float[jax.Array, "a=4 J=3"]:
    with jax.ensure_compile_time_eval():
        return jnp.ones((1, 3)) / 4


def _default_weights() -> Float[jax.Array, "q=1"]:
    with jax.ensure_compile_time_eval():
        return jnp.ones((1,)) / 6


class QuadratureTetra(Scheme):
    _points: Float[jax.Array, "a J"] = struct.array(factory=_default_points)
    _weights: Float[jax.Array, " q"] = struct.array(factory=_default_weights)

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
