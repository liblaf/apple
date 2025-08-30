from typing import Self

import felupe.quadrature
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import struct

from .scheme import Scheme


def _default_points() -> Float[Array, "q=1 J=3"]:
    with jax.ensure_compile_time_eval():
        return jnp.ones((1, 3)) / 4.0


def _default_weights() -> Float[Array, "q=1"]:
    with jax.ensure_compile_time_eval():
        return jnp.ones((1,)) / 6.0


class QuadratureTetra(Scheme):
    points: Float[Array, "q=1 J=3"] = struct.array(factory=_default_points)
    weights: Float[Array, "q=1"] = struct.array(factory=_default_weights)

    @classmethod
    def from_order(cls, order: int = 1) -> Self:
        with jax.ensure_compile_time_eval():
            return cls.from_felupe(felupe.quadrature.Tetrahedron(order=order))
