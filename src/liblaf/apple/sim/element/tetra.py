from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple import struct
from liblaf.apple.sim.quadrature import QuadratureTetra

from .element import Element


@struct.pytree
class ElementTetra(Element):
    @property
    @override
    def points(self) -> Float[Array, "points=4 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            )

    @property
    @override
    def quadrature(self) -> QuadratureTetra:
        with jax.ensure_compile_time_eval():
            return QuadratureTetra.from_order(1)

    @override
    def function(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[Array, "points=4"]:
        coords = jnp.asarray(coords)
        r, s, t = coords
        return jnp.asarray([1 - r - s - t, r, s, t])

    @override
    def gradient(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[Array, "points=4 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            )

    @override
    def hessian(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[Array, "points=4 dim=3 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.zeros((4, 3, 3), dtype=float)
