from typing import override

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple.jax.fem.quadrature import QuadratureTetra

from ._element import Element


@jarp.define
class ElementTetra(Element):
    @property
    @override
    def points(self) -> Float[Array, "points dim"]:
        return jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @property
    @override
    def quadrature(self) -> QuadratureTetra:
        return QuadratureTetra()

    @override
    def function(self, coords: Float[Array, " dim"]) -> Float[Array, "points=4"]:
        r, s, t = coords
        return jnp.asarray([1.0 - r - s - t, r, s, t])

    @override
    def gradient(self, coords: Float[Array, " dim"]) -> Float[Array, "points dim"]:
        return jnp.asarray(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    @override
    def hessian(self, coords: Float[Array, " dim"]) -> Float[Array, "points dim dim"]:
        return jnp.zeros((4, 3, 3))
