from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple import struct

from .element import Element


@struct.pytree
class ElementTriangle(Element):
    @property
    @override
    def points(self) -> Float[Array, "points=3 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray([[0, 0], [1, 0], [0, 1]], dtype=float)

    @override
    def function(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[Array, "points=3"]:
        coords = jnp.asarray(coords)
        r, s = coords
        return jnp.asarray([1 - r - s, r, s])

    @override
    def gradient(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[Array, "points=3 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray([[-1, -1], [1, 0], [0, 1]], dtype=float)

    @override
    def hessian(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[Array, "points=3 dim=2 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.zeros((3, 2, 2), dtype=float)
