from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple.sim.abc import Element


class ElementTriangle(Element):
    """A 2D triangle element formulation with linear shape functions.

    References:
        1. [felupe.Triangle](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Triangle)
    """

    @property
    def points(self) -> Float[jax.Array, "points=3 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray([[0, 0], [1, 0], [0, 1]], dtype=float)

    @override
    def function(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[jax.Array, "points=3"]:
        coords = jnp.asarray(coords)
        r, s = coords
        return jnp.asarray([1 - r - s, r, s], dtype=float)

    @override
    def gradient(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[jax.Array, "points=3 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray([[-1, -1], [1, 0], [0, 1]], dtype=float)

    @override
    def hessian(
        self, coords: Float[ArrayLike, "dim=2"], /
    ) -> Float[jax.Array, "points=3 dim=2 dim=2"]:
        with jax.ensure_compile_time_eval():
            return jnp.zeros((3, 2, 2))
