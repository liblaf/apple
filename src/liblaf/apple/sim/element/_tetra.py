from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple.sim.abc import Element


class ElementTetra(Element):
    """A 3D tetrahedron element formulation with linear shape functions.

    References:
        1. [felupe.Tetra](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Tetra)
    """

    @property
    @override
    def points(self) -> Float[jax.Array, "points=4 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            )

    @override
    def function(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[jax.Array, "points=4"]:
        coords = jnp.asarray(coords)
        r, s, t = coords
        return jnp.asarray([1 - r - s - t, r, s, t])

    @override
    def gradient(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[jax.Array, "points=4 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            )

    @override
    def hessian(
        self, coords: Float[ArrayLike, "dim=3"], /
    ) -> Float[jax.Array, "points=4 dim=3 dim=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.zeros((4, 3, 3))
