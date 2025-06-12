from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from ._element import Element


class ElementTetra(Element):
    """A 3D tetrahedron element formulation with linear shape functions.

    References:
        1. [felupe.Tetra](https://felupe.readthedocs.io/en/latest/felupe/element.html#felupe.Tetra)
    """

    @property
    def n_points(self) -> int:
        return 4

    @override
    def function(self, coords: Float[ArrayLike, "3"], /) -> Float[jax.Array, "4"]:
        coords = jnp.asarray(coords)
        r, s, t = coords
        return jnp.asarray([1 - r - s - t, r, s, t])

    @override
    def gradient(self, coords: Float[ArrayLike, "3"], /) -> Float[jax.Array, "4 3"]:
        return jnp.asarray([[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
