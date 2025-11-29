from typing import Self

import felupe.quadrature
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach import tree


@tree.define
class Scheme:
    points: Float[Array, "q J"] = tree.array()
    weights: Float[Array, " q"] = tree.array()

    @classmethod
    def from_felupe(cls, scheme: felupe.quadrature.Scheme) -> Self:
        return cls(
            points=jnp.asarray(scheme.points), weights=jnp.asarray(scheme.weights)
        )

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
