from typing import Self

import felupe
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float


@jarp.define
class Scheme:
    points: Float[Array, "quadrature dim"] = jarp.array()
    weights: Float[Array, " quadrature"] = jarp.array()

    @classmethod
    def from_felupe(cls, schema: felupe.quadrature.Scheme) -> Self:
        return cls(
            points=jnp.asarray(schema.points), weights=jnp.asarray(schema.weights)
        )

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]
