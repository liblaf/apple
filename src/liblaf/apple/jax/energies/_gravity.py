from typing import Self

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, ArrayLike, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import MASS, POINT_ID
from liblaf.apple.jax.model import JaxEnergy

type Vector = Float[Array, "points dim"]
type Scalar = Float[Array, ""]


@tree.define
class Gravity(JaxEnergy):
    gravity: Float[Array, " dim"]
    indices: Integer[Array, " points"]
    mass: Float[Array, " points"]

    @classmethod
    def from_pyvista(
        cls, obj: pv.DataSet, gravity: Float[ArrayLike, " dim"] | None = None
    ) -> Self:
        if gravity is None:
            gravity = jnp.asarray([0.0, -9.81, 0.0])
        return cls(
            gravity=jnp.asarray(gravity),
            indices=jnp.asarray(obj.point_data[POINT_ID]),
            mass=jnp.asarray(obj.point_data[MASS]),
        )

    def fun(self, u: Vector) -> Scalar:
        u: Float[Array, "points dim"] = u[self.indices]
        return -jnp.vdot(self.mass, jnp.vecdot(u, self.gravity, axis=-1))
