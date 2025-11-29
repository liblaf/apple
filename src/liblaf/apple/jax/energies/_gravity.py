from typing import Self, override

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, ArrayLike, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import MASS, POINT_ID
from liblaf.apple.jax.model import JaxEnergy

type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


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

    @override
    def fun(self, u: Vector) -> Scalar:
        u = u[self.indices]
        return -jnp.vdot(self.mass, jnp.vecdot(u, self.gravity, axis=-1))

    @override
    def hess_diag(self, u: Vector) -> Updates:
        return jnp.zeros_like(u[self.indices]), self.indices
