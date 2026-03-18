from typing import Self, override

import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf.apple.consts import GLOBAL_POINT_ID
from liblaf.apple.jax.model import JaxEnergy, JaxEnergyState

type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@jarp.define
class JaxPointForce(JaxEnergy):
    force: Vector = jarp.field()
    indices: Index = jarp.field()

    @classmethod
    def from_pyvista(cls, obj: pv.DataSet) -> Self:
        return cls(
            force=jnp.asarray(obj.point_data["Force"]),
            indices=jnp.asarray(obj.point_data[GLOBAL_POINT_ID]),
        )

    @override
    @jarp.jit(inline=True)
    def fun(self, state: JaxEnergyState, u: Vector) -> Scalar:
        return -jnp.vdot(self.force, u[self.indices])

    @override
    @jarp.jit(inline=True)
    def hess_diag(self, state: JaxEnergyState, u: Vector) -> Updates:
        return jnp.zeros_like(u[self.indices]), self.indices
