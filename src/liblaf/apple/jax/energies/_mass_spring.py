from typing import Self, override

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import POINT_ID, STIFFNESS
from liblaf.apple.jax.model import JaxEnergy

type Vector = Float[Array, "points dim"]
type Scalar = Float[Array, ""]


@tree.define
class MassSpring(JaxEnergy):
    edges: Integer[Array, " edges 2"]
    stiffness: Float[Array, " edges"]

    @classmethod
    def from_pyvista(cls, obj: pv.PolyData) -> Self:
        point_id: Integer[np.ndarray, " points"] = obj.point_data[POINT_ID]
        edges: Integer[np.ndarray, "edges 2"] = obj.lines.reshape((-1, 3))[:, 1:]
        return cls(
            edges=jnp.asarray(point_id[edges]),
            stiffness=jnp.asarray(obj.cell_data[STIFFNESS]),
        )

    @override
    def fun(self, u: Vector) -> Scalar:
        u_local: Float[Array, " edges 2 3"] = u[self.edges]
        delta: Float[Array, "edges 3"] = u_local[:, 1, :] - u_local[:, 0, :]
        energy: Float[Array, " edges"] = 0.5 * self.stiffness * jnp.vecdot(delta, delta)
        return jnp.sum(energy)
