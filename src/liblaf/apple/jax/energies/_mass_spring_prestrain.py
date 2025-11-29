import logging
from typing import Self, override

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import LENGTH, POINT_ID, PRESTRAIN, STIFFNESS
from liblaf.apple.jax import math
from liblaf.apple.jax.model import JaxEnergy

logger = logging.getLogger(__name__)


type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@tree.define
class MassSpringPrestrain(JaxEnergy):
    edges: Integer[Array, " edges 2"]
    length: Float[Array, " edges"]
    prestrain: Float[Array, " edges"]
    stiffness: Float[Array, " edges"]

    @classmethod
    def from_pyvista(cls, obj: pv.PolyData) -> Self:
        if LENGTH not in obj.cell_data:
            obj = obj.compute_cell_sizes(length=True, area=False, volume=False)  # pyright: ignore[reportAssignmentType]
        point_id: Integer[np.ndarray, " points"] = obj.point_data[POINT_ID]
        edges: Integer[np.ndarray, "edges 2"] = obj.lines.reshape((-1, 3))[:, 1:]
        length: Float[Array, " edges"] = jnp.asarray(obj.cell_data[LENGTH])
        if jnp.any(length < 0.0):
            logger.warning("Length < 0")
        return cls(
            edges=jnp.asarray(point_id[edges]),
            length=length,
            prestrain=jnp.asarray(obj.cell_data[PRESTRAIN]),
            stiffness=jnp.asarray(obj.cell_data[STIFFNESS]),
        )

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    @override
    def fun(self, u: Vector) -> Scalar:
        u_local: Float[Array, "edges 2 3"] = u[self.edges]
        delta: Float[Array, "edges 3"] = u_local[:, 1, :] - u_local[:, 0, :]
        delta -= (
            self.prestrain[:, jnp.newaxis]
            * self.length[:, jnp.newaxis]
            * math.normalize(jax.lax.stop_gradient(delta))
        )
        energy: Float[Array, " edges"] = 0.5 * self.stiffness * jnp.vecdot(delta, delta)
        return jnp.sum(energy)

    @override
    def grad(self, u: Vector) -> Updates:
        u_local: Float[Array, "edges 2 3"] = u[self.edges]
        delta: Float[Array, "edges 3"] = u_local[:, 1, :] - u_local[:, 0, :]
        delta -= (
            self.prestrain[:, jnp.newaxis]
            * self.length[:, jnp.newaxis]
            * math.normalize(jax.lax.stop_gradient(delta))
        )
        force: Float[Array, "edges 3"] = self.stiffness[:, jnp.newaxis] * delta
        values: Float[Array, "edges 2 3"] = jnp.stack([-force, force], axis=1)
        return values.reshape((self.n_edges * 2, 3)), self.edges.flatten()

    @override
    def hess_diag(self, u: Vector) -> Updates:
        values: Float[Array, "edges*2 3"] = einops.repeat(
            self.stiffness, "edges -> (edges i) j", i=2, j=3
        )
        return values, self.edges.flatten()
