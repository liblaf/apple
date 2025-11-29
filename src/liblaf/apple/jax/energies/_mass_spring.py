import logging
from typing import Self, override

import einops
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import LENGTH, POINT_ID, STIFFNESS
from liblaf.apple.jax.model import JaxEnergy

logger: logging.Logger = logging.getLogger(__name__)

type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@tree.define
class MassSpring(JaxEnergy):
    edges: Integer[Array, " edges 2"]
    length: Float[Array, " edges"]
    points: Float[Array, "edges 2 3"]
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
            points=jnp.asarray(obj.points[edges]),
            stiffness=jnp.asarray(obj.cell_data[STIFFNESS]),
        )

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    @override
    def fun(self, u: Vector) -> Scalar:
        x: Float[Array, "edges 2 3"] = self.points + u[self.edges]
        delta: Float[Array, "edges 3"] = x[:, 1, :] - x[:, 0, :]
        energy: Float[Array, " edges"] = (
            0.5
            * self.stiffness
            * jnp.square(jnp.linalg.norm(delta, axis=-1) - self.length)
        )
        return jnp.sum(energy)

    @override
    def grad(self, u: Vector) -> Updates:
        x: Float[Array, "edges 2 3"] = self.points + u[self.edges]
        delta: Float[Array, "edges 3"] = x[:, 1, :] - x[:, 0, :]
        length: Float[Array, " edges"] = jnp.linalg.norm(delta, axis=-1)
        direction: Float[Array, "edges 3"] = (
            delta / jnp.where(length > 0, length, 1.0)[:, jnp.newaxis]
        )
        force: Float[Array, "edges 3"] = (
            self.stiffness[:, jnp.newaxis]
            * (length - self.length)[:, jnp.newaxis]
            * direction
        )
        grad: Float[Array, "edges 2 3"] = jnp.stack([-force, force], axis=1)
        return grad, self.edges.flatten()

    @override
    def hess_diag(self, u: Vector) -> Updates:
        values: Float[Array, "edges*2 3"] = einops.repeat(
            self.stiffness, "edges -> (edges i) j", i=2, j=3
        )
        return values, self.edges.flatten()
