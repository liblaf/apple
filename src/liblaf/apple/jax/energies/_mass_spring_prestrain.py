import logging
from typing import Self

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import LENGTH, POINT_ID, PRESTRAIN, STIFFNESS

from ._mass_spring import MassSpring

logger = logging.getLogger(__name__)


type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@tree.define
class MassSpringPrestrain(MassSpring):
    @classmethod
    def from_pyvista(cls, obj: pv.PolyData) -> Self:
        if LENGTH not in obj.cell_data:
            obj = obj.compute_cell_sizes(length=True, area=False, volume=False)  # pyright: ignore[reportAssignmentType]
        point_id: Integer[np.ndarray, " points"] = obj.point_data[POINT_ID]
        edges: Integer[np.ndarray, "edges 2"] = obj.lines.reshape((-1, 3))[:, 1:]
        length: Float[Array, " edges"] = jnp.asarray(obj.cell_data[LENGTH])
        if jnp.any(length < 0.0):
            logger.warning("Length < 0")
        prestrain: Float[Array, " edges"] = jnp.asarray(obj.cell_data[PRESTRAIN])
        return cls(
            edges=jnp.asarray(point_id[edges]),
            length=length * (1.0 + prestrain),
            points=jnp.asarray(obj.points[edges]),
            stiffness=jnp.asarray(obj.cell_data[STIFFNESS]),
        )
