from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer

from liblaf.apple import elem, testing

from ._geometry import Geometry


class Domain(flax.struct.PyTreeNode):
    cells: Integer[jax.Array, "cells 4"] = flax.struct.field()
    points: Integer[jax.Array, "points 3"] = flax.struct.field()
    geometry: Geometry = flax.struct.field(pytree_node=False)

    cell_mass: Float[jax.Array, " cells"] = flax.struct.field(default=None)
    cell_sizes: Float[jax.Array, "cells"] = flax.struct.field(default=None)
    dh_dX: Float[jax.Array, "cells 4 3"] = flax.struct.field(default=None)
    point_mass: Float[jax.Array, " points"] = flax.struct.field(default=None)

    def __post_init__(self) -> None:
        testing.assert_shape(self.cells, (self.n_cells, 4))

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> Self:
        cells: Integer[jax.Array, "cells 4"] = jnp.asarray(geometry.cells, dtype=int)
        points: Float[jax.Array, "points 3"] = jnp.asarray(geometry.points)
        dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points[cells])
        return cls(
            cells=cells,
            dh_dX=dh_dX,
            points=points,
            geometry=geometry,
            cell_mass=geometry.cell_mass,
            cell_sizes=geometry.cell_sizes,
            point_mass=geometry.point_mass,
        )

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def n_points(self) -> int:
        return self.geometry.n_points
