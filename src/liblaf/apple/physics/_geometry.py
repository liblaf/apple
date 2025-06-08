from typing import Self

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import elem


@attrs.define
class Geometry:
    mesh: pv.UnstructuredGrid

    id: str = "geometry"

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def cell_mass(self) -> Float[jax.Array, "cells"]:
        return self.density * self.cell_sizes

    @property
    def cell_sizes(self) -> Float[jax.Array, "cells"]:
        if "Volume" not in self.mesh.cell_data:
            self.mesh = self.mesh.compute_cell_sizes()
        return jnp.asarray(self.mesh.cell_data["Volume"])

    @property
    def cells(self) -> Integer[jax.Array, "cells 4"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA], dtype=int)

    @property
    def density(self) -> Float[jax.Array, "cells"]:
        if "density" not in self.mesh.cell_data:
            return jnp.ones((self.n_cells,))
        return jnp.asarray(self.mesh.cell_data["density"])

    @density.setter
    def density(self, values: Float[ArrayLike, "cells"]) -> None:
        values: Float[np.ndarray, " cells"] = np.asarray(values)
        values = np.broadcast_to(values, (self.n_cells,))
        self.mesh.cell_data["density"] = values

    @property
    def length(self) -> float:
        return self.mesh.length

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def point_mass(self) -> Float[jax.Array, " points"]:
        cell_mass: Float[jax.Array, " cells"] = self.cell_mass
        cell_mass: Float[jax.Array, "cells 4"] = (
            jnp.broadcast_to(cell_mass[:, None], (self.n_cells, 4)) / 4.0
        )
        point_mass: Float[jax.Array, " points"] = elem.tetra.segment_sum(
            cell_mass, self.cells, n_points=self.n_points
        )
        return point_mass

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return jnp.asarray(self.mesh.points)

    def warp(
        self,
        displacements: Float[jax.Array, "points 3"],
        *,
        velocities: Float[jax.Array, "points 3"] | None = None,
    ) -> Self:
        mesh: pv.UnstructuredGrid = self.mesh.copy()
        mesh.point_data["displacements"] = displacements
        mesh.warp_by_vector("displacements", inplace=True)
        if velocities is not None:
            mesh.point_data["velocities"] = velocities
        return type(self)(mesh=mesh, id=self.id)
