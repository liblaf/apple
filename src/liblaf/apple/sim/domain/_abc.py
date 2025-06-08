from typing import Self

import flax.struct
import jax
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple.sim import geometry as _g


class Domain(flax.struct.PyTreeNode):
    geometry: _g.Geometry = flax.struct.field(default=None)

    @classmethod
    def from_geometry(cls, geometry: _g.Geometry) -> Self:
        return cls(geometry=geometry)

    # region Inherited

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.geometry.area

    @property
    def cells(self) -> Integer[jax.Array, "cells ..."]:
        return self.geometry.cells

    @property
    def mesh(self) -> pv.DataSet:
        return self.geometry.mesh

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def n_points(self) -> int:
        return self.geometry.n_points

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.geometry.points

    @property
    def volume(self) -> Float[jax.Array, " cells"]:
        return self.geometry.volume

    # endregion Inherited
