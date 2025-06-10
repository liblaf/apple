from typing import Self

import jax
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim import geometry as _g


class Domain(struct.Node):
    geometry: _g.Geometry = struct.data(default=None)

    @classmethod
    def from_geometry(cls, geometry: _g.Geometry) -> Self:
        return cls(geometry=geometry)

    # region Delegation

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.geometry.area

    @property
    def boundary(self) -> "Domain":
        raise NotImplementedError

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

    # endregion Delegation
