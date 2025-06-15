from collections.abc import MutableMapping
from typing import Self

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim.abc.element import Element
from liblaf.apple.sim.abc.quadrature import Scheme

from ._attributes import GeometryAttributes


class Geometry(struct.PyTree):
    _pyvista: pv.DataSet = struct.static(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet) -> Self:
        raise NotImplementedError

    # region Structure

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def pyvista(self) -> pv.DataSet:
        return self._pyvista

    @property
    def quadrature(self) -> Scheme:
        return self.element.quadrature

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> int:
        return 3

    @property
    def n_cells(self) -> int:
        return self.pyvista.n_cells

    @property
    def n_points(self) -> int:
        return self.pyvista.n_points

    # endregion Shape

    # region Array

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        raise NotImplementedError

    @property
    def points(self) -> Float[jax.Array, "points dim"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.points)

    # endregion Array

    # region Attributes

    @property
    def cell_data(self) -> GeometryAttributes:
        return GeometryAttributes(self.pyvista.cell_data)

    @property
    def field_data(self) -> GeometryAttributes:
        return GeometryAttributes(self.pyvista.field_data)

    @property
    def original_cell_id(self) -> Integer[jax.Array, "cells"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.cell_data["cell-id"])

    @property
    def original_point_id(self) -> Integer[jax.Array, "points"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.point_data["point-id"])

    @property
    def point_data(self) -> GeometryAttributes:
        return GeometryAttributes(self.pyvista.point_data)

    @property
    def user_dict(self) -> MutableMapping:
        return self.pyvista.user_dict

    # endregion Attributes

    # region Geometric Operations

    @property
    def boundary(self) -> "Geometry":
        raise NotImplementedError

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        raise NotImplementedError

    def warp_by_vector(self, displacement: Float[ArrayLike, "points dim"]) -> Self:
        mesh: pv.DataSet = self.pyvista.copy()
        mesh.point_data["displacement"] = displacement
        mesh = mesh.warp_by_vector("displacement")
        return self.evolve(_pyvista=mesh)

    # endregion Geometric Operations
