from collections.abc import Iterator, MutableMapping
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e


class GeometryAttributes(struct.PyTree, MutableMapping[str, jax.Array]):
    attributes: pv.DataSetAttributes = struct.static(default=None)

    def __getitem__(self, key: str) -> jax.Array:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.attributes[key])

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        self.attributes[key] = np.asarray(value)

    def __delitem__(self, key: str) -> None:
        del self.attributes[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.attributes.keys()

    def __len__(self) -> int:
        return len(self.attributes)


class Geometry(struct.PyTree):
    _pyvista: pv.DataSet = struct.static(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet) -> Self:
        raise NotImplementedError

    # region Structure

    @property
    def element(self) -> _e.Element:
        raise NotImplementedError

    @property
    def pyvista(self) -> pv.DataSet:
        return self._pyvista

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

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, "points dim"]) -> Self:
        mesh: pv.DataSet = self.pyvista.copy()
        mesh.point_data["displacement"] = displacement
        mesh = mesh.warp_by_vector("displacement")
        return self.evolve(_pyvista=mesh)

    # endregion Geometric Operations
