from collections.abc import Iterator, Mapping, MutableMapping
from typing import Self

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e


class MeshAttributes(Mapping[str, jax.Array], struct.PyTree):
    attributes: pv.DataSetAttributes = struct.static(default=None)

    def __getitem__(self, key: str) -> jax.Array:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.attributes[key])

    def __iter__(self) -> Iterator[str]:
        yield from self.attributes.keys()

    def __len__(self) -> int:
        return len(self.attributes)


class Mesh(struct.PyTree):
    _pyvista: pv.DataSet = struct.static(default=None)

    # region Shape

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_cells(self) -> int:
        return self.pyvista.n_cells

    @property
    def n_points(self) -> int:
        return self.pyvista.n_points

    # endregion Shape

    @property
    def boundary(self) -> "Mesh":
        raise NotImplementedError

    @property
    def cell_data(self) -> MeshAttributes:
        with jax.ensure_compile_time_eval():
            return MeshAttributes(self.pyvista.cell_data)

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        raise NotImplementedError

    @property
    def element(self) -> _e.Element:
        raise NotImplementedError

    @property
    def field_data(self) -> MeshAttributes:
        with jax.ensure_compile_time_eval():
            return MeshAttributes(self.pyvista.field_data)

    @property
    def point_data(self) -> MeshAttributes:
        with jax.ensure_compile_time_eval():
            return MeshAttributes(self.pyvista.point_data)

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.points)

    @property
    def pyvista(self) -> pv.DataSet:
        return self._pyvista

    @property
    def user_dict(self) -> MutableMapping:
        return self.pyvista.user_dict

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, "points J"]) -> Self:
        mesh: pv.DataSet = self.pyvista
        mesh.point_data["displacement"] = displacement
        mesh = mesh.warp_by_vector("displacement")
        return self.evolve(pyvista=mesh)
