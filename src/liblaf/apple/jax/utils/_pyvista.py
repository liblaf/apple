import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer

from liblaf.apple.jax import math


def get_point_id(mesh: pv.DataSet) -> Integer[Array, " p"]:
    return math.asarray(mesh.point_data["point-id"], dtype=jnp.int32)


def get_cells_local(mesh: pv.DataSet) -> Integer[Array, "c a"]:
    if isinstance(mesh, pv.PolyData):
        return math.asarray(mesh.regular_faces, dtype=jnp.int32)
    if isinstance(mesh, pv.UnstructuredGrid):
        return math.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=jnp.int32)  # pyright: ignore[reportArgumentType]
    raise NotImplementedError


def get_cells_global(mesh: pv.DataSet) -> Integer[Array, "c a"]:
    point_id: Integer[Array, " p"] = get_point_id(mesh)
    cells_local: Integer[Array, "c a"] = get_cells_local(mesh)
    return point_id[cells_local]
