import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf import apple


def point_mass(mesh: pv.UnstructuredGrid) -> Float[jax.Array, " V"]:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    cells: Integer[np.ndarray, "C 4"] = mesh.cells_dict[pv.CellType.TETRA]
    density: Float[np.ndarray, " C"] = mesh.cell_data["density"]
    volume: Float[np.ndarray, " C"] = mesh.cell_data["Volume"]
    cell_mass: Float[np.ndarray, " C"] = density * volume
    point_mass: Float[jax.Array, " V"] = apple.elem.tetra.segment_sum(
        jnp.asarray(cell_mass[cells]), jnp.asarray(cells), n_points=mesh.n_points
    )
    return point_mass
