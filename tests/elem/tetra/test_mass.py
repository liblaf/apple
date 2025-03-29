import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple


def random_density(n_cells: int) -> Float[jnp.ndarray, " C"]:
    key: PRNGKeyArray = jax.random.key(0)
    return jax.random.uniform(key, (n_cells,))


def mass_naive(mesh: pv.UnstructuredGrid) -> Float[jax.Array, " P"]:
    mesh = mesh.compute_cell_sizes(volume=True)  # pyright: ignore[reportAssignmentType]
    mesh.point_data["mass"] = 0.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data["mass"] = mesh.cell_data["density"] * mesh.cell_data["Volume"]
    for cell_id, cell in enumerate(mesh.cell):
        for point_id in cell.point_ids:
            mesh.point_data["mass"][point_id] += 0.25 * mesh.cell_data["mass"][cell_id]
    return jnp.asarray(mesh.point_data["mass"])


def test_mass(mesh_pv: pv.UnstructuredGrid) -> None:
    mesh_pv.cell_data["density"] = np.asarray(random_density(mesh_pv.n_cells))
    actual: Float[jax.Array, " P"] = apple.elem.tetra.mass(mesh_pv)
    expected: Float[jax.Array, " P"] = mass_naive(mesh_pv)
    np.testing.assert_allclose(actual, expected)
