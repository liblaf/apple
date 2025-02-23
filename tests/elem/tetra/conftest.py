import felupe
import jax
import pytest
import pytetwild
import pyvista as pv
from jaxtyping import Float, PRNGKeyArray


@pytest.fixture(scope="package")
def mesh_pv() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return mesh


@pytest.fixture(scope="package")
def mesh_felupe(mesh_pv: pv.UnstructuredGrid) -> felupe.Mesh:
    mesh_pv = mesh_pv.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    mesh = felupe.Mesh(mesh_pv.points, mesh_pv.cells_dict[pv.CellType.TETRA], "tetra")
    mesh = mesh.flip(mesh_pv.cell_data["Volume"] < 0)
    return mesh


@pytest.fixture(scope="package")
def region(mesh_felupe: felupe.Mesh) -> felupe.RegionTetra:
    return felupe.RegionTetra(mesh_felupe)


@pytest.fixture(scope="package")
def displacement(mesh_pv: pv.UnstructuredGrid) -> Float[jax.Array, "c I=3"]:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    u: Float[jax.Array, "c I=3"] = jax.random.uniform(
        subkey, (mesh_pv.n_points, 3), minval=-mesh_pv.length, maxval=mesh_pv.length
    )
    return u
