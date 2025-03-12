import jax
import jax.numpy as jnp
import pytest
import pytest_codspeed
import pytetwild
import pyvista as pv
from jaxtyping import Float, Integer, PRNGKeyArray

from liblaf import apple


@pytest.fixture(scope="package")
def mesh() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(surface)
    return mesh


def potential(
    F: Float[jax.Array, "3 3"],
    lmbda: Float[jax.Array, ""] = 3.0,  # pyright: ignore[reportArgumentType]
    mu: Float[jax.Array, ""] = 1.0,  # pyright: ignore[reportArgumentType]
) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "3 3"]
    R, _S = apple.polar_rv(F)
    R = jax.lax.stop_gradient(R)  # TODO: support gradient of `polar_rv()`
    Psi: Float[jax.Array, ""] = (
        mu * jnp.sum((F - R) ** 2) + lmbda * (jnp.linalg.det(F) - 1) ** 2
    )
    return Psi


@apple.jit()
def fun(
    u: Float[jax.Array, "C 4 3"], dh_dX: Float[jax.Array, "C 3 4"]
) -> Float[jax.Array, "C 4"]:
    F: Float[jax.Array, "C 3 3"] = apple.elem.tetra.deformation_gradient(u, dh_dX)
    Psi: Float[jax.Array, "C 4"] = jax.vmap(potential)(F)
    return jnp.sum(Psi, axis=-1)


def test_potential(
    mesh: pv.UnstructuredGrid, benchmark: pytest_codspeed.BenchmarkFixture
) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    cells: Integer[jax.Array, "C 4"] = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA])
    u: Float[jax.Array, "P 3"] = jax.random.uniform(key, (mesh.n_points, 3))
    u: Float[jax.Array, "C 4 3"] = u[cells]
    dh_dX: Float[jax.Array, "C 3 4"] = apple.elem.tetra.dh_dX(
        jnp.asarray(mesh.points)[cells]
    )
    for _ in range(3):
        jax.block_until_ready(fun(u, dh_dX))

    def run() -> None:
        for _ in range(100):
            jax.block_until_ready(fun(u, dh_dX))

    benchmark(run)
