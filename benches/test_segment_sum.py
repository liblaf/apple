import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer, PRNGKeyArray
from pytest_benchmark.fixture import BenchmarkFixture

from liblaf import apple


def test_segment_sum(
    benchmark: BenchmarkFixture, mesh: pv.UnstructuredGrid, key: PRNGKeyArray
) -> None:
    cells: Integer[jax.Array, "C 4"] = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA])
    x: Float[jax.Array, "V 3"] = jax.random.uniform(key, (mesh.n_points, 3))
    x: Float[jax.Array, "C 4 3"] = x[cells]

    def fun(x: Float[jax.Array, "C 4 3"]) -> Float[jax.Array, "V 3"]:
        return apple.elem.tetra.segment_sum(x, cells, mesh.n_points)

    benchmark(fun, x)
