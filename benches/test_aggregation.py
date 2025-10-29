import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import pyvista as pv
import warp as wp
import warp.jax_experimental.ffi
import warp.types as wpt
from jaxtyping import Array
from pytest_codspeed import BenchmarkFixture

from liblaf import melon

if not wp.is_cuda_available():
    pytest.skip(reason="CUDA not available", allow_module_level=True)


def ids(param: float) -> str:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=param)
    return f"lr={param},P={mesh.n_points},C={mesh.n_cells}"


@pytest.fixture(scope="module", params=[0.05, 0.02, 0.01], ids=ids)
def mesh(request: pytest.FixtureRequest) -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=request.param)
    return mesh


@pytest.mark.benchmark(group="hello")
def test_atomic_add(benchmark: BenchmarkFixture, mesh: pv.UnstructuredGrid) -> None:
    @wp.kernel
    def kernel(
        points: wp.array(dtype=wpt.vec3d),
        cells: wp.array(dtype=wpt.vec4i),
        output: wp.array(dtype=wpt.float64),
    ) -> None:
        tid = wp.tid()
        cell = cells[tid]
        v0 = cell[0]
        v1 = cell[1]
        v2 = cell[2]
        v3 = cell[3]
        p0 = points[v0]
        p1 = points[v1]
        p2 = points[v2]
        p3 = points[v3]
        volume = wp.abs(
            wp.determinant(wp.matrix_from_cols(p0 - p3, p1 - p3, p2 - p3))
        ) / type(p0[0])(6.0)
        output[v0] += volume / type(volume)(4.0)
        output[v1] += volume / type(volume)(4.0)
        output[v2] += volume / type(volume)(4.0)
        output[v3] += volume / type(volume)(4.0)

    @warp.jax_experimental.ffi.jax_callable
    def callable_(
        points: wp.array(dtype=wpt.vec3d),
        cells: wp.array(dtype=wpt.vec4i),
        output: wp.array(dtype=wpt.float64),
    ) -> None:
        output.zero_()
        wp.launch(kernel, dim=cells.shape, inputs=[points, cells], outputs=[output])

    @eqx.filter_jit
    def fun(points: Array, cells: Array) -> Array:
        output: Array
        (output,) = callable_(points, cells, output_dims={"output": (points.shape[0],)})
        return output

    def target(points: Array, cells: Array) -> Array:
        output: Array = fun(points, cells)
        return jax.block_until_ready(output)

    points: Array = jnp.asarray(mesh.points, dtype=jnp.float64)
    cells: Array = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=jnp.int32)  # pyright: ignore[reportArgumentType]
    points = jax.block_until_ready(points)
    cells = jax.block_until_ready(cells)

    output: Array = benchmark(target, points, cells)  # pyright: ignore[reportArgumentType]

    assert output.sum().item() == pytest.approx(mesh.volume)


@pytest.mark.benchmark(group="hello")
def test_segment_sum(benchmark: BenchmarkFixture, mesh: pv.UnstructuredGrid) -> None:
    @warp.jax_experimental.ffi.jax_kernel
    @wp.kernel
    def kernel(
        points: wp.array(dtype=wpt.matrix((4, 3), wpt.float64)),
        output: wp.array(dtype=wpt.vec4d),
    ) -> None:
        tid = wp.tid()
        p0 = points[tid][0]
        p1 = points[tid][1]
        p2 = points[tid][2]
        p3 = points[tid][3]
        volume = wp.abs(
            wp.determinant(wp.matrix_from_cols(p0 - p3, p1 - p3, p2 - p3))
        ) / type(p0[0])(6.0)
        output[tid][0] = volume / type(volume)(4.0)
        output[tid][1] = volume / type(volume)(4.0)
        output[tid][2] = volume / type(volume)(4.0)
        output[tid][3] = volume / type(volume)(4.0)

    @eqx.filter_jit
    def fun(points: Array, cells: Array) -> Array:
        output: Array
        (output,) = kernel(
            points[cells],
            output_dims={"output": (cells.shape[0],)},
            launch_dims=(cells.shape[0],),
        )
        output = jax.ops.segment_sum(
            output.flatten(), cells.flatten(), num_segments=points.shape[0]
        )
        return output

    def target(points: Array, cells: Array) -> Array:
        output: Array = fun(points, cells)
        return jax.block_until_ready(output)

    points: Array = jnp.asarray(mesh.points, dtype=jnp.float64)
    cells: Array = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=jnp.int32)  # pyright: ignore[reportArgumentType]
    points = jax.block_until_ready(points)
    cells = jax.block_until_ready(cells)

    output: Array = benchmark(target, points, cells)  # pyright: ignore[reportArgumentType]

    assert output.sum().item() == pytest.approx(mesh.volume)
