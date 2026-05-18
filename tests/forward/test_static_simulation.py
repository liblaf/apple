import jax
import numpy as np
import pytest
import pyvista as pv
import warp as wp

jax.config.update("jax_enable_x64", val=True)


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="Warp JAX FFI is registered for the active GPU backend",
)


def _make_embedded_tetra_mesh(mu_name: str) -> pv.UnstructuredGrid:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            4,
            0,
            1,
            2,
            4,
            4,
            0,
            1,
            4,
            3,
            4,
            0,
            4,
            2,
            3,
            4,
            4,
            1,
            2,
            3,
        ],
        dtype=np.int64,
    )
    cell_types = np.full(4, pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells, cell_types, points)
    mesh.cell_data[mu_name] = np.ones(mesh.n_cells, dtype=np.float64)
    return mesh


def test_forward_static_simulation_end_to_end() -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE, MU
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Arap

    wp.init()

    mesh = _make_embedded_tetra_mesh(MU.vtk)
    builder = ModelBuilder()
    builder.add_vertices(mesh)

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[:4, :] = True
    fixed_value[3] = np.array([0.2, -0.1, 0.15])
    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value

    builder.add_fixed(mesh)
    builder.add_potential(Arap.from_pyvista(mesh))
    model = builder.finalize()
    forward = Forward(model)

    initial_energy = float(forward.problem.fun(forward.state))
    solution = forward.step()
    final_energy = float(forward.problem.fun(forward.state))

    assert solution.result.name == "PRIMARY_SUCCESS"
    assert model.n_free == 3
    assert np.isfinite(initial_energy)
    assert np.isfinite(final_energy)
    assert final_energy < initial_energy

    u_full = np.asarray(forward.state.u)
    np.testing.assert_allclose(u_full[:4], fixed_value[:4])
    np.testing.assert_allclose(u_full[4], [0.05, -0.025, 0.0375], atol=1e-8)
