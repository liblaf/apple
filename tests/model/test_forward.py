import jax.numpy as jnp
import numpy as np
import pyvista as pv
from liblaf.peach.optim import PNCG

from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import DIRICHLET_MASK, DIRICHLET_VALUE, MU
from liblaf.apple.warp import Arap


def test_forward() -> None:
    mesh: pv.UnstructuredGrid = pv.examples.download_letter_a()  # pyright: ignore[reportAssignmentType]
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.point_data[DIRICHLET_MASK] = mesh.points[:, 1] < mesh.bounds.y_min + 0.1 * (
        mesh.bounds.y_max - mesh.bounds.y_min
    )
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))

    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic: Arap = Arap.from_pyvista(mesh)
    builder.add_energy(elastic)

    model: Model = builder.finalize()
    rng: np.random.Generator = np.random.default_rng()
    model.u_free = jnp.asarray(
        rng.uniform(-mesh.length, mesh.length, model.u_free.shape)
    )
    forward = Forward(model, optimizer=PNCG(max_steps=1000, rtol=1e-15))
    solution: PNCG.Solution = forward.step()
    assert solution.success
    np.testing.assert_allclose(model.u_full, 0.0, atol=1e-3 * mesh.length)
