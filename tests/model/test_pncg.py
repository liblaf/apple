import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float
from liblaf.peach.optim import Result

from liblaf.apple.consts import DIRICHLET_MASK, DIRICHLET_VALUE, MU
from liblaf.apple.model import Forward, ModelBuilder
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpArap

type Vector = Float[Array, " 1"]


class LogBarrierObjective:
    def update(self, _state: Vector, params: Vector) -> Vector:
        return params

    def fun(self, state: Vector):
        x = state[0]
        x_safe = jnp.where(x > 0.0, x, 1.0)
        value = 0.5 * jnp.square(x + 2.0) - 1e-2 * jnp.log(x_safe)
        return jnp.where(x > 0.0, value, jnp.inf)

    def grad(self, state: Vector) -> Vector:
        x = state[0]
        return jnp.asarray([x + 2.0 - 1e-2 / x])

    def hess_diag(self, state: Vector) -> Vector:
        x = state[0]
        return jnp.asarray([1.0 + 1e-2 / jnp.square(x)])

    def hess_quad(self, state: Vector, p: Vector):
        x = state[0]
        return (1.0 + 1e-2 / jnp.square(x)) * jnp.square(p[0])


def test_pncg_backtracks_to_stay_in_domain() -> None:
    optimizer = PNCG(
        jit=False,
        max_delta=jnp.asarray(jnp.inf),
        max_steps=jnp.asarray(64),
        max_line_search_steps=jnp.asarray(12),
        stagnation_max_restarts=jnp.asarray(8),
        stagnation_patience=jnp.asarray(2),
        atol=jnp.asarray(1e-12),
        rtol=jnp.asarray(1e-8),
        atol_primary=jnp.asarray(1e-12),
        rtol_primary=jnp.asarray(1e-8),
    )
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])

    solution, state = optimizer.minimize(objective, params0, params0)

    assert solution.result in {Result.PRIMARY_SUCCESS, Result.SECONDARY_SUCCESS}
    assert np.isfinite(np.asarray(solution.params)).all()
    assert np.isfinite(np.asarray(objective.fun(state)))
    assert np.asarray(solution.params)[0] > 0.0


def test_pncg_uses_backtracking_step_instead_of_invalid_quadratic_step() -> None:
    optimizer = PNCG(
        jit=False,
        max_delta=jnp.asarray(jnp.inf),
        max_steps=jnp.asarray(1),
        max_line_search_steps=jnp.asarray(12),
    )
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])
    opt_state0, _ = optimizer.init(objective, params0, params0)

    state1, opt_state1 = optimizer.step(objective, params0, opt_state0)

    assert np.isfinite(np.asarray(objective.fun(state1)))
    assert np.asarray(opt_state1.params)[0] > 0.0
    assert 0.0 < np.asarray(opt_state1.alpha) < 1.0


def test_forward_uses_enhanced_pncg_by_default() -> None:
    mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()  # pyright: ignore[reportAssignmentType]
    mesh.point_data[DIRICHLET_MASK] = np.broadcast_to(
        (mesh.points[:, 1] < mesh.bounds.y_min + 1e-3)[:, np.newaxis],
        (mesh.n_points, 3),
    ).copy()
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))
    mesh.cell_data[MU] = np.ones((mesh.n_cells,))

    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(WarpArap.from_pyvista(mesh, name="elastic"))
    model = builder.finalize()

    forward = Forward(model)

    assert isinstance(forward.optimizer, PNCG)
