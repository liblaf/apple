import attrs
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float
from liblaf.peach.optim import Result

from liblaf.apple.consts import DIRICHLET_MASK, DIRICHLET_VALUE, MU
from liblaf.apple.model import Forward, ModelBuilder
from liblaf.apple.optim import (
    PNCG,
    HessianDamping,
    LineSearchSettings,
)
from liblaf.apple.warp import WarpArap

type Vector = Float[Array, " 1"]


class LogBarrierObjective:
    def update_state(self, parameters: Vector) -> Vector:
        return parameters

    def update(self, _state: Vector, params: Vector) -> Vector:
        return params

    def objective_value(self, state: Vector):
        return self.fun(state)

    def fun(self, state: Vector):
        x = state[0]
        x_safe = jnp.where(x > 0.0, x, 1.0)
        value = 0.5 * jnp.square(x + 2.0) - 1e-2 * jnp.log(x_safe)
        return jnp.where(x > 0.0, value, jnp.inf)

    def gradient(self, state: Vector) -> Vector:
        return self.grad(state)

    def grad(self, state: Vector) -> Vector:
        x = state[0]
        return jnp.asarray([x + 2.0 - 1e-2 / x])

    def hessian_diagonal(self, state: Vector) -> Vector:
        return self.hess_diag(state)

    def hess_diag(self, state: Vector) -> Vector:
        x = state[0]
        return jnp.asarray([1.0 + 1e-2 / jnp.square(x)])

    def curvature_along(self, state: Vector, p: Vector):
        return self.hess_quad(state, p)

    def hess_quad(self, state: Vector, p: Vector):
        x = state[0]
        return (1.0 + 1e-2 / jnp.square(x)) * jnp.square(p[0])


class QuadraticObjective:
    def update_state(self, parameters: Vector) -> Vector:
        return parameters

    def update(self, _state: Vector, params: Vector) -> Vector:
        return params

    def objective_value(self, state: Vector):
        return self.fun(state)

    def fun(self, state: Vector):
        return 0.5 * jnp.square(state[0])

    def gradient(self, state: Vector) -> Vector:
        return self.grad(state)

    def grad(self, state: Vector) -> Vector:
        return jnp.asarray([state[0]])

    def hessian_diagonal(self, state: Vector) -> Vector:
        return self.hess_diag(state)

    def hess_diag(self, state: Vector) -> Vector:
        del state
        return jnp.asarray([1.0])

    def curvature_along(self, state: Vector, p: Vector):
        return self.hess_quad(state, p)

    def hess_quad(self, state: Vector, p: Vector):
        del state
        return jnp.square(p[0])


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


def test_pncg_termination_uses_grad_norm_instead_of_decrease() -> None:
    optimizer = PNCG(
        jit=False,
        atol=jnp.asarray(1e-12),
        rtol=jnp.asarray(1e-8),
        atol_primary=jnp.asarray(1e-12),
        rtol_primary=jnp.asarray(1e-8),
    )
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])

    opt_state, opt_stats = optimizer.init(objective, params0, params0)
    opt_state = attrs.evolve(
        opt_state,
        n_steps=jnp.asarray(1, dtype=jnp.int32),
        decrease=jnp.asarray(1e-12),
        first_decrease=jnp.asarray(1.0),
        first_grad_norm=jnp.asarray(1.0),
        grad=jnp.asarray([1e-1]),
        hess_diag=jnp.asarray([1.0]),
        hess_quad=jnp.asarray(1.0),
    )

    assert not bool(
        np.asarray(optimizer.terminate(objective, params0, opt_state, opt_stats))
    )
    assert optimizer.postprocess(
        objective, params0, opt_state, opt_stats
    ).result not in {
        Result.PRIMARY_SUCCESS,
        Result.SECONDARY_SUCCESS,
    }


def test_pncg_stagnation_detection_uses_value_instead_of_decrease() -> None:
    optimizer = PNCG(jit=False)
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])

    opt_state, _ = optimizer.init(objective, params0, params0)
    opt_state = attrs.evolve(
        opt_state,
        params=jnp.asarray([0.2]),
        decrease=jnp.asarray(10.0),
        best_decrease=jnp.asarray(1.0),
        best_value=jnp.asarray(0.0),
        best_params=jnp.asarray([0.1]),
        stagnation_counter=jnp.asarray(3, dtype=jnp.int32),
    )

    next_state = optimizer._detect_stagnation(opt_state, jnp.asarray(0.5))

    assert np.asarray(next_state.stagnation_counter) == 4
    assert np.asarray(next_state.best_params)[0] == np.asarray(opt_state.best_params)[0]

    improved_state = optimizer._detect_stagnation(opt_state, jnp.asarray(-0.5))

    assert np.asarray(improved_state.stagnation_counter) == 0
    assert np.asarray(improved_state.best_value) == -0.5
    assert np.asarray(improved_state.best_params)[0] == 0.2


def test_pncg_stats_include_gradient_and_value_history() -> None:
    optimizer = PNCG(jit=False)
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])

    opt_state, opt_stats = optimizer.init(objective, params0, params0)
    opt_state = attrs.evolve(
        opt_state,
        grad=jnp.asarray([3.0]),
        first_value=jnp.asarray(4.0),
        best_value=jnp.asarray(1.5),
        first_grad_norm=jnp.asarray(5.0),
        first_grad_norm_max=jnp.asarray(6.0),
        decrease=jnp.asarray(2.0),
        first_decrease=jnp.asarray(8.0),
    )

    opt_stats = optimizer.update_stats(objective, params0, opt_state, opt_stats)

    assert np.asarray(opt_stats.grad_norm) == 3.0
    assert np.asarray(opt_stats.grad_norm_max) == 3.0
    assert np.asarray(opt_stats.first_value) == 4.0
    assert np.asarray(opt_stats.best_value) == 1.5
    assert np.asarray(opt_stats.first_grad_norm) == 5.0
    assert np.asarray(opt_stats.first_grad_norm_max) == 6.0


def test_pncg_solve_returns_result_object() -> None:
    optimizer = PNCG(jit=False)
    objective = LogBarrierObjective()

    result = optimizer.solve(objective, initial_parameters=jnp.asarray([0.1]))

    assert result.success
    assert result.status in {"target_converged", "acceptable_converged"}
    assert result.final_iteration.regularization_strength == 0.0
    assert np.asarray(result.final_iteration.parameters)[0] > 0.0


def test_pncg_constant_damping_affects_preconditioner_and_curvature() -> None:
    optimizer = PNCG(
        jit=False, damping=HessianDamping(constant_strength=jnp.asarray(0.25))
    )
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])
    opt_state0, _ = optimizer.init(objective, params0, params0)

    _state1, opt_state1 = optimizer.step(objective, params0, opt_state0)

    expected_preconditioner = 1.0 / (objective.hess_diag(params0)[0] + 0.25)
    np.testing.assert_allclose(
        np.asarray(opt_state1.preconditioner)[0], expected_preconditioner
    )
    search_direction = np.asarray(opt_state1.search_direction)[0]
    expected_curvature = (
        objective.hess_quad(params0, opt_state1.search_direction)
        + 0.25 * search_direction * search_direction
    )
    np.testing.assert_allclose(
        np.asarray(opt_state1.hess_quad), np.asarray(expected_curvature)
    )


def test_pncg_rejected_step_forces_steepest_descent_next_iteration() -> None:
    optimizer = PNCG(
        jit=False,
        max_delta=jnp.asarray(jnp.inf),
        max_line_search_steps=jnp.asarray(0),
    )
    objective = LogBarrierObjective()
    params0 = jnp.asarray([0.1])
    opt_state0, _ = optimizer.init(objective, params0, params0)

    state1, opt_state1 = optimizer.step(objective, params0, opt_state0)
    assert np.asarray(opt_state1.force_steepest_descent_next)

    _state2, opt_state2 = optimizer.step(objective, state1, opt_state1)
    assert np.asarray(opt_state2.used_steepest_descent_reset)
    assert np.asarray(opt_state2.beta) == 0.0


def test_pncg_line_search_maximum_step_inf_norm_caps_step() -> None:
    optimizer = PNCG(
        jit=False,
        line_search=LineSearchSettings(maximum_step_inf_norm=jnp.asarray(0.25)),
    )
    objective = QuadraticObjective()
    params0 = jnp.asarray([10.0])
    opt_state0, _ = optimizer.init(objective, params0, params0)

    _state1, opt_state1 = optimizer.step(objective, params0, opt_state0)

    step_size = np.asarray(opt_state1.alpha * jnp.abs(opt_state1.search_direction))[0]
    assert step_size <= 0.25 + 1e-12
