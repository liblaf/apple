import jax.numpy as jnp
import numpy as np
import pyvista as pv

from liblaf.apple.consts import (
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    STIFFNESS,
)
from liblaf.apple.jax.energies import JaxMassSpring, JaxPointForce
from liblaf.apple.model import (
    Forward,
    ForwardStage,
    MaterialReference,
    Model,
    ModelBuilder,
    StageState,
)
from liblaf.apple.optim import ConvergenceCriteria, PNCGOverrides


def make_model(*, include_force: bool = False) -> Model:
    mesh: pv.PolyData = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), resolution=1)  # pyright: ignore[reportAssignmentType]
    mesh.point_data[DIRICHLET_MASK] = np.array(
        [[True, True, True], [False, False, False]]
    )
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))
    mesh.cell_data[STIFFNESS] = np.ones((mesh.n_cells,))

    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    builder.add_dirichlet(mesh)
    spring = JaxMassSpring.from_pyvista(mesh)
    spring.name = "spring"
    builder.add_energy(spring)
    if include_force:
        builder.add_energy(
            JaxPointForce(
                name="load",
                force=jnp.zeros((mesh.n_points, 3)),
                indices=jnp.asarray(mesh.point_data[GLOBAL_POINT_ID]),
            )
        )
    return builder.finalize()


def test_forward_step_converges_to_zero() -> None:
    model = make_model()
    model.u_free = jnp.asarray([1e-2, 0.0, 0.0])

    forward = Forward(model)
    solution = forward.step()

    assert solution.success
    np.testing.assert_allclose(model.u_full[1, 0], 0.0, atol=1e-3)


def test_forward_solve_returns_result() -> None:
    model = make_model()
    model.u_free = jnp.asarray([1e-2, 0.0, 0.0])
    forward = Forward(model)

    result = forward.solve()

    assert result.success
    assert result.final_solver_result is not None
    assert forward.last_result is result


def test_forward_reads_and_writes_material_values_across_energies() -> None:
    model = make_model(include_force=True)
    forward = Forward(model)

    values = forward.read_material_values()

    stiffness_ref = MaterialReference("spring", "stiffness")
    force_ref = MaterialReference("load", "force")
    assert stiffness_ref in values
    assert force_ref in values

    updated_stiffness = 2.0 * jnp.ones_like(values[stiffness_ref])
    updated_force = 0.5 * jnp.ones_like(values[force_ref])
    forward.write_material_values(
        {stiffness_ref: updated_stiffness, force_ref: updated_force}
    )

    values = forward.read_material_values()
    np.testing.assert_allclose(values[stiffness_ref], updated_stiffness)
    np.testing.assert_allclose(values[force_ref], updated_force)


def test_forward_restore_last_successful_stage_restores_displacement_materials_and_dirichlet() -> (
    None
):
    model = make_model()
    forward = Forward(model)
    stiffness_ref = MaterialReference("spring", "stiffness")
    base_stiffness = forward.read_material_values()[stiffness_ref]

    zero_dirichlet = jnp.zeros_like(model.u_full)
    raised_dirichlet = zero_dirichlet.at[:, 1].set(0.1)

    class TestProgram:
        def state_at(self, *, progress: float, forward: Forward) -> StageState:
            del forward
            return StageState(
                dirichlet_values=zero_dirichlet if progress < 0.5 else raised_dirichlet,
                material_values={
                    stiffness_ref: (1.0 + progress) * jnp.ones_like(base_stiffness),
                },
            )

    state_program = TestProgram()
    forward.model.u_free = jnp.asarray([1e-2, 0.0, 0.0])
    forward.state = forward.model.init_state(forward.model.u_full)
    success_stage = ForwardStage(name="success", progress=0.0, initial_guess="current")
    success_result = forward.solve_stage(success_stage, state_program=state_program)
    assert success_result.solver_result.success

    fail_stage = ForwardStage(
        name="failure",
        progress=1.0,
        initial_guess="current",
        solver_overrides=PNCGOverrides(
            convergence=ConvergenceCriteria(
                target_relative_gradient_norm=jnp.asarray(1e-16),
                acceptable_relative_gradient_norm=jnp.asarray(1e-16),
                target_absolute_gradient_norm=jnp.asarray(0.0),
                acceptable_absolute_gradient_norm=jnp.asarray(0.0),
                maximum_iterations=jnp.asarray(0, dtype=jnp.int32),
            )
        ),
    )
    fail_result = forward.solve_stage(fail_stage, state_program=state_program)

    assert not fail_result.solver_result.success
    np.testing.assert_allclose(forward.read_material_values()[stiffness_ref], 2.0)
    np.testing.assert_allclose(
        forward.model.dirichlet.dirichlet_value,
        forward.model.dirichlet.get_fixed(raised_dirichlet),
    )

    forward.restore_last_successful_stage()

    np.testing.assert_allclose(forward.read_material_values()[stiffness_ref], 1.0)
    np.testing.assert_allclose(
        forward.model.dirichlet.dirichlet_value,
        forward.model.dirichlet.get_fixed(zero_dirichlet),
    )
