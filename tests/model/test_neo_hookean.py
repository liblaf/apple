import importlib.util
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from liblaf.peach.optim import PNCG

from liblaf.apple.consts import ACTIVATION, DIRICHLET_MASK, DIRICHLET_VALUE, LAMBDA, MU
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import WarpNeoHookean, WarpNeoHookeanMuscle


def make_tetra_mesh() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()  # pyright: ignore[reportAssignmentType]
    bottom_mask = mesh.points[:, 1] < mesh.bounds.y_min + 1e-3 * (
        mesh.bounds.y_max - mesh.bounds.y_min
    )
    mesh.point_data[DIRICHLET_MASK] = np.broadcast_to(
        bottom_mask[:, np.newaxis], (mesh.n_points, 3)
    ).copy()
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))
    mesh.cell_data["Fraction"] = np.ones((mesh.n_cells,))
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    return mesh


def make_model(
    mesh: pv.UnstructuredGrid,
    make_energy: Callable[[pv.UnstructuredGrid], WarpNeoHookean | WarpNeoHookeanMuscle],
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    builder.add_dirichlet(mesh)
    energy = make_energy(mesh)
    builder.add_energy(energy)
    return builder.finalize()


def small_u_free(model: Model, mesh: pv.UnstructuredGrid):
    scale = 1e-3 * mesh.length
    return jnp.linspace(-scale, scale, model.n_free)


def make_state(model: Model, u_free):
    u_full = model.dirichlet.to_full(u_free)
    state = model.init_state(u_full)
    return model.update(state, u_full)


def test_neo_hookean_passive_smoke() -> None:
    mesh = make_tetra_mesh()
    model = make_model(
        mesh, lambda mesh: WarpNeoHookean.from_pyvista(mesh, name="elastic")
    )
    u_free = small_u_free(model, mesh)
    state = make_state(model, u_free)

    value = model.fun(state)
    grad = model.grad(state)
    hess_diag = model.hess_diag(state)

    assert np.isfinite(np.asarray(value)).all()
    assert np.isfinite(np.asarray(grad)).all()
    assert np.isfinite(np.asarray(hess_diag)).all()

    model.u_free = u_free
    forward = Forward(
        model,
        optimizer=PNCG(
            max_steps=jnp.asarray(200),
            rtol=jnp.asarray(1e-8),
            rtol_primary=jnp.asarray(1e-10),
        ),
    )
    solution = forward.step()
    assert solution.success
    np.testing.assert_allclose(model.u_full, 0.0, atol=1e-3 * mesh.length)


def test_neo_hookean_muscle_matches_passive_for_zero_activation() -> None:
    mesh_passive = make_tetra_mesh()
    mesh_muscle = make_tetra_mesh()
    passive = make_model(
        mesh_passive,
        lambda mesh: WarpNeoHookean.from_pyvista(mesh, name="elastic"),
    )
    muscle = make_model(
        mesh_muscle,
        lambda mesh: WarpNeoHookeanMuscle.from_pyvista(mesh, name="elastic"),
    )

    u_free = small_u_free(passive, mesh_passive)
    passive_state = make_state(passive, u_free)
    muscle_state = make_state(muscle, u_free)
    p = passive.dirichlet.to_full(u_free[::-1])

    np.testing.assert_allclose(
        np.asarray(passive.fun(passive_state)),
        np.asarray(muscle.fun(muscle_state)),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(passive.grad(passive_state)),
        np.asarray(muscle.grad(muscle_state)),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(passive.hess_diag(passive_state)),
        np.asarray(muscle.hess_diag(muscle_state)),
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(passive.hess_quad(passive_state, p)),
        np.asarray(muscle.hess_quad(muscle_state, p)),
        rtol=1e-6,
        atol=1e-8,
    )


def test_neo_hookean_muscle_activation_mixed_derivative_is_finite() -> None:
    mesh = make_tetra_mesh()
    mesh.cell_data[ACTIVATION][:] = np.asarray([[0.05, -0.02, 0.03, 0.01, -0.01, 0.02]])
    model = make_model(
        mesh,
        lambda mesh: WarpNeoHookeanMuscle.from_pyvista(
            mesh, name="elastic", requires_grad=("activation",)
        ),
    )
    u_free = small_u_free(model, mesh)
    state = make_state(model, u_free)
    p = model.dirichlet.to_full(jnp.linspace(1e-4, 2e-4, model.n_free))

    grads = model.mixed_derivative_prod(state, p)
    activation_grad = np.asarray(grads["elastic"]["activation"])

    assert activation_grad.shape == (mesh.n_cells, 6)
    assert np.isfinite(activation_grad).all()
    assert not np.allclose(activation_grad, 0.0)


def test_neo_hookean_bottom_arch_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "exp/2026/01/28/smas/src/20-forward-bottom-dirichlet-neo-hookean.py"
    )
    spec = importlib.util.spec_from_file_location("bottom_dirichlet_nh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        arch_height=0.25,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(mesh, cfg.arch_height)
    energies = list(model.warp.energies.values())

    assert cfg.arch_height == 0.25
    assert sum(isinstance(energy, WarpNeoHookean) for energy in energies) == 1
    assert sum(isinstance(energy, WarpNeoHookeanMuscle) for energy in energies) == 2


def test_neo_hookean_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "exp/2026/01/28/smas/src/20-forward-ext-force-neo-hookean.py"
    spec = importlib.util.spec_from_file_location("ext_force_nh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.1,
        lambda_value=9.0,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(
        mesh, cfg.activation, cfg.force_scale, cfg.lambda_value
    )
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())
    passive = next(
        energy for energy in warp_energies if isinstance(energy, WarpNeoHookean)
    )
    muscles = [
        energy for energy in warp_energies if isinstance(energy, WarpNeoHookeanMuscle)
    ]

    assert cfg.activation == 2.5
    assert cfg.force_scale == 0.1
    assert cfg.lambda_value == 9.0
    assert sum(isinstance(energy, WarpNeoHookean) for energy in warp_energies) == 1
    assert (
        sum(isinstance(energy, WarpNeoHookeanMuscle) for energy in warp_energies) == 2
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    assert module.format_force_scale(cfg.force_scale) == "1e-1"
    assert module.format_lambda_value(cfg.lambda_value) == "9"
    np.testing.assert_allclose(
        np.asarray(passive.read_materials()["lambda_"]),
        cfg.lambda_value,
    )
    for energy in muscles:
        np.testing.assert_allclose(
            np.asarray(energy.read_materials()["lambda_"]),
            cfg.lambda_value * 1.0e2,
        )
