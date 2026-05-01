import importlib.util
from pathlib import Path

import numpy as np

from liblaf.apple.consts import GLOBAL_POINT_ID
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.model import Forward, MaterialReference
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle


def test_stable_neo_hookean_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root / "exp/2026/01/28/smas/src/20-forward-ext-force-stable-neo-hookean.py"
    )
    spec = importlib.util.spec_from_file_location("ext_force_stable_nh", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.1,
        lambda_value=49.0,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(
        mesh, cfg.activation, cfg.force_scale, cfg.lambda_value
    )
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())
    passive = next(
        energy for energy in warp_energies if isinstance(energy, WarpStableNeoHookean)
    )
    muscles = [
        energy
        for energy in warp_energies
        if isinstance(energy, WarpStableNeoHookeanMuscle)
    ]

    assert cfg.activation == 2.5
    assert cfg.force_scale == 0.1
    assert cfg.lambda_value == 49.0
    assert (
        sum(isinstance(energy, WarpStableNeoHookean) for energy in warp_energies) == 1
    )
    assert (
        sum(isinstance(energy, WarpStableNeoHookeanMuscle) for energy in warp_energies)
        == 2
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    assert module.format_force_scale(cfg.force_scale) == "1e-1"
    assert module.format_lambda_value(cfg.lambda_value) == "49"
    np.testing.assert_allclose(
        np.asarray(passive.read_materials()["lambda_"]),
        cfg.lambda_value,
    )
    for energy in muscles:
        np.testing.assert_allclose(
            np.asarray(energy.read_materials()["lambda_"]),
            cfg.lambda_value * 1.0e2,
        )


def test_stable_neo_hookean_uniform_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "exp/2026/01/28/smas/src/20-forward-ext-force-stable-neo-hookean-uniform.py"
    )
    spec = importlib.util.spec_from_file_location("ext_force_stable_nh_uniform", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.05,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(mesh, cfg.activation, cfg.force_scale)
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())
    force = next(energy for energy in jax_energies if isinstance(energy, JaxPointForce))
    active = np.asarray(force.force)[:, 1]
    active = active[active > 0.0]

    assert cfg.activation == 2.5
    assert cfg.force_scale == 0.05
    assert (
        sum(isinstance(energy, WarpStableNeoHookean) for energy in warp_energies) == 1
    )
    assert (
        sum(isinstance(energy, WarpStableNeoHookeanMuscle) for energy in warp_energies)
        == 2
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    np.testing.assert_allclose(active, cfg.force_scale)
    assert module.format_force_scale(cfg.force_scale) == "5e-2"


def test_stable_neo_hookean_fat_only_top_dirichlet_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "exp/2026/01/28/smas/src/20-forward-ext-force-stable-neo-hookean-fat-only-top-dirichlet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ext_force_stable_nh_fat_only_top_dirichlet", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=(
            repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46-coarse.vtu"
        ),
        force_scale=0.1,
        lambda_value=49.0,
        mu_value=2.0,
        surface_tolerance=1.0e-2,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_pure_fat_model(
        mesh,
        cfg.force_scale,
        cfg.lambda_value,
        cfg.mu_value,
        cfg.surface_tolerance,
    )
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())
    passive = next(
        energy for energy in warp_energies if isinstance(energy, WarpStableNeoHookean)
    )
    force = next(energy for energy in jax_energies if isinstance(energy, JaxPointForce))

    assert cfg.force_scale == 0.1
    assert cfg.lambda_value == 49.0
    assert cfg.mu_value == 2.0
    assert (
        sum(isinstance(energy, WarpStableNeoHookean) for energy in warp_energies) == 1
    )
    assert (
        sum(isinstance(energy, WarpStableNeoHookeanMuscle) for energy in warp_energies)
        == 0
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    assert module.format_force_scale(cfg.force_scale) == "1e-1"
    assert module.format_material_value(cfg.lambda_value) == "49"
    assert module.format_material_value(cfg.mu_value) == "2"

    materials = passive.read_materials()
    np.testing.assert_allclose(np.asarray(materials["fraction"]), 1.0)
    np.testing.assert_allclose(np.asarray(materials["lambda_"]), cfg.lambda_value)
    np.testing.assert_allclose(np.asarray(materials["mu"]), cfg.mu_value)

    top_mask = mesh.points[:, 1] >= mesh.bounds[3] - cfg.surface_tolerance
    fixed_mask = np.asarray(model.dirichlet.fixed_mask)[
        mesh.point_data[GLOBAL_POINT_ID]
    ]
    assert np.all(fixed_mask[top_mask])
    assert not np.any(fixed_mask[~top_mask])

    bottom_mask = mesh.points[:, 1] <= mesh.bounds[2] + cfg.surface_tolerance
    force_array = np.asarray(force.force)
    np.testing.assert_allclose(force_array[:, [0, 2]], 0.0)
    np.testing.assert_allclose(force_array[~bottom_mask], 0.0)
    assert np.any(force_array[bottom_mask, 1] > 0.0)


def test_stable_neo_hookean_uniform_ramped_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "exp/2026/01/28/smas/src/20-forward-ext-force-stable-neo-hookean-uniform-ramped.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ext_force_stable_nh_uniform_ramped", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.05,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(mesh, cfg.activation, cfg.force_scale)
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())

    assert cfg.activation == 2.5
    assert cfg.force_scale == 0.05
    assert (
        sum(isinstance(energy, WarpStableNeoHookean) for energy in warp_energies) == 1
    )
    assert (
        sum(isinstance(energy, WarpStableNeoHookeanMuscle) for energy in warp_energies)
        == 2
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    assert module.format_force_scale(cfg.force_scale) == "5e-2"


def test_stable_neo_hookean_uniform_ramped_force_program_scales_force() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = (
        repo_root
        / "exp/2026/01/28/smas/src/20-forward-ext-force-stable-neo-hookean-uniform-ramped.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ext_force_stable_nh_uniform_ramped", script
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.05,
    )
    mesh = module.load_mesh(cfg)
    model = module.build_phace_v3(mesh, cfg.activation, cfg.force_scale)
    forward = Forward(model)
    plan = module.build_force_ramp_plan(forward, cfg)
    force_ref = MaterialReference("force", "force")
    final_force = forward.read_material_values()[force_ref]

    zero_force = plan.state_program.state_at(
        progress=0.0, forward=forward
    ).material_values[force_ref]
    half_force = plan.state_program.state_at(
        progress=0.5, forward=forward
    ).material_values[force_ref]
    full_force = plan.state_program.state_at(
        progress=1.0, forward=forward
    ).material_values[force_ref]

    np.testing.assert_allclose(zero_force, 0.0)
    np.testing.assert_allclose(half_force, 0.5 * np.asarray(final_force))
    np.testing.assert_allclose(full_force, final_force)
