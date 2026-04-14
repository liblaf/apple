import importlib.util
from pathlib import Path

import numpy as np

from liblaf.apple.consts import SMAS_FRACTION
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.warp import (
    WarpArap,
    WarpArapMuscle,
    WarpVolumePreservationDeterminant,
)


def test_corotational_ext_force_example_builds() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "exp/2026/01/28/smas/src/20-forward-ext-force.py"
    spec = importlib.util.spec_from_file_location("ext_force_corotational", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = module.Config(
        input=repo_root / "exp/2026/01/28/smas/data/10-input-smas46-muscle46.vtu",
        activation=2.5,
        force_scale=0.1,
        lambda_value=3.0,
    )
    mesh = module.load_mesh(cfg)
    smas_frac = np.asarray(mesh.cell_data[SMAS_FRACTION])
    fat_frac = 1.0 - smas_frac
    model = module.build_phace_v3(
        mesh, cfg.activation, cfg.force_scale, cfg.lambda_value
    )
    warp_energies = list(model.warp.energies.values())
    jax_energies = list(model.jax.energies.values())
    vol = next(
        energy
        for energy in warp_energies
        if isinstance(energy, WarpVolumePreservationDeterminant)
    )
    expected_lambda = fat_frac * cfg.lambda_value + smas_frac * cfg.lambda_value * 1.0e2

    assert cfg.activation == 2.5
    assert cfg.force_scale == 0.1
    assert cfg.lambda_value == 3.0
    assert sum(isinstance(energy, WarpArap) for energy in warp_energies) == 1
    assert sum(isinstance(energy, WarpArapMuscle) for energy in warp_energies) == 2
    assert (
        sum(
            isinstance(energy, WarpVolumePreservationDeterminant)
            for energy in warp_energies
        )
        == 1
    )
    assert sum(isinstance(energy, JaxPointForce) for energy in jax_energies) == 1
    np.testing.assert_allclose(
        np.asarray(vol.read_materials()["lambda_"]),
        expected_lambda,
    )
    assert module.format_force_scale(cfg.force_scale) == "1e-1"
    assert module.format_lambda_value(cfg.lambda_value) == "3"
