from typing import no_type_check

import jarp.warp.types as wpt
import numpy as np
import warp as wp

from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

EPS: float = 1.0e-6


@wp.struct
class StableNeoHookeanMaterials:
    fraction: wp.array1d(dtype=wpt.floating)
    lambda_: wp.array1d(dtype=wpt.floating)
    mu: wp.array1d(dtype=wpt.floating)


@wp.struct
class StableNeoHookeanMuscleMaterials:
    activation: wp.array1d(dtype=wpt.vector(6))
    fraction: wp.array1d(dtype=wpt.floating)
    lambda_: wp.array1d(dtype=wpt.floating)
    mu: wp.array1d(dtype=wpt.floating)


@wp.kernel
@no_type_check
def _energy_density_kernel(
    F: wp.array(dtype=wp.mat33d),
    materials: StableNeoHookeanMaterials,
    output: wp.array(dtype=wp.float64),
) -> None:
    cid = wp.tid()
    output[cid] = WarpStableNeoHookean.energy_density_func(F[cid], materials, cid)


@wp.kernel
@no_type_check
def _first_piola_kirchhoff_kernel(
    F: wp.array(dtype=wp.mat33d),
    materials: StableNeoHookeanMaterials,
    output: wp.array(dtype=wp.mat33d),
) -> None:
    cid = wp.tid()
    output[cid] = WarpStableNeoHookean.first_piola_kirchhoff_func(
        F[cid], materials, cid
    )


@wp.kernel
@no_type_check
def _muscle_energy_density_kernel(
    F: wp.array(dtype=wp.mat33d),
    materials: StableNeoHookeanMuscleMaterials,
    output: wp.array(dtype=wp.float64),
) -> None:
    cid = wp.tid()
    output[cid] = WarpStableNeoHookeanMuscle.energy_density_func(F[cid], materials, cid)


@wp.kernel
@no_type_check
def _muscle_first_piola_kirchhoff_kernel(
    F: wp.array(dtype=wp.mat33d),
    materials: StableNeoHookeanMuscleMaterials,
    output: wp.array(dtype=wp.mat33d),
) -> None:
    cid = wp.tid()
    output[cid] = WarpStableNeoHookeanMuscle.first_piola_kirchhoff_func(
        F[cid], materials, cid
    )


def _make_materials(batch: int) -> StableNeoHookeanMaterials:
    materials = StableNeoHookeanMaterials()
    materials.fraction = wp.from_numpy(
        np.linspace(0.25, 1.0, batch, dtype=np.float64), dtype=wp.float64
    )
    materials.lambda_ = wp.from_numpy(
        np.linspace(1.5, 7.5, batch, dtype=np.float64), dtype=wp.float64
    )
    materials.mu = wp.from_numpy(
        np.linspace(0.5, 2.5, batch, dtype=np.float64), dtype=wp.float64
    )
    return materials


def _make_muscle_materials(batch: int) -> StableNeoHookeanMuscleMaterials:
    materials = StableNeoHookeanMuscleMaterials()
    materials.activation = wp.from_numpy(
        np.array(
            [
                [0.05, -0.02, 0.03, 0.01, -0.01, 0.02],
                [-0.04, 0.06, 0.02, -0.015, 0.005, 0.01],
                [0.02, 0.01, -0.03, 0.0, 0.015, -0.02],
            ],
            dtype=np.float64,
        )[:batch],
        dtype=wpt.vector(6),
    )
    materials.fraction = wp.from_numpy(
        np.linspace(0.25, 1.0, batch, dtype=np.float64), dtype=wp.float64
    )
    materials.lambda_ = wp.from_numpy(
        np.linspace(1.5, 7.5, batch, dtype=np.float64), dtype=wp.float64
    )
    materials.mu = wp.from_numpy(
        np.linspace(0.5, 2.5, batch, dtype=np.float64), dtype=wp.float64
    )
    return materials


def _energy_density(
    F: np.ndarray, materials: StableNeoHookeanMaterials
) -> np.ndarray:
    F_wp = wp.from_numpy(F, dtype=wp.mat33d)
    output_wp = wp.zeros(F.shape[0], dtype=wp.float64)
    wp.launch(
        _energy_density_kernel,
        dim=F.shape[0],
        inputs=[F_wp, materials],
        outputs=[output_wp],
    )
    return output_wp.numpy()


def _muscle_energy_density(
    F: np.ndarray, materials: StableNeoHookeanMuscleMaterials
) -> np.ndarray:
    F_wp = wp.from_numpy(F, dtype=wp.mat33d)
    output_wp = wp.zeros(F.shape[0], dtype=wp.float64)
    wp.launch(
        _muscle_energy_density_kernel,
        dim=F.shape[0],
        inputs=[F_wp, materials],
        outputs=[output_wp],
    )
    return output_wp.numpy()


def _first_piola_kirchhoff(
    F: np.ndarray, materials: StableNeoHookeanMaterials
) -> np.ndarray:
    F_wp = wp.from_numpy(F, dtype=wp.mat33d)
    output_wp = wp.zeros(F.shape[0], dtype=wp.mat33d)
    wp.launch(
        _first_piola_kirchhoff_kernel,
        dim=F.shape[0],
        inputs=[F_wp, materials],
        outputs=[output_wp],
    )
    return output_wp.numpy()


def _muscle_first_piola_kirchhoff(
    F: np.ndarray, materials: StableNeoHookeanMuscleMaterials
) -> np.ndarray:
    F_wp = wp.from_numpy(F, dtype=wp.mat33d)
    output_wp = wp.zeros(F.shape[0], dtype=wp.mat33d)
    wp.launch(
        _muscle_first_piola_kirchhoff_kernel,
        dim=F.shape[0],
        inputs=[F_wp, materials],
        outputs=[output_wp],
    )
    return output_wp.numpy()


def _activation_matrix(activation: np.ndarray) -> np.ndarray:
    A = np.zeros((*activation.shape[:-1], 3, 3), dtype=activation.dtype)
    A[..., 0, 0] = 1.0 + activation[..., 0]
    A[..., 1, 1] = 1.0 + activation[..., 1]
    A[..., 2, 2] = 1.0 + activation[..., 2]
    A[..., 0, 1] = activation[..., 3]
    A[..., 1, 0] = activation[..., 3]
    A[..., 0, 2] = activation[..., 4]
    A[..., 2, 0] = activation[..., 4]
    A[..., 1, 2] = activation[..., 5]
    A[..., 2, 1] = activation[..., 5]
    return A


def test_stable_neo_hookean_energy_matches_finite_difference() -> None:
    F = np.array(
        [
            [[1.05, 0.02, -0.01], [0.01, 0.97, 0.03], [-0.02, 0.04, 1.02]],
            [[0.92, -0.05, 0.02], [0.03, 1.08, -0.04], [0.01, 0.02, 1.04]],
            [[1.10, 0.08, 0.00], [-0.03, 0.95, 0.05], [0.02, -0.01, 0.98]],
        ],
        dtype=np.float64,
    )
    dF = np.array(
        [
            [[0.02, -0.01, 0.00], [0.01, 0.03, -0.02], [-0.01, 0.00, 0.02]],
            [[-0.03, 0.02, 0.01], [0.00, -0.01, 0.04], [0.02, -0.02, 0.01]],
            [[0.01, 0.00, -0.02], [0.03, -0.04, 0.01], [-0.01, 0.02, 0.00]],
        ],
        dtype=np.float64,
    )
    materials = _make_materials(F.shape[0])

    assert np.all(np.linalg.det(F) > 0.0)
    assert np.all(np.linalg.det(F + 0.5 * EPS * dF) > 0.0)
    assert np.all(np.linalg.det(F - 0.5 * EPS * dF) > 0.0)

    psi_plus = _energy_density(F + 0.5 * EPS * dF, materials)
    psi_minus = _energy_density(F - 0.5 * EPS * dF, materials)
    finite_difference = (psi_plus - psi_minus) / EPS

    P = _first_piola_kirchhoff(F, materials)
    expected = np.einsum("bij,bij->b", P, dF)

    np.testing.assert_allclose(finite_difference, expected, rtol=1.0e-6, atol=1.0e-9)


def test_stable_neo_hookean_muscle_energy_matches_finite_difference() -> None:
    F = np.array(
        [
            [[1.05, 0.02, -0.01], [0.01, 0.97, 0.03], [-0.02, 0.04, 1.02]],
            [[0.92, -0.05, 0.02], [0.03, 1.08, -0.04], [0.01, 0.02, 1.04]],
            [[1.10, 0.08, 0.00], [-0.03, 0.95, 0.05], [0.02, -0.01, 0.98]],
        ],
        dtype=np.float64,
    )
    dF = np.array(
        [
            [[0.02, -0.01, 0.00], [0.01, 0.03, -0.02], [-0.01, 0.00, 0.02]],
            [[-0.03, 0.02, 0.01], [0.00, -0.01, 0.04], [0.02, -0.02, 0.01]],
            [[0.01, 0.00, -0.02], [0.03, -0.04, 0.01], [-0.01, 0.02, 0.00]],
        ],
        dtype=np.float64,
    )
    materials = _make_muscle_materials(F.shape[0])
    activation = materials.activation.numpy().view(np.float64).reshape(F.shape[0], 6)
    A = _activation_matrix(activation)

    assert np.all(np.linalg.det(F @ A) > 0.0)
    assert np.all(np.linalg.det((F + 0.5 * EPS * dF) @ A) > 0.0)
    assert np.all(np.linalg.det((F - 0.5 * EPS * dF) @ A) > 0.0)

    psi_plus = _muscle_energy_density(F + 0.5 * EPS * dF, materials)
    psi_minus = _muscle_energy_density(F - 0.5 * EPS * dF, materials)
    finite_difference = (psi_plus - psi_minus) / EPS

    P = _muscle_first_piola_kirchhoff(F, materials)
    expected = np.einsum("bij,bij->b", P, dF)

    np.testing.assert_allclose(finite_difference, expected, rtol=1.0e-6, atol=1.0e-9)
