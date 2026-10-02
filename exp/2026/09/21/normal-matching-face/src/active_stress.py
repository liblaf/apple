"""Activation-induced stress diagnostic for the frozen Raw6 material.

The frozen material has W(F, B) = mu/2 (||F B||^2 - 3) + g(det F), with B
constructed directly from the six Raw6 values.  The diagnostic below subtracts
the passive response at the *same* physical F.  It therefore isolates only the
activation-dependent term and deliberately does not claim a full tissue stress.
"""

from __future__ import annotations

import numpy as np


def raw6_matrix(q: np.ndarray) -> np.ndarray:
    """Return the symmetric B = I + sym(q) used by make_activation_mat33."""
    q = np.asarray(q, dtype=np.float64)
    assert q.ndim >= 1
    assert q.shape[-1] == 6
    B = np.broadcast_to(np.eye(3), (*q.shape[:-1], 3, 3)).copy()
    B[..., 0, 0] += q[..., 0]
    B[..., 1, 1] += q[..., 1]
    B[..., 2, 2] += q[..., 2]
    B[..., 0, 1] = B[..., 1, 0] = q[..., 3]
    B[..., 1, 2] = B[..., 2, 1] = q[..., 4]
    B[..., 0, 2] = B[..., 2, 0] = q[..., 5]
    return B


def deformation_gradient(
    rest_points: np.ndarray, deformed_points: np.ndarray, tets: np.ndarray
) -> np.ndarray:
    """Linear-tetrahedron F with the same column-edge convention as FacePhysics."""
    rest = np.asarray(rest_points, dtype=np.float64)
    current = np.asarray(deformed_points, dtype=np.float64)
    cells = np.asarray(tets, dtype=np.int64)
    assert rest.shape == current.shape
    assert rest.shape[1] == 3
    assert cells.ndim == 2
    assert cells.shape[1] == 4
    Dm = np.swapaxes(rest[cells[:, 1:]] - rest[cells[:, :1]], 1, 2)
    Ds = np.swapaxes(current[cells[:, 1:]] - current[cells[:, :1]], 1, 2)
    return Ds @ np.linalg.inv(Dm)


def effective_reference_stress(
    B: np.ndarray, muscle_fraction: np.ndarray, mu_mpa: float
) -> np.ndarray:
    """Q = fraction * mu * (B B^T - I), in MPa, for the mixed-cell energy."""
    B = np.asarray(B, dtype=np.float64)
    fraction = np.asarray(muscle_fraction, dtype=np.float64)
    assert B.shape[-2:] == (3, 3)
    assert fraction.shape == B.shape[:-2]
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(fraction))
    assert np.all((fraction >= 0.0) & (fraction <= 1.0))
    assert mu_mpa > 0.0
    return fraction[..., None, None] * mu_mpa * (B @ np.swapaxes(B, -1, -2) - np.eye(3))


def activation_cauchy_difference(
    F: np.ndarray, Q_mpa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Δσ = F Q Fᵀ / J and its orientation-preserving validity mask.

    Invalid (J <= 0) cells are returned as NaN.  No absolute determinant or
    clipping is used, because neither is part of the frozen constitutive law.
    """
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q_mpa, dtype=np.float64)
    assert F.shape == Q.shape
    assert F.shape[-2:] == (3, 3)
    J = np.linalg.det(F)
    valid = np.isfinite(J) & (J > 0.0)
    sigma = np.full_like(F, np.nan)
    sigma[valid] = (
        F[valid] @ Q[valid] @ np.swapaxes(F[valid], -1, -2) / J[valid, None, None]
    )
    return sigma, valid


def active_energy_density(
    F: np.ndarray,
    B: np.ndarray,
    mu_mpa: np.ndarray | float,
    lambda_mpa: np.ndarray | float,
) -> np.ndarray:
    """Exact frozen Raw6 energy density before quadrature/fraction weighting."""
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    G = F @ B
    J = np.linalg.det(F)
    return (
        0.5 * np.asarray(mu_mpa) * (np.sum(G * G, axis=(-2, -1)) - 3.0)
        - np.asarray(mu_mpa) * (J - 1.0)
        + 0.5 * np.asarray(lambda_mpa) * (J - 1.0) ** 2
    )


def active_first_piola(
    F: np.ndarray,
    B: np.ndarray,
    mu_mpa: np.ndarray | float,
    lambda_mpa: np.ndarray | float,
) -> np.ndarray:
    """Exact frozen first Piola stress, using cof(F) for d det(F)/dF."""
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    J = np.linalg.det(F)
    cofactor = J[..., None, None] * np.swapaxes(np.linalg.inv(F), -1, -2)
    return (
        np.asarray(mu_mpa)[..., None, None] * F @ B @ np.swapaxes(B, -1, -2)
        + (-np.asarray(mu_mpa) + np.asarray(lambda_mpa) * (J - 1.0))[..., None, None]
        * cofactor
    )
