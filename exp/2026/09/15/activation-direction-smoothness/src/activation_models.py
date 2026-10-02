"""Two-dimensional muscle activation parameterizations and smoothness.

The matrix returned here is the active-strain factor ``B``.  In particular,
``learned_direction`` uses explicit strength and angle coordinates,
``B = I + s n(theta) n(theta)^T``.  This keeps the strength derivative
nonzero at ``s = 0``, unlike a ``v v^T`` parameterization initialized at zero.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

MODES = {
    "unconstrained",
    "contraction_only",
    "learned_direction",
    "x_contraction",
}


def dofs(mode: str) -> int:
    """Return the number of scalar controls per active triangle."""
    assert mode in MODES, mode
    if mode in {"unconstrained", "contraction_only"}:
        return 3
    if mode == "learned_direction":
        return 2
    return 1


def initialize(n: int, mode: str) -> np.ndarray:
    """Return the flat inactive control vector for ``n`` active triangles."""
    assert isinstance(n, (int, np.integer))
    assert n >= 0
    return np.zeros(n * dofs(mode), dtype=float)


def _controls(q: np.ndarray, mode: str) -> np.ndarray:
    q = np.asarray(q)
    assert q.ndim == 1, q.shape
    assert np.issubdtype(q.dtype, np.floating), q.dtype
    width = dofs(mode)
    assert q.size % width == 0, (q.size, width)
    assert np.all(np.isfinite(q))
    return q.reshape(-1, width)


def _symmetric(q: np.ndarray) -> np.ndarray:
    values = q.reshape(-1, 3)
    result = np.empty((len(values), 2, 2), dtype=q.dtype)
    result[:, 0, 0] = values[:, 0]
    result[:, 1, 1] = values[:, 1]
    result[:, 0, 1] = values[:, 2]
    result[:, 1, 0] = values[:, 2]
    return result


def matrices(q: np.ndarray, mode: str) -> np.ndarray:
    """Map flat controls to one muscle ``B`` matrix per active triangle."""
    values = _controls(q, mode)
    identity = np.eye(2, dtype=values.dtype)
    result = np.broadcast_to(identity, (len(values), 2, 2)).copy()
    if mode in {"unconstrained", "contraction_only"}:
        offset = _symmetric(values.ravel())
        if mode == "contraction_only":
            eigenvalues = np.linalg.eigvalsh(offset)
            assert np.all(eigenvalues >= -1e-12), eigenvalues.min()
        result += offset
    elif mode == "learned_direction":
        strength = values[:, 0]
        assert np.all(strength >= 0.0), strength.min(initial=0.0)
        angle = values[:, 1]
        axis = np.column_stack((np.cos(angle), np.sin(angle)))
        result += strength[:, None, None] * axis[:, :, None] * axis[:, None, :]
    else:
        assert mode == "x_contraction", mode
        strength = values[:, 0]
        assert np.all(strength >= 0.0), strength.min(initial=0.0)
        result[:, 0, 0] += strength
    return result


def project(q: np.ndarray, mode: str) -> np.ndarray:
    """Project controls onto the mode's feasible set."""
    values = _controls(q, mode)
    if mode == "unconstrained":
        return q.copy()
    if mode in {"learned_direction", "x_contraction"}:
        result = values.copy()
        result[:, 0] = np.maximum(result[:, 0], 0.0)
        return result.ravel()
    assert mode == "contraction_only", mode
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(values.ravel()))
    bounded = np.maximum(eigenvalues, 0.0)
    projected = (eigenvectors * bounded[:, None, :]) @ eigenvectors.swapaxes(1, 2)
    return np.column_stack(
        (projected[:, 0, 0], projected[:, 1, 1], projected[:, 0, 1])
    ).ravel()


def pullback(q: np.ndarray, mode: str, gradient_B: np.ndarray) -> np.ndarray:
    """Apply the exact chain rule from ``dL/dB`` to flat controls."""
    values = _controls(q, mode)
    gradient_B = np.asarray(gradient_B)
    assert gradient_B.shape == (len(values), 2, 2), gradient_B.shape
    assert np.issubdtype(gradient_B.dtype, np.floating), gradient_B.dtype
    assert np.all(np.isfinite(gradient_B))
    if mode in {"unconstrained", "contraction_only"}:
        return np.column_stack(
            (
                gradient_B[:, 0, 0],
                gradient_B[:, 1, 1],
                gradient_B[:, 0, 1] + gradient_B[:, 1, 0],
            )
        ).ravel()
    if mode == "x_contraction":
        return gradient_B[:, 0, 0].copy()
    assert mode == "learned_direction", mode
    strength, angle = values.T
    axis = np.column_stack((np.cos(angle), np.sin(angle)))
    tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
    gradient_strength = np.einsum("ei,eij,ej->e", axis, gradient_B, axis)
    derivative_angle = strength[:, None, None] * (
        tangent[:, :, None] * axis[:, None, :] + axis[:, :, None] * tangent[:, None, :]
    )
    gradient_angle = np.sum(gradient_B * derivative_angle, axis=(1, 2))
    return np.column_stack((gradient_strength, gradient_angle)).ravel()


def smoothness(B: np.ndarray, edges: np.ndarray) -> tuple[float, np.ndarray]:
    """Return mean edgewise squared Frobenius jumps and ``dR/dB``."""
    B = np.asarray(B)
    edges = np.asarray(edges)
    assert B.ndim == 3, B.shape
    assert B.shape[1:] == (2, 2), B.shape
    assert np.issubdtype(B.dtype, np.floating), B.dtype
    assert np.all(np.isfinite(B))
    assert edges.ndim == 2, edges.shape
    assert edges.shape[1] == 2, edges.shape
    assert np.issubdtype(edges.dtype, np.integer), edges.dtype
    assert len(edges) > 0
    assert np.all(edges >= 0)
    assert np.all(edges < len(B))
    assert np.all(edges[:, 0] != edges[:, 1])
    delta = B[edges[:, 0]] - B[edges[:, 1]]
    value = float(np.mean(np.sum(delta * delta, axis=(1, 2))))
    gradient = np.zeros_like(B)
    scaled = (2.0 / len(edges)) * delta
    np.add.at(gradient, edges[:, 0], scaled)
    np.add.at(gradient, edges[:, 1], -scaled)
    return value, gradient


def gradient_mapping(q: np.ndarray, g: np.ndarray, mode: str) -> np.ndarray:
    """Return the constraint-aware first-order diagnostic.

    The PSD mode uses the matrix Frobenius metric.  Its packed off-diagonal
    gradient is therefore divided by two before the unit projected step.
    """
    values = _controls(q, mode)
    gradient = np.asarray(g)
    assert gradient.shape == q.shape, (gradient.shape, q.shape)
    assert np.issubdtype(gradient.dtype, np.floating), gradient.dtype
    assert np.all(np.isfinite(gradient))
    if mode == "contraction_only":
        matrix_gradient = (gradient.reshape(-1, 3) / (1.0, 1.0, 2.0)).ravel()
        return q - project(q - matrix_gradient, mode)
    result = gradient.copy().reshape(values.shape)
    if mode in {"learned_direction", "x_contraction"}:
        result[(values[:, 0] <= 1e-12) & (result[:, 0] > 0.0), 0] = 0.0
    return result.ravel()


def _finite_difference_gradient(
    function: Callable[[np.ndarray], float], x: np.ndarray, step: float
) -> np.ndarray:
    result = np.empty_like(x)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction.flat[index] = step
        result.flat[index] = (function(x + direction) - function(x - direction)) / (
            2.0 * step
        )
    return result


def gates() -> dict[str, object]:
    """Run independent derivative and feasibility checks for all modes."""
    rng = np.random.default_rng(20260915)
    step = 1e-6
    controls = {
        "unconstrained": np.array([0.13, -0.07, 0.04, -0.02, 0.11, -0.05]),
        "contraction_only": np.array([0.16, 0.09, 0.03, 0.08, 0.14, -0.02]),
        "learned_direction": np.array([0.17, 0.31, 0.09, -0.72]),
        "x_contraction": np.array([0.12, 0.05]),
    }
    pullback_errors: dict[str, float] = {}
    for mode, q in controls.items():
        gradient_B = rng.normal(size=(2, 2, 2))
        analytic = pullback(q, mode, gradient_B)
        numeric = _finite_difference_gradient(
            lambda trial, mode=mode, gradient_B=gradient_B: float(
                np.sum(matrices(trial, mode) * gradient_B)
            ),
            q,
            step,
        )
        error = float(np.max(np.abs(analytic - numeric)))
        assert error < 1e-9, (mode, error, analytic, numeric)
        pullback_errors[mode] = error

    B = rng.normal(size=(4, 2, 2))
    edges = np.array([[0, 1], [1, 2], [1, 3], [2, 3]], dtype=int)
    value, analytic_B = smoothness(B, edges)
    numeric_B = _finite_difference_gradient(
        lambda trial: smoothness(trial.reshape(B.shape), edges)[0], B.ravel(), step
    ).reshape(B.shape)
    smoothness_error = float(np.max(np.abs(analytic_B - numeric_B)))
    assert smoothness_error < 2e-9, smoothness_error
    assert value >= 0.0

    indefinite = np.array([[0.2, -0.5, 0.3]])
    psd_q = project(indefinite.ravel(), "contraction_only")
    psd_offset = matrices(psd_q, "contraction_only") - np.eye(2)
    psd_min_eigenvalue = float(np.linalg.eigvalsh(psd_offset).min())
    assert psd_min_eigenvalue >= -1e-14

    learned_q = project(np.array([-0.4, 0.37, 0.2, -1.1]), "learned_direction")
    learned_offset = matrices(learned_q, "learned_direction") - np.eye(2)
    learned_second_singular = float(
        np.linalg.svd(learned_offset, compute_uv=False)[:, 1].max()
    )
    assert learned_second_singular < 1e-15
    assert np.all(learned_q.reshape(-1, 2)[:, 0] >= 0.0)
    inactive_pullback = pullback(
        initialize(1, "learned_direction"),
        "learned_direction",
        np.array([[[1.0, 0.0], [0.0, 0.0]]]),
    )
    assert np.array_equal(inactive_pullback, np.array([1.0, 0.0]))

    x_q = project(np.array([-0.2, 0.3]), "x_contraction")
    x_offset = matrices(x_q, "x_contraction") - np.eye(2)
    x_forbidden_max_abs = float(np.max(np.abs(x_offset[:, [0, 1, 1], [1, 0, 1]])))
    assert x_forbidden_max_abs == 0.0
    assert np.all(x_q >= 0.0)

    signed_q = np.array([0.23, 0.41, 0.08, -0.63])
    flipped_q = signed_q.copy().reshape(-1, 2)
    flipped_q[:, 1] += np.pi
    sign_invariance_error = float(
        np.max(
            np.abs(
                matrices(signed_q, "learned_direction")
                - matrices(flipped_q.ravel(), "learned_direction")
            )
        )
    )
    assert sign_invariance_error < 2e-16, sign_invariance_error

    return {
        "pullback_max_abs_errors": pullback_errors,
        "smoothness_derivative_max_abs_error": smoothness_error,
        "projected_psd_min_eigenvalue": psd_min_eigenvalue,
        "learned_rank_one_second_singular_max": learned_second_singular,
        "inactive_learned_strength_gradient": float(inactive_pullback[0]),
        "x_forbidden_entry_max_abs": x_forbidden_max_abs,
        "learned_axis_sign_invariance_max_abs_error": sign_invariance_error,
    }


if __name__ == "__main__":
    print(json.dumps(gates(), indent=2, sort_keys=True))
