"""Activation constraints in the historical (xx, yy, xy) coordinates."""

import numpy as np


def symmetric(controls: np.ndarray) -> np.ndarray:
    values = controls.reshape(-1, 3)
    result = np.empty((len(values), 2, 2))
    result[:, 0, 0], result[:, 1, 1] = values[:, 0], values[:, 1]
    result[:, 0, 1] = result[:, 1, 0] = values[:, 2]
    return result


def project(controls: np.ndarray, mode: str) -> np.ndarray:
    """Project S=B-I; spectral clipping uses the matrix Frobenius norm.

    This is not the Euclidean projection in packed (xx, yy, xy) coordinates:
    the Frobenius norm counts the shared off-diagonal entry twice.
    """
    if mode == "x_contraction":
        return np.maximum(controls, 0)
    if mode == "unconstrained":
        return controls.copy()
    assert mode == "contraction_only", mode
    values, vectors = np.linalg.eigh(symmetric(controls))
    projected = (vectors * np.maximum(values, 0)[:, None, :]) @ vectors.swapaxes(1, 2)
    return np.column_stack(
        (projected[:, 0, 0], projected[:, 1, 1], projected[:, 0, 1])
    ).ravel()


def gradient_mapping(
    controls: np.ndarray, gradient: np.ndarray, mode: str
) -> np.ndarray:
    """First-order diagnostic; the spectral case uses a unit Frobenius step.

    Packed dL/dq_xy includes both matrix entries, so its Frobenius gradient
    entry is half as large. This diagnostic does not determine the Adam step.
    """
    if mode == "contraction_only":
        matrix_gradient = (gradient.reshape(-1, 3) / (1, 1, 2)).ravel()
        return controls - project(controls - matrix_gradient, mode)
    assert mode in {"x_contraction", "unconstrained"}, mode
    result = gradient.copy()
    if mode == "x_contraction":
        result[(controls <= 1e-12) & (gradient > 0)] = 0
    return result


def gates() -> dict[str, float]:
    """Check a rotated indefinite tensor, feasible identity, and KKT normals."""
    angle = 0.37
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    trial = rotation @ np.diag([-2.0, 3.0]) @ rotation.T
    packed = np.array([trial[0, 0], trial[1, 1], trial[0, 1]])
    expected = rotation @ np.diag([0.0, 3.0]) @ rotation.T
    actual = symmetric(project(packed, "contraction_only"))[0]
    error = float(np.max(np.abs(actual - expected)))
    assert error < 1e-14
    B = np.eye(2) + actual
    assert np.linalg.eigvalsh(B).min() >= 1 - 1e-14
    assert np.linalg.svd(np.linalg.inv(B), compute_uv=False).max() <= 1 + 1e-14
    assert np.array_equal(project(np.zeros(3), "contraction_only"), np.zeros(3))
    # At S=0 a PSD gradient is in the constrained optimum's normal condition.
    assert np.array_equal(
        gradient_mapping(np.zeros(3), np.array([1.0, 2.0, 0.0]), "contraction_only"),
        np.zeros(3),
    )
    descent = gradient_mapping(
        np.zeros(3), np.array([-1.0, 0.0, 0.0]), "contraction_only"
    )
    assert np.allclose(descent, [-1, 0, 0])
    # A pure shear gradient has matrix eigenvalues +/- 1, not +/- 2.
    shear = gradient_mapping(np.zeros(3), np.array([0.0, 0.0, 2.0]), "contraction_only")
    assert np.allclose(shear, [-0.5, -0.5, 0.5])
    return {
        "rotated_projection_max_abs_error": error,
        "identity_and_normal_cone_checks": True,
    }
