"""Mandel-isometric activation-stress parameterizations for staged fitting.

Mandel order is ``(xx, yy, zz, sqrt(2) xy, sqrt(2) yz, sqrt(2) xz)``.
Consequently Euclidean control norms equal symmetric-matrix Frobenius norms.
``rankone_learned`` uses ``(a, vx, vy, vz)`` and requires :func:`project_`
after optimizer steps. Its direction gradient is zero at ``a == 0``. Use
:func:`initialize_learned_zero_amplitude_axes` before stage 4 to select a
descent direction without injecting unphysical positive activation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

Mode = Literal["symmetric6", "psd6", "rankone_fixed", "rankone_learned"]

MANDEL_ORDER = ("xx", "yy", "zz", "sqrt2_xy", "sqrt2_yz", "sqrt2_xz")
SQRT2 = float(np.sqrt(2.0))
STRESS_REF_MPA = 0.012 / (2.0 * 1.49)
EIGH_CHUNK_SIZE = 4096
_MODES = frozenset(("symmetric6", "psd6", "rankone_fixed", "rankone_learned"))


def _mode(mode: str) -> None:
    assert mode in _MODES, mode


def _normalize_axis(axis: torch.Tensor) -> torch.Tensor:
    """Normalize nonzero axes while preserving their coordinate sign."""
    assert axis.shape[-1] == 3
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    assert bool(torch.all(norm > 0)), "learned rank-one axes must be nonzero"
    return axis / norm


def _canonical_axis(axis: torch.Tensor) -> torch.Tensor:
    """Normalize an initialization axis and choose a deterministic tensor sign."""
    unit = _normalize_axis(axis)
    index = unit.abs().argmax(dim=-1, keepdim=True)
    sign = torch.gather(unit, -1, index).sign()
    return unit * torch.where(sign == 0, torch.ones_like(sign), sign)


def mandel_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert Mandel6 vectors to symmetric tensors."""
    assert q.shape[-1] == 6
    matrix = torch.zeros((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2] = q.unbind(-1)[:3]
    matrix[..., 0, 1] = matrix[..., 1, 0] = q[..., 3] / SQRT2
    matrix[..., 1, 2] = matrix[..., 2, 1] = q[..., 4] / SQRT2
    matrix[..., 0, 2] = matrix[..., 2, 0] = q[..., 5] / SQRT2
    return matrix


def matrix_to_mandel(matrix: torch.Tensor) -> torch.Tensor:
    """Convert symmetric tensors to the explicit Mandel6 order."""
    assert matrix.shape[-2:] == (3, 3)
    assert torch.allclose(matrix, matrix.mT, rtol=1e-10, atol=1e-12)
    return torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 1, 1],
            matrix[..., 2, 2],
            SQRT2 * matrix[..., 0, 1],
            SQRT2 * matrix[..., 1, 2],
            SQRT2 * matrix[..., 0, 2],
        ),
        dim=-1,
    )


def mandel_frobenius2(q: torch.Tensor) -> torch.Tensor:
    """Return ||Q||_F^2 from a Mandel6 control vector.

    Use the ordinary Euclidean norm of a Mandel6 gradient for physical-gradient
    reporting: it already has the required factor two for off-diagonals.
    """
    assert q.shape[-1] == 6
    return q.square().sum(dim=-1)


@torch.no_grad()
def _eigh_bounded(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run no-grad 3x3 spectral decompositions in fixed-size CUDA workspaces.

    Full face fields contain 288,235 tensors.  Chunking bounds cuSOLVER workspace
    to ``EIGH_CHUNK_SIZE`` tensors and deliberately never switches algorithms.
    """
    assert matrix.shape[-2:] == (3, 3)
    assert not matrix.requires_grad
    flat = matrix.reshape(-1, 3, 3)
    values = torch.empty((len(flat), 3), dtype=matrix.dtype, device=matrix.device)
    vectors = torch.empty_like(flat)
    for start in range(0, len(flat), EIGH_CHUNK_SIZE):
        stop = min(start + EIGH_CHUNK_SIZE, len(flat))
        value, vector = torch.linalg.eigh(flat[start:stop])
        values[start:stop], vectors[start:stop] = value, vector
    return values.reshape((*matrix.shape[:-2], 3)), vectors.reshape(matrix.shape)


def _principal_positive(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return largest nonnegative eigenvalue and its deterministic axis."""
    values, vectors = _eigh_bounded((matrix + matrix.mT) / 2)
    amplitude = values[..., -1].clamp_min(0)
    axis = _canonical_axis(vectors[..., :, -1])
    zero = torch.linalg.matrix_norm(matrix, dim=(-2, -1)) == 0
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1
    axis = torch.where(zero[..., None], fallback, axis)
    return amplitude, axis


def zero_amplitude_count(qhat: torch.Tensor) -> int:
    """Count zero input tensors requiring deterministic rank-one axis fallback."""
    assert qhat.shape[-2:] == (3, 3)
    return int((torch.linalg.matrix_norm(qhat, dim=(-2, -1)) == 0).sum().item())


def controls_from_matrix(
    qhat: torch.Tensor, mode: Mode
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Initialize controls from a dimensionless symmetric activation tensor."""
    _mode(mode)
    assert qhat.shape[-2:] == (3, 3)
    qhat = (qhat + qhat.mT) / 2
    if mode == "symmetric6":
        return matrix_to_mandel(qhat), None
    if mode == "psd6":
        values, vectors = _eigh_bounded(qhat)
        projected = (vectors * values.clamp_min(0).unsqueeze(-2)) @ vectors.mT
        return matrix_to_mandel(projected), None
    amplitude, axis = _principal_positive(qhat)
    if mode == "rankone_fixed":
        return amplitude.unsqueeze(-1), axis
    return torch.cat((amplitude.unsqueeze(-1), axis), dim=-1), None


def learned_controls_from_fixed(
    q_fixed: torch.Tensor, fixed_axes: torch.Tensor
) -> torch.Tensor:
    """Convert rank-one fixed-axis controls to learned-axis controls.

    This conversion retains every parent axis, including axes at exactly zero
    amplitude that cannot be reconstructed from its zero physical tensor.
    ``q_fixed`` is expected to have already been projected, so negative
    amplitudes are rejected rather than silently changed.
    """
    assert q_fixed.shape[-1] == 1
    assert fixed_axes.shape == (*q_fixed.shape[:-1], 3)
    assert bool(torch.all(q_fixed[..., 0] >= 0))
    return torch.cat((q_fixed, _normalize_axis(fixed_axes)), dim=-1)


def _gradient_matrix(gradient: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    """Validate a physical symmetric stress gradient and return it as matrices."""
    batch_shape = controls.shape[:-1]
    if gradient.shape == (*batch_shape, 6):
        return mandel_to_matrix(gradient)
    assert gradient.shape == (*batch_shape, 3, 3)
    assert torch.allclose(gradient, gradient.mT, rtol=1e-10, atol=1e-12)
    return gradient


@torch.no_grad()
def initialize_learned_zero_amplitude_axes(
    controls: torch.Tensor, stress_gradient: torch.Tensor
) -> torch.Tensor:
    """Select descent axes for exactly-zero learned rank-one amplitudes.

    ``stress_gradient`` is the total objective derivative with respect to the
    dimensionless symmetric activation tensor, supplied as a symmetric matrix
    or a Mandel6 vector. For each exactly-zero amplitude, its axis is replaced
    only if the minimum eigenvalue is strictly negative; that makes the
    one-sided amplitude derivative negative. Positive-amplitude axes and
    zero-amplitude axes with a PSD gradient are returned unchanged.

    The input controls must already be feasible (nonnegative amplitudes and
    nonzero axes). Exact zero is intentional: near-zero amplitudes are left to
    the optimizer. For degenerate minimum eigenvalues, any canonicalized basis
    vector returned by ``torch.linalg.eigh`` is an equally valid descent axis.
    """
    assert controls.shape[-1] == 4
    assert bool(torch.all(controls[..., 0] >= 0))
    _normalize_axis(controls[..., 1:])
    gradient = _gradient_matrix(stress_gradient, controls)
    values, vectors = _eigh_bounded((gradient + gradient.mT) / 2)
    axis = _canonical_axis(vectors[..., :, 0])
    replace = (controls[..., 0] == 0) & (values[..., 0] < 0)
    initialized = controls.clone()
    initialized[..., 1:] = torch.where(replace[..., None], axis, controls[..., 1:])
    return initialized


def matrices(
    q: torch.Tensor, mode: Mode, fixed_axes: torch.Tensor | None = None
) -> torch.Tensor:
    """Return dimensionless symmetric activation tensors from controls."""
    _mode(mode)
    if mode in {"symmetric6", "psd6"}:
        assert fixed_axes is None
        assert q.shape[-1] == 6
        return mandel_to_matrix(q)
    if mode == "rankone_fixed":
        assert q.shape[-1] == 1
        assert fixed_axes is not None
        axis = _normalize_axis(fixed_axes)
        amplitude = q[..., 0].clamp_min(0)
    else:
        assert fixed_axes is None
        assert q.shape[-1] == 4
        amplitude, axis = q[..., 0].clamp_min(0), _normalize_axis(q[..., 1:])
    return amplitude[..., None, None] * axis[..., :, None] * axis[..., None, :]


@torch.no_grad()
def project_(q: torch.Tensor, mode: Mode) -> torch.Tensor:
    """Project in-place after an optimizer step and return ``q``."""
    _mode(mode)
    if mode == "symmetric6":
        assert q.shape[-1] == 6
        return q
    if mode == "psd6":
        assert q.shape[-1] == 6
        values, vectors = _eigh_bounded(mandel_to_matrix(q))
        projected = (vectors * values.clamp_min(0).unsqueeze(-2)) @ vectors.mT
        q.copy_(matrix_to_mandel(projected))
        return q
    if mode == "rankone_fixed":
        assert q.shape[-1] == 1
        q.clamp_(min=0)
        return q
    assert q.shape[-1] == 4
    q[..., :1].clamp_(min=0)
    q[..., 1:].copy_(_normalize_axis(q[..., 1:]))
    return q


def project_stress_numpy(stress_mpa: np.ndarray) -> np.ndarray:
    """PSD-project symmetric MPa tensors, preserving NumPy batch shape."""
    stress = np.asarray(stress_mpa, dtype=np.float64)
    assert stress.shape[-2:] == (3, 3)
    values, vectors = np.linalg.eigh((stress + np.swapaxes(stress, -1, -2)) / 2)
    return (vectors * np.maximum(values, 0)[..., None, :]) @ np.swapaxes(
        vectors, -1, -2
    )


def initialize_from_stress_numpy(
    stress_mpa: np.ndarray, mode: Mode, stress_ref_mpa: float = STRESS_REF_MPA
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert MPa tensors to projected staged controls using CPU Torch math."""
    assert np.isfinite(stress_ref_mpa)
    assert stress_ref_mpa > 0
    matrix = torch.from_numpy(np.asarray(stress_mpa, dtype=np.float64) / stress_ref_mpa)
    q, axes = controls_from_matrix(matrix, mode)
    return q.numpy(), None if axes is None else axes.numpy()
