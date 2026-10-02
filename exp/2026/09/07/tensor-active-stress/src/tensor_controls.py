"""Six orthonormal coordinates and the declared PSD stress constraint."""

from __future__ import annotations

import math

import torch

SQRT2 = math.sqrt(2.0)
EIGEN_BATCH_SIZE = 1024


def matrices(q: torch.Tensor) -> torch.Tensor:
    """Unpack xx, yy, zz, sqrt(2) xy, sqrt(2) yz, sqrt(2) xz."""
    assert q.ndim == 2
    assert q.shape[1] == 6
    x, y, z, xy, yz, xz = q.unbind(-1)
    return torch.stack(
        (
            x,
            xy / SQRT2,
            xz / SQRT2,
            xy / SQRT2,
            y,
            yz / SQRT2,
            xz / SQRT2,
            yz / SQRT2,
            z,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def coordinates(matrix: torch.Tensor) -> torch.Tensor:
    assert matrix.ndim == 3
    assert matrix.shape[1:] == (3, 3)
    return torch.stack(
        (
            matrix[:, 0, 0],
            matrix[:, 1, 1],
            matrix[:, 2, 2],
            SQRT2 * matrix[:, 0, 1],
            SQRT2 * matrix[:, 1, 2],
            SQRT2 * matrix[:, 0, 2],
        ),
        dim=-1,
    )


@torch.no_grad()
def project(q: torch.Tensor, maximum: float) -> dict[str, float]:
    """Euclidean/Frobenius projection onto 0 <= Z <= maximum I.

    Projection is part of projected Adam, outside its differentiation graph.
    It is not an LL^T parameterization and its zero state has no dead gradient.
    """
    assert maximum > 0
    squared_change = q.new_zeros(())
    negative = q.new_zeros((), dtype=torch.int64)
    upper = q.new_zeros((), dtype=torch.int64)
    for block in q.split(EIGEN_BATCH_SIZE):
        # Bound the CUDA eigensolver workspace without changing the control space.
        before = block.clone()
        values, vectors = torch.linalg.eigh(matrices(block))
        bounded = values.clamp(0.0, maximum)
        block.copy_(
            coordinates((vectors * bounded[:, None, :]) @ vectors.transpose(-1, -2))
        )
        squared_change += (block - before).square().sum()
        negative += (values < 0).sum()
        upper += (values > maximum).sum()
    return {
        "projection_rms": float((squared_change / q.numel()).sqrt()),
        "projected_negative_eigenvalue_fraction": float(negative / (3 * len(q))),
        "projected_upper_eigenvalue_fraction": float(upper / (3 * len(q))),
    }


@torch.no_grad()
def eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """Compute every cell's eigenvalues with bounded CUDA workspace."""
    return torch.cat(
        [torch.linalg.eigvalsh(block) for block in matrix.split(EIGEN_BATCH_SIZE)]
    )


def raw6_matrices(q: torch.Tensor) -> torch.Tensor:
    """The historical Raw6 direct offset map, preserving its coordinate scale."""
    scale = q.new_tensor((1, 1, 1, SQRT2, SQRT2, SQRT2))
    return torch.eye(3, device=q.device, dtype=q.dtype) + matrices(q * scale)
