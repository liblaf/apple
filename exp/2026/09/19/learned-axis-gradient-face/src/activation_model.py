# ruff: noqa: PT018
"""Three-degree-of-freedom contraction-only learned-axis activation tensors."""

from __future__ import annotations

import torch


def normalize_axis(axis: torch.Tensor) -> torch.Tensor:
    """Return a unit axis, rejecting a zero ambient direction."""
    assert axis.shape[-1] == 3
    norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    assert bool(torch.all(norm > 0))
    return axis / norm


def matrices(strength: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Return B = I + s n n^T for nonnegative strengths and unit axes."""
    assert strength.shape == axis.shape[:-1]
    assert bool(torch.all(strength >= 0))
    n = normalize_axis(axis)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + strength[..., None, None] * n[..., :, None] * n[..., None, :]


def inverse_matrices(strength: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Return A = B^-1, with eigenvalues 1, 1, and 1/(1+s)."""
    assert strength.shape == axis.shape[:-1]
    assert bool(torch.all(strength >= 0))
    n = normalize_axis(axis)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye - (strength / (1 + strength))[..., None, None] * (
        n[..., :, None] * n[..., None, :]
    )


def pack(strength: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Pack B-I as [xx, yy, zz, xy, yz, xz]."""
    assert strength.shape == axis.shape[:-1]
    assert bool(torch.all(strength >= 0))
    n = normalize_axis(axis)
    delta = strength[..., None, None] * n[..., :, None] * n[..., None, :]
    return torch.stack(
        (
            delta[..., 0, 0],
            delta[..., 1, 1],
            delta[..., 2, 2],
            delta[..., 0, 1],
            delta[..., 1, 2],
            delta[..., 0, 2],
        ),
        dim=-1,
    )


def neutral(
    count: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Return separately trainable zero strengths and canonical ambient axes."""
    assert count > 0
    strength = torch.nn.Parameter(torch.zeros(count, device=device, dtype=dtype))
    axis = torch.nn.Parameter(torch.zeros((count, 3), device=device, dtype=dtype))
    with torch.no_grad():
        axis[:, 0] = 1
    return strength, axis


@torch.no_grad()
def project_(strength: torch.Tensor, axis: torch.Tensor) -> None:
    """Project ambient parameters onto s >= 0 and the unit sphere."""
    assert strength.shape == axis.shape[:-1]
    strength.clamp_(min=0)
    axis.copy_(normalize_axis(axis))


def packed_gradient_to_symmetric(gradient: torch.Tensor) -> torch.Tensor:
    """Convert Raw6 derivatives to a symmetric Frobenius-gradient matrix."""
    assert gradient.shape[-1] == 6
    result = torch.zeros(
        (*gradient.shape[:-1], 3, 3), device=gradient.device, dtype=gradient.dtype
    )
    result[..., 0, 0], result[..., 1, 1], result[..., 2, 2] = gradient[..., :3].unbind(
        dim=-1
    )
    result[..., 0, 1] = result[..., 1, 0] = gradient[..., 3] / 2
    result[..., 1, 2] = result[..., 2, 1] = gradient[..., 4] / 2
    result[..., 0, 2] = result[..., 2, 0] = gradient[..., 5] / 2
    return result


def _canonical_sign(axis: torch.Tensor) -> torch.Tensor:
    index = axis.abs().argmax(dim=-1, keepdim=True)
    sign = torch.gather(axis, -1, index).sign()
    assert bool(torch.all(sign != 0))
    return axis * sign


def axes_from_packed_gradient(
    gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the most descending unit contraction-only axis from Raw6 data."""
    values, vectors = torch.linalg.eigh(packed_gradient_to_symmetric(gradient))
    axis = _canonical_sign(vectors[..., :, 0])
    return axis, values


def packed_smoothness(
    packed: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weighted mean edge Frobenius-squared activation disagreement."""
    assert packed.ndim == 2 and packed.shape[1] == 6
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert edges.dtype == torch.long
    assert edges.numel() > 0
    assert int(edges.min()) >= 0 and int(edges.max()) < len(packed)
    if weights is None:
        weights = torch.ones(len(edges), device=packed.device, dtype=packed.dtype)
    assert weights.shape == (len(edges),)
    assert bool(torch.all(weights > 0))
    difference = packed[edges[:, 0]] - packed[edges[:, 1]]
    energy = difference[:, :3].square().sum(dim=-1) + 2 * difference[
        :, 3:
    ].square().sum(dim=-1)
    return (weights * energy).sum() / weights.sum()


def smoothness(
    strength: torch.Tensor,
    axis: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return weighted mean edge Frobenius-squared activation disagreement."""
    return packed_smoothness(pack(strength, axis), edges, weights)
