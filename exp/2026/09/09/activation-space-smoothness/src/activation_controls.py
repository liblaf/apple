"""Shared dimensionless activation fields and their smoothness prior."""

from __future__ import annotations

import math
from typing import Literal

import torch

SQRT2 = math.sqrt(2.0)
EIGEN_BATCH_SIZE = 1024


def symmetric_matrices(q: torch.Tensor) -> torch.Tensor:
    """Unpack ``xx, yy, zz, sqrt(2) xy, sqrt(2) yz, sqrt(2) xz``."""
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


def symmetric_coordinates(matrix: torch.Tensor) -> torch.Tensor:
    """Pack symmetric matrices in Frobenius-orthonormal coordinates."""
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


def raw6_matrices(q: torch.Tensor) -> torch.Tensor:
    """Map historical Raw6 coordinates to ``B = I + sym(q)``."""
    assert q.ndim == 2
    assert q.shape[1] == 6
    x, y, z, xy, yz, xz = q.unbind(-1)
    offset = torch.stack((x, xy, xz, xy, y, yz, xz, yz, z), dim=-1).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    return identity + offset


def raw6_c(q: torch.Tensor) -> torch.Tensor:
    """Return the historical Raw6 control field ``C = B - I``."""
    b = raw6_matrices(q)
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    return b - identity


def baseline_z(q: torch.Tensor) -> torch.Tensor:
    """Return the Raw6 baseline's effective field ``Z = B B^T - I``."""
    b = raw6_matrices(q)
    identity = torch.eye(3, dtype=q.dtype, device=q.device)
    return b @ b.transpose(-1, -2) - identity


def tensile_z(q: torch.Tensor) -> torch.Tensor:
    """Return direct dimensionless tensile-stress coordinates as matrices.

    The caller maintains the PSD constraint with :func:`project_psd_` after
    each optimizer step. Keeping projection outside autograd gives the exactly
    inactive state a nonzero first-order response.
    """
    return symmetric_matrices(q)


def learned_axis_c(v: torch.Tensor) -> torch.Tensor:
    """Return the learned-axis control field ``C = v v^T``."""
    assert v.ndim == 2
    assert v.shape[1] == 3
    return v.unsqueeze(-1) * v.unsqueeze(-2)


def control_c(q: torch.Tensor, model: Literal["raw6", "learned-axis"]) -> torch.Tensor:
    """Return the model's optimization field ``C = B - I``.

    Direct tensile-stress controls have no ``C`` representation and are
    intentionally rejected instead of being assigned fallback semantics.
    """
    assert model in {"raw6", "learned-axis"}
    if model == "raw6":
        return raw6_c(q)
    return learned_axis_c(q)


def learned_axis_b(v: torch.Tensor) -> torch.Tensor:
    """Return ``B = I + v v^T`` for one learned axis per active cell."""
    outer = learned_axis_c(v)
    identity = torch.eye(3, dtype=v.dtype, device=v.device)
    return identity + outer


def learned_axis_z(v: torch.Tensor) -> torch.Tensor:
    """Return ``Z = B B^T - I = (2 + ||v||^2) v v^T``."""
    outer = learned_axis_c(v)
    squared_norm = v.square().sum(dim=-1, keepdim=True).unsqueeze(-1)
    return (2.0 + squared_norm) * outer


def learned_axis_raw6(v: torch.Tensor) -> torch.Tensor:
    """Pack ``v v^T`` in the historical unscaled Raw6 coordinates."""
    assert v.ndim == 2
    assert v.shape[1] == 3
    x, y, z = v.unbind(-1)
    return torch.stack(
        (x.square(), y.square(), z.square(), x * y, y * z, x * z), dim=-1
    )


def learned_axis_tensile(v: torch.Tensor) -> torch.Tensor:
    """Pack the learned-axis ``Z`` in Frobenius-orthonormal coordinates."""
    return symmetric_coordinates(learned_axis_z(v))


def common_initial_controls(
    labels: torch.Tensor,
    model: Literal["raw6", "tensor", "learned-axis"],
    *,
    seed: int = 20260909,
    strength: float = 0.001,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a matched rank-one ``B``/``Z`` field by muscle label.

    ``strength`` is ``s = ||v||^2`` in ``B = I + s n n^T``. Axes are
    sampled isotropically on CPU so the same seed is independent of the output
    device. Every active cell with the same integer label receives the same
    axis; no anatomical direction is used.
    """
    assert labels.ndim == 1
    assert len(labels) > 0
    assert labels.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    assert isinstance(seed, int)
    assert math.isfinite(strength)
    assert strength > 0.0
    assert model in {"raw6", "tensor", "learned-axis"}
    output_dtype = torch.get_default_dtype() if dtype is None else dtype
    assert output_dtype.is_floating_point
    output_device = labels.device if device is None else torch.device(device)

    labels_cpu = labels.detach().to(device="cpu", dtype=torch.int64)
    unique = torch.unique(labels_cpu, sorted=True)
    expected_labels = torch.arange(len(unique), dtype=torch.int64, device="cpu")
    assert torch.equal(unique, expected_labels)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axes = torch.randn(
        (len(unique), 3),
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    axes /= torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    v = math.sqrt(strength) * axes[labels_cpu]
    v = v.to(device=output_device, dtype=output_dtype)
    if model == "learned-axis":
        return v
    if model == "raw6":
        return learned_axis_raw6(v)
    return learned_axis_tensile(v)


@torch.no_grad()
def project_psd_(q: torch.Tensor) -> dict[str, float]:
    """Project direct orthonormal coordinates onto the PSD cone in place."""
    assert q.ndim == 2
    assert q.shape[1] == 6
    assert len(q) > 0
    squared_change = q.new_zeros(())
    negative = q.new_zeros((), dtype=torch.int64)
    for block in q.split(EIGEN_BATCH_SIZE):
        before = block.clone()
        values, vectors = torch.linalg.eigh(symmetric_matrices(block))
        bounded = values.clamp_min(0.0)
        block.copy_(
            symmetric_coordinates(
                (vectors * bounded[:, None, :]) @ vectors.transpose(-1, -2)
            )
        )
        squared_change += (block - before).square().sum()
        negative += (values < 0).sum()
    return {
        "projection_rms": float((squared_change / q.numel()).sqrt()),
        "projected_negative_eigenvalue_fraction": float(negative / (3 * len(q))),
    }


@torch.no_grad()
def eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """Compute symmetric eigenvalues in bounded-workspace batches."""
    assert matrix.ndim == 3
    assert matrix.shape[1:] == (3, 3)
    assert len(matrix) > 0
    return torch.cat(
        [torch.linalg.eigvalsh(block) for block in matrix.split(EIGEN_BATCH_SIZE)]
    )


def smoothness(
    z: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    edge_weight: torch.Tensor,
    smooth_length_m: float,
    active_volume: float,
) -> torch.Tensor:
    """Return the dimensionless volume-normalized FV Dirichlet energy.

    ``edge_weight`` is shared-face area divided by centroid distance, multiplied
    by the harmonic muscle fraction. The graph must contain same-muscle edges
    only. ``active_volume`` is the summed rest tetrahedron volume multiplied by
    muscle fraction.
    """
    assert z.ndim == 3
    assert z.shape[1:] == (3, 3)
    assert i.ndim == j.ndim == edge_weight.ndim == 1
    assert len(i) == len(j) == len(edge_weight)
    assert smooth_length_m > 0.0
    assert active_volume > 0.0
    delta = z[i] - z[j]
    return (
        smooth_length_m**2
        * (edge_weight * delta.square().sum(dim=(-2, -1))).sum()
        / active_volume
    )
