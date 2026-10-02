# ruff: noqa: EM101, TRY003
"""Candidate latent-axis activation coordinates for a later screen.

Each control is a vector ``v`` with ``||v||^2 <= amax``. Control vectors are
interpolated to cells before constructing the log-activation tensor

``H = 1.5 v v^T - 0.5 ||v||^2 I``.

The resulting ``A_inv = exp(H)`` has one stretch ``exp(||v||^2)``, two equal
transverse stretches ``exp(-||v||^2 / 2)``, and unit determinant. Fitted axes
are latent kinematic directions, not identified anatomical fibers.

The physical tensor is invariant under ``v -> -v``, but vector interpolation
is not. A future integration must define and preserve a control-sign gauge;
otherwise physically equivalent opposite controls can cancel at a cell. This
module intentionally does not choose that policy or integrate with the running
inverse. Its nonzero initialization must be fitted from the target-independent,
sign-consistent rest ``ActivationFiber`` field and then projected to the bound,
with the interpolation error reported. Ring muscles require tangents evaluated
or fitted at all four control centers; repeating one regional vector cannot
represent their curved direction field.
"""

from __future__ import annotations

import math

import torch


def shape(n_controls: int) -> tuple[int, int]:
    """Return the optimization-coordinate shape for ``n_controls`` vectors."""
    if n_controls < 1:
        raise ValueError("n_controls must be positive")
    return n_controls, 3


def project(v: torch.Tensor, amax: float) -> torch.Tensor:
    """Project each latent control vector onto ``||v|| <= sqrt(amax)``."""
    _check_vectors(v)
    if not math.isfinite(amax) or amax < 0.0:
        raise ValueError("amax must be finite and nonnegative")
    if amax == 0.0:
        return torch.zeros_like(v)
    radius = v.new_tensor(amax).sqrt()
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    scale = (radius / norm.clamp_min(radius)).clamp_max(1.0)
    return v * scale


def interpolate(
    controls: torch.Tensor,
    control_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Interpolate control vectors with a nonnegative partition of unity."""
    _check_vectors(controls)
    if control_indices.ndim != 2:
        raise ValueError("control_indices must have shape (cells, local_controls)")
    if control_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("control_indices must have an integer dtype")
    if weights.shape != control_indices.shape or not weights.is_floating_point():
        raise ValueError("weights must be floating point and match control_indices")
    if control_indices.device != controls.device or weights.device != controls.device:
        raise ValueError("controls, control_indices, and weights must share a device")
    if weights.dtype != controls.dtype:
        raise ValueError("controls and weights must share a floating-point dtype")
    if not torch.isfinite(weights).all() or torch.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if not torch.allclose(
        weights.sum(dim=1),
        torch.ones(weights.shape[0], **_options(weights)),
        rtol=0.0,
        atol=10.0 * torch.finfo(weights.dtype).eps * weights.shape[1],
    ):
        raise ValueError("weights must sum to one for every cell")
    if torch.any(control_indices < 0) or torch.any(control_indices >= len(controls)):
        raise ValueError("control_indices contain an out-of-range control")
    return (weights[..., None] * controls[control_indices]).sum(dim=1)


def matrices(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A_inv, H)`` for cellwise latent kinematic vectors."""
    _check_vectors(v)
    a = v.square().sum(dim=-1)
    outer = torch.einsum("ni,nj->nij", v, v)
    identity = torch.eye(3, **_options(v)).expand(len(v), 3, 3)
    H = 1.5 * outer - 0.5 * a[:, None, None] * identity
    return torch.matrix_exp(H), H


def packed(Ainv: torch.Tensor) -> torch.Tensor:
    """Pack symmetric ``A_inv - I`` as ``xx yy zz xy yz xz`` for Warp."""
    if Ainv.ndim != 3 or Ainv.shape[-2:] != (3, 3) or not Ainv.is_floating_point():
        raise ValueError("Ainv must be floating point with shape (cells, 3, 3)")
    identity = torch.eye(3, **_options(Ainv))
    offset = Ainv - identity
    return torch.stack(
        (
            offset[:, 0, 0],
            offset[:, 1, 1],
            offset[:, 2, 2],
            offset[:, 0, 1],
            offset[:, 1, 2],
            offset[:, 0, 2],
        ),
        dim=-1,
    )


def validate() -> None:
    """Run CPU invariants and autograd finite-difference checks."""
    dtype = torch.float64
    amax = -math.log(0.65)
    radius = math.sqrt(amax)

    zero = torch.zeros(shape(3), dtype=dtype, requires_grad=True)
    zero_Ainv, zero_H = matrices(zero)
    identity = torch.eye(3, dtype=dtype).expand(3, 3, 3)
    assert torch.equal(zero_H, torch.zeros_like(zero_H))
    assert torch.equal(zero_Ainv, identity)
    assert torch.equal(packed(zero_Ainv), torch.zeros((3, 6), dtype=dtype))
    zero_jacobian = torch.autograd.functional.jacobian(lambda x: matrices(x)[0], zero)
    assert torch.isfinite(zero_jacobian).all()
    assert torch.equal(zero_jacobian, torch.zeros_like(zero_jacobian))

    raw = torch.tensor(
        ((0.9, -0.4, 0.2), (-0.3, 0.1, 0.7), (0.2, 0.3, -0.1)),
        dtype=dtype,
    )
    v = project(raw, amax)
    assert torch.linalg.vector_norm(v, dim=1).max() <= radius + 1e-14
    Ainv, H = matrices(v)
    a = v.square().sum(dim=1)
    assert torch.allclose(H, H.transpose(1, 2))
    trace = torch.diagonal(H, dim1=-2, dim2=-1).sum(dim=-1)
    assert torch.allclose(trace, torch.zeros(3, dtype=dtype))
    assert torch.allclose(torch.linalg.det(Ainv), torch.ones(3, dtype=dtype))
    assert torch.allclose(matrices(-v)[0], Ainv)
    assert torch.allclose(torch.linalg.eigvalsh(H)[:, 2], a)
    assert torch.allclose(
        torch.linalg.eigvalsh(H)[:, :2], -0.5 * a[:, None].expand(3, 2)
    )
    assert torch.allclose(torch.linalg.eigvalsh(Ainv)[:, 2], torch.exp(a))
    assert torch.allclose(
        torch.linalg.eigvalsh(Ainv)[:, :2], torch.exp(-0.5 * a)[:, None].expand(3, 2)
    )

    control_indices = torch.tensor(((0, 1, 2, 3), (0, 1, 2, 3)))
    weights = torch.tensor(((0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)), dtype=dtype)
    controls = project(
        torch.tensor(
            (
                (0.9, 0.1, -0.2),
                (-0.2, 0.8, 0.1),
                (0.2, -0.1, 0.7),
                (-0.6, -0.3, 0.2),
            ),
            dtype=dtype,
        ),
        amax,
    )
    local = interpolate(controls, control_indices, weights)
    assert torch.linalg.vector_norm(local, dim=1).max() <= radius + 1e-14
    repeated = torch.tensor(((0.1, -0.2, 0.3),), dtype=dtype).expand(4, 3)
    assert torch.allclose(interpolate(repeated, control_indices, weights), repeated[:2])
    axis = torch.tensor((radius, 0.0, 0.0), dtype=dtype)
    antipodal = torch.stack((axis, -axis, axis, -axis))
    equal_weights = torch.full((1, 4), 0.25, dtype=dtype)
    one_cell = torch.tensor(((0, 1, 2, 3),))
    assert torch.allclose(matrices(antipodal[:2])[0][0], matrices(antipodal[:2])[0][1])
    assert torch.equal(
        interpolate(antipodal, one_cell, equal_weights),
        torch.zeros((1, 3), dtype=dtype),
    )

    nonzero_test = torch.tensor(
        ((0.2, -0.1, 0.05), (-0.15, 0.07, 0.11)),
        dtype=dtype,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(lambda x: matrices(x)[0], (nonzero_test,))
    zero_test = torch.zeros((2, 3), dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: matrices(x)[0], (zero_test,))
    control_test = controls.detach().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda x: packed(matrices(interpolate(x, control_indices, weights))[0]),
        (control_test,),
    )


def _check_vectors(v: torch.Tensor) -> None:
    if v.ndim != 2 or v.shape[1] != 3 or not v.is_floating_point():
        raise ValueError("latent vectors must be floating point with shape (n, 3)")
    if not torch.isfinite(v).all():
        raise ValueError("latent vectors must be finite")


def _options(tensor: torch.Tensor) -> dict[str, torch.dtype | torch.device]:
    return {"dtype": tensor.dtype, "device": tensor.device}
