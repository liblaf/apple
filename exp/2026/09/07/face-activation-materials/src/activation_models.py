# ruff: noqa: C901, EM101, EM102, TRY003
"""Activation-coordinate maps for the activation-space constraints experiment.

The existing Warp material field stores the symmetric components of
``A_inv - I`` in ``(xx, yy, zz, xy, yz, xz)`` order.  This module keeps the
new coordinate systems in PyTorch until that adapter boundary.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

_TENSOR_CAP_FACTOR = float(np.sqrt(1.5))


def shape(mode: str, n: int) -> tuple[int, ...]:
    """Return the unconstrained optimization-coordinate shape for ``mode``."""
    if n < 1:
        raise ValueError("n must be positive")
    match mode:
        case "Raw6" | "G6":
            return (n, 6)
        case "G5":
            return (n, 5)
        case "F":
            return (n, 1)
        case "Shared":
            return (1,)
        case _:
            raise ValueError(f"unknown activation mode: {mode!r}")


def project(q: torch.Tensor, mode: str, amax: float) -> torch.Tensor:
    """Project coordinates onto their declared experimental bound.

    ``Raw6`` deliberately stays unbounded: it is the historical reference
    model.  G6 and G5 use orthonormal tensor coordinates, so their Euclidean
    coordinate norm equals ``||H||_F``.
    """
    if amax < 0:
        raise ValueError("amax must be nonnegative")
    _check_shape(q, mode, q.shape[0] if q.ndim else 1)
    if mode == "Raw6":
        return q
    if mode in {"F", "Shared"}:
        return q.clamp(0.0, amax)
    cap = q.new_tensor(_TENSOR_CAP_FACTOR * amax)
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    return q * (cap / norm.clamp_min(cap)).clamp_max(1.0)


def matrices(
    q: torch.Tensor,
    mode: str,
    fibers: torch.Tensor,
    gamma: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A_inv, H)`` for activation coordinates and reference fibers.

    For Raw6, ``H`` is its symmetric offset for diagnostics; it is not a
    matrix logarithm.  All other modes use ``A_inv = exp(H)``.
    """
    if fibers.ndim != 2 or fibers.shape[-1] != 3:
        raise ValueError("fibers must have shape (n, 3)")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    n = fibers.shape[0]
    _check_shape(q, mode, n)
    dtype, device = q.dtype, q.device
    if not q.is_floating_point():
        raise TypeError("activation coordinates must be floating point")
    fibers = fibers.to(dtype=dtype, device=device)

    if mode == "Raw6":
        H = _symmetric6(q)
        return torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3) + H, H
    if mode == "G6":
        H = torch.einsum("ni,ijk->njk", q, _g6_basis(dtype, device))
    elif mode == "G5":
        H = torch.einsum("ni,ijk->njk", q, _g5_basis(dtype, device))
    else:
        norms = torch.linalg.vector_norm(fibers, dim=-1)
        if torch.any(norms == 0):
            raise ValueError("fibers must be nonzero")
        f = fibers / norms[:, None]
        P = torch.einsum("ni,nj->nij", f, f)
        identity = torch.eye(3, dtype=dtype, device=device).expand(n, 3, 3)
        a = q.reshape(-1) if mode == "F" else q.reshape(1).expand(n)
        H = a[:, None, None] * ((1.0 + gamma) * P - gamma * identity)
        Q = identity - P
        Ainv = (
            torch.exp(a)[:, None, None] * P + torch.exp(-gamma * a)[:, None, None] * Q
        )
        return Ainv, H
    return torch.matrix_exp(H), H


def packed(Ainv: torch.Tensor) -> torch.Tensor:
    """Pack the symmetric Warp offset ``A_inv - I`` as xx yy zz xy yz xz."""
    if Ainv.ndim < 2 or Ainv.shape[-2:] != (3, 3):
        raise ValueError("Ainv must end in shape (3, 3)")
    identity = torch.eye(3, dtype=Ainv.dtype, device=Ainv.device)
    offset = Ainv - identity
    return torch.stack(
        (
            offset[..., 0, 0],
            offset[..., 1, 1],
            offset[..., 2, 2],
            offset[..., 0, 1],
            offset[..., 1, 2],
            offset[..., 0, 2],
        ),
        dim=-1,
    )


def face_graph(
    points: np.ndarray,
    tets: np.ndarray,
    active_ids: Iterable[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build once-only face-sharing edges among active tetrahedra.

    The returned indices are local indices into ``active_ids`` so they directly
    index per-active-cell control arrays.  Faces shared through an inactive or
    differently compartmented cell are therefore absent.
    """
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.intp)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (points, 3)")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError("tets must have shape (cells, 4)")
    ids = (
        np.flatnonzero(active_ids)
        if np.asarray(active_ids).dtype == bool
        else np.asarray(active_ids, dtype=np.intp)
    )
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(tets)):
        raise ValueError("active_ids must be valid tetrahedron indices")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("active_ids must not repeat cells")

    faces: dict[tuple[int, int, int], list[int]] = {}
    for local, cell in enumerate(tets[ids]):
        for omitted in range(4):
            key = tuple(sorted(np.delete(cell, omitted).tolist()))
            faces.setdefault(key, []).append(local)
    edges: list[tuple[int, int, float]] = []
    centers = points[tets[ids]].mean(axis=1)
    for face, owners in faces.items():
        if len(owners) == 2:
            i, j = owners
            tri = points[np.asarray(face)]
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            distance = np.linalg.norm(centers[i] - centers[j])
            if distance == 0.0:
                raise ValueError("face-sharing tetrahedron centroids must differ")
            edges.append((i, j, area / distance))
        elif len(owners) > 2:
            raise ValueError("a tetrahedral face has more than two active owners")
    if not edges:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=float),
        )
    array = np.asarray(edges)
    return array[:, 0].astype(np.intp), array[:, 1].astype(np.intp), array[:, 2]


def validate() -> None:
    """Run lightweight invariant and derivative checks for this adapter."""
    dtype = torch.float64
    fibers = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=dtype)
    identity = torch.eye(3, dtype=dtype).expand(2, 3, 3)
    for mode in ("Raw6", "G6", "G5", "F", "Shared"):
        q = torch.zeros(shape(mode, 2), dtype=dtype)
        Ainv, _ = matrices(q, mode, fibers)
        assert torch.allclose(Ainv, identity)
        assert torch.allclose(packed(Ainv), torch.zeros((2, 6), dtype=dtype))
        if mode != "Raw6":
            assert torch.allclose(torch.linalg.det(Ainv), torch.ones(2, dtype=dtype))
    q = torch.tensor([[0.2], [0.2]], dtype=dtype)
    Aplus, _ = matrices(q, "F", fibers)
    Aminus, _ = matrices(q, "F", -fibers)
    assert torch.allclose(Aplus, Aminus)
    assert torch.allclose(torch.linalg.det(Aplus), torch.ones(2, dtype=dtype))
    gamma0, _ = matrices(q, "F", fibers, gamma=0.0)
    assert torch.allclose(torch.linalg.det(gamma0), torch.exp(q[:, 0]))
    assert torch.allclose(
        packed(gamma0),
        torch.tensor(
            (
                (torch.exp(q[0, 0]) - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, torch.exp(q[1, 0]) - 1.0, 0.0, 0.0, 0.0, 0.0),
            ),
            dtype=dtype,
        ),
    )
    assert torch.allclose(
        project(torch.tensor([[-2.0], [4.0]], dtype=dtype), "F", 0.3),
        torch.tensor([[0.0], [0.3]], dtype=dtype),
    )
    capped = project(torch.full((2, 5), 3.0, dtype=dtype), "G5", 0.3)
    assert torch.all(
        torch.linalg.vector_norm(capped, dim=-1) <= _TENSOR_CAP_FACTOR * 0.3 + 1e-12
    )
    g5_q = torch.tensor(
        ((0.1, -0.2, 0.03, -0.04, 0.05), (-0.2, 0.1, -0.04, 0.03, -0.02)), dtype=dtype
    )
    g5_Ainv, _ = matrices(g5_q, "G5", fibers)
    assert torch.allclose(torch.linalg.det(g5_Ainv), torch.ones(2, dtype=dtype))
    scalar_q = torch.tensor(((0.1,), (0.2,)), dtype=dtype)
    same_fibers = fibers[:1].expand_as(fibers)
    _, fiber_H = matrices(scalar_q, "F", same_fibers)
    assert torch.allclose(
        fiber_H.square().sum(dim=(1, 2)) / 1.5, scalar_q[:, 0].square()
    )
    assert torch.allclose(
        (fiber_H[0] - fiber_H[1]).square().sum() / 1.5,
        (scalar_q[0, 0] - scalar_q[1, 0]).square(),
    )
    for mode, test_q in (
        (
            "Raw6",
            torch.tensor(
                (
                    (0.1, -0.1, 0.05, 0.02, -0.03, 0.04),
                    (0.02, 0.03, -0.04, 0.01, 0.05, -0.02),
                ),
                dtype=dtype,
            ),
        ),
        (
            "G6",
            torch.tensor(
                (
                    (0.1, -0.1, 0.05, 0.02, -0.03, 0.04),
                    (0.02, 0.03, -0.04, 0.01, 0.05, -0.02),
                ),
                dtype=dtype,
            ),
        ),
        ("G5", g5_q),
        ("F", scalar_q),
        ("Shared", torch.tensor((0.1,), dtype=dtype)),
    ):
        test_q.requires_grad_()
        assert torch.autograd.gradcheck(
            lambda x, mode=mode: matrices(x, mode, fibers)[0], (test_q,)
        )


def _check_shape(q: torch.Tensor, mode: str, n: int) -> None:
    expected = shape(mode, n)
    if tuple(q.shape) != expected:
        raise ValueError(
            f"{mode} coordinates must have shape {expected}, got {tuple(q.shape)}"
        )


def _symmetric6(q: torch.Tensor) -> torch.Tensor:
    H = torch.zeros((*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    H[..., 0, 0], H[..., 1, 1], H[..., 2, 2] = q[..., 0], q[..., 1], q[..., 2]
    H[..., 0, 1] = H[..., 1, 0] = q[..., 3]
    H[..., 1, 2] = H[..., 2, 1] = q[..., 4]
    H[..., 0, 2] = H[..., 2, 0] = q[..., 5]
    return H


def _g6_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    basis = torch.zeros((6, 3, 3), dtype=dtype, device=device)
    basis[0, 0, 0], basis[1, 1, 1], basis[2, 2, 2] = 1.0, 1.0, 1.0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    basis[3, 0, 1] = basis[3, 1, 0] = inv_sqrt2
    basis[4, 1, 2] = basis[4, 2, 1] = inv_sqrt2
    basis[5, 0, 2] = basis[5, 2, 0] = inv_sqrt2
    return basis


def _g5_basis(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    basis = _g6_basis(dtype, device)[[3, 4, 5]]
    diagonal = torch.zeros((2, 3, 3), dtype=dtype, device=device)
    diagonal[0, 0, 0], diagonal[0, 1, 1] = 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)
    diagonal[1, 0, 0] = diagonal[1, 1, 1] = 1.0 / np.sqrt(6.0)
    diagonal[1, 2, 2] = -2.0 / np.sqrt(6.0)
    return torch.cat((diagonal, basis))
