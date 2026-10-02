# ruff: noqa: EM101, TRY003
"""Exact free-DOF CSR Newton matrix from frozen FEM and IPC Hessians.

The FEM matrix is assembled by :mod:`assembled_fem_hvp`; IPC still originates
from its exact CPU SciPy CSR.  This module deliberately performs the merge and
free-DOF restriction in SciPy, where arbitrary collision and free-index maps
are straightforward to audit.  Its timing metadata makes that CPU work
visible rather than presenting it as a GPU sparse-solver speed-up.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import scipy.sparse
import torch


def _as_csr(matrix: Any, *, name: str) -> scipy.sparse.csr_matrix:
    if not scipy.sparse.issparse(matrix):
        message = f"{name} must be a SciPy sparse matrix"
        raise TypeError(message)
    result = matrix.tocsr(copy=False)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        message = f"{name} must be square"
        raise ValueError(message)
    if not np.isfinite(result.data).all():
        message = f"{name} contains non-finite values"
        raise ValueError(message)
    result.sum_duplicates()
    result.sort_indices()
    return result


def _fem_cpu_csr(fem: Any) -> scipy.sparse.csr_matrix:
    """Transfer BSR structure/blocks and expand them to exact scalar CSR."""
    matrix = fem.matrix
    if matrix.layout != torch.sparse_bsr:
        raise TypeError("fem.matrix must be a Torch BSR tensor")
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 3:
        raise ValueError("FEM BSR matrix must be square with 3-by-3 blocks")
    values = matrix.values().detach().cpu().numpy()
    columns = matrix.col_indices().detach().cpu().numpy()
    rows = matrix.crow_indices().detach().cpu().numpy()
    result = scipy.sparse.bsr_matrix(
        (values, columns, rows), shape=tuple(matrix.shape), blocksize=(3, 3)
    ).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    return result


def _collision_global_csr(
    model: Any, state: Any, *, n_full: int
) -> scipy.sparse.csr_matrix:
    collision = getattr(model, "collision", None)
    collision_state = getattr(state, "collision", None)
    if collision is None or collision_state is None:
        raise ValueError(
            "FreeSparseHessian requires an owned collision and collision state"
        )
    local = _as_csr(getattr(collision_state, "hess", None), name="state.collision.hess")
    local_vertices = (
        collision.indices.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    if local.shape != (3 * len(local_vertices), 3 * len(local_vertices)):
        raise ValueError("collision Hessian shape differs from collision.indices")
    if local_vertices.size and (
        local_vertices.min() < 0 or 3 * local_vertices.max() + 2 >= n_full
    ):
        raise ValueError("collision.indices contains a vertex outside the full state")
    # A collision Hessian is expressed in collision-local vertex DOFs.  Scatter
    # both axes through the (possibly non-contiguous) global vertex map. COO is
    # used only for the existing IPC nonzeros; no dense global matrix is made.
    local_to_full = (3 * local_vertices[:, None] + np.arange(3)).reshape(-1)
    coo = local.tocoo(copy=False)
    result = scipy.sparse.coo_matrix(
        (coo.data, (local_to_full[coo.row], local_to_full[coo.col])),
        shape=(n_full, n_full),
    ).tocsr()
    result.sum_duplicates()
    result.sort_indices()
    return result


def _relative_asymmetry(matrix: scipy.sparse.csr_matrix) -> float:
    skew = matrix - matrix.T
    denominator = max(float(scipy.sparse.linalg.norm(matrix)), np.finfo(float).tiny)
    return float(scipy.sparse.linalg.norm(skew) / denominator)


def _pattern_hash(matrix: scipy.sparse.csr_matrix) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(matrix.shape, dtype=np.int64).tobytes())
    digest.update(matrix.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(matrix.indices.astype(np.int64, copy=False).tobytes())
    return digest.hexdigest()


def _torch_csr(
    matrix: scipy.sparse.csr_matrix, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.sparse_csr_tensor(
        torch.as_tensor(matrix.indptr, dtype=torch.int64, device=device),
        torch.as_tensor(matrix.indices, dtype=torch.int64, device=device),
        torch.as_tensor(matrix.data, dtype=dtype, device=device),
        size=matrix.shape,
        dtype=dtype,
        device=device,
    )


def _persistent_bytes(matrix: torch.Tensor) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in (matrix.crow_indices(), matrix.col_indices(), matrix.values())
    )


class FreeSparseHessian:
    """Exact FEM-plus-IPC Hessian restricted to the model's free DOFs.

    ``matrix`` is the full symmetric free-space CSR matrix. ``lower`` contains
    its diagonal-inclusive lower triangle for cuDSS symmetric storage. A
    nonzero ``shift`` is the linear-system shift only; it is reported
    separately and never changes the claimed physical Hessian.
    """

    def __init__(self, model: Any, state: Any, fem: Any, shift: float = 0.0) -> None:
        if not state.u.is_cuda:
            raise ValueError("FreeSparseHessian is GPU-only")
        self._model = model
        self._state_device = state.u.device
        self._state_dtype = state.u.dtype
        self._n_full = int(model.n_points) * int(model.dim)
        if self._n_full != state.u.numel():
            raise ValueError("model dimensions differ from state.u")
        self.matrix: torch.Tensor
        self.lower: torch.Tensor
        self.persistent_bytes = 0
        self.metadata: dict[str, Any] = {
            "method": "exact_frozen_fem_ipc_free_sparse_csr",
            "exact_physical_hessian": True,
            "cpu_scipy_merge_and_restriction": True,
        }
        self.setup(state, fem, shift)

    def setup(self, state: Any, fem: Any, shift: float = 0.0) -> None:
        """Refresh FEM, scatter exact IPC CSR, then restrict to free DOFs."""
        if state.u.device != self._state_device or state.u.dtype != self._state_dtype:
            raise ValueError("state device or dtype differs from this sparse matrix")
        if state.u.numel() != self._n_full:
            raise ValueError("state size differs from this sparse matrix")
        if not np.isfinite(shift):
            raise ValueError("shift must be finite")
        started = time.perf_counter()
        fem_started = time.perf_counter()
        fem.setup(state)
        torch.cuda.synchronize(state.u.device)
        fem_setup_seconds = time.perf_counter() - fem_started
        scipy_started = time.perf_counter()
        fem_csr = _fem_cpu_csr(fem)
        if fem_csr.shape != (self._n_full, self._n_full):
            raise ValueError("FEM matrix shape differs from model full DOFs")
        collision_csr = _collision_global_csr(self._model, state, n_full=self._n_full)
        full = (fem_csr + collision_csr).tocsr()
        full.sum_duplicates()
        full.sort_indices()
        free = (
            self._model.dof_map.free_indices.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        if free.ndim != 1 or not len(free):
            raise ValueError("model.dof_map.free_indices must be nonempty rank one")
        if (
            free.min() < 0
            or free.max() >= self._n_full
            or len(np.unique(free)) != len(free)
        ):
            raise ValueError("free_indices must be unique full-DOF indexes")
        restricted = full[free][:, free].tocsr()
        restricted.sum_duplicates()
        restricted.sort_indices()
        relative_asymmetry = _relative_asymmetry(restricted)
        # CPU IPC is normally float64. This accepts only ordinary floating
        # roundoff; anything larger is an assembly bug and must remain visible.
        tolerance = 2.0e-11 if restricted.dtype.itemsize >= 8 else 2.0e-5
        if relative_asymmetry > tolerance:
            message = (
                "assembled free Hessian is materially asymmetric: "
                f"{relative_asymmetry:.3e}"
            )
            raise ValueError(message)
        # The measured asymmetry is only roundoff, so form the exact symmetric
        # representative required by cuDSS. This is recorded in metadata.
        symmetric = (0.5 * (restricted + restricted.T)).tocsr()
        symmetric.sum_duplicates()
        symmetric.sort_indices()
        physical_pattern_hash = _pattern_hash(symmetric)
        if shift:
            shifted = (
                symmetric
                + float(shift)
                * scipy.sparse.eye(
                    symmetric.shape[0], format="csr", dtype=symmetric.dtype
                )
            ).tocsr()
            shifted.sum_duplicates()
            shifted.sort_indices()
        else:
            shifted = symmetric
        lower = scipy.sparse.tril(shifted, format="csr")
        scipy_seconds = time.perf_counter() - scipy_started
        upload_started = time.perf_counter()
        self.matrix = _torch_csr(shifted, device=state.u.device, dtype=state.u.dtype)
        self.lower = _torch_csr(lower, device=state.u.device, dtype=state.u.dtype)
        torch.cuda.synchronize(state.u.device)
        upload_seconds = time.perf_counter() - upload_started
        self.persistent_bytes = _persistent_bytes(self.matrix) + _persistent_bytes(
            self.lower
        )
        self.metadata.update(
            {
                "n_full": self._n_full,
                "n_free": len(free),
                "fem_nnz": int(fem_csr.nnz),
                "collision_nnz_local": int(state.collision.hess.nnz),
                "collision_nnz_scattered": int(collision_csr.nnz),
                "free_nnz": int(symmetric.nnz),
                "lower_nnz": int(lower.nnz),
                "pattern_hash": physical_pattern_hash,
                "shift": float(shift),
                "physical_hessian_pattern_hash": physical_pattern_hash,
                "matrix_pattern_hash": _pattern_hash(shifted),
                "relative_asymmetry_before_roundoff_symmetrization": relative_asymmetry,
                "roundoff_symmetrization_tolerance": tolerance,
                "fem_setup_seconds": fem_setup_seconds,
                "cpu_scipy_seconds": scipy_seconds,
                "gpu_upload_seconds": upload_seconds,
                "numeric_setup_seconds": time.perf_counter() - started,
                "persistent_bytes": self.persistent_bytes,
            }
        )


def _synthetic_check() -> None:
    """CPU-only mapping check with noncontiguous collision and free indices."""
    fem = scipy.sparse.csr_matrix(np.diag(np.arange(1.0, 13.0)))
    local = scipy.sparse.csr_matrix(
        np.array(
            [
                [2.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.5, 3.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
            ]
        )
    )
    # collision vertices 3 and 0 map their local DOFs into nonadjacent global slots.
    mapping = np.array([9, 10, 11, 0, 1, 2], dtype=np.int64)
    coo = local.tocoo()
    mapped = scipy.sparse.coo_matrix(
        (coo.data, (mapping[coo.row], mapping[coo.col])), shape=(12, 12)
    ).tocsr()
    free = np.array([10, 1, 9, 2], dtype=np.int64)
    result = (fem + mapped)[free][:, free].toarray()
    expected = np.array(
        [
            [14.0, 0.0, 0.5, 0.0],
            [0.0, 8.0, 0.0, 0.0],
            [0.5, 0.0, 12.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
        ]
    )
    np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    _synthetic_check()
