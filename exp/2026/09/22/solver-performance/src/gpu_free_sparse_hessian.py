# ruff: noqa: EM101, PLR0915, TRY003
"""GPU numeric refresh for the exact free FEM-plus-IPC Hessian.

Unlike :mod:`free_sparse_hessian`, this module never transfers the large FEM
BSR values back to CPU for every Newton state.  It caches the FEM/free-DOF
symbolic map, uploads only the comparatively small CPU IPC CSR values, and
scatters both contributions into GPU CSR values.  IPC sparsity is deliberately
*not* assumed static: changed IPC patterns remap into the cached structure or
build a new exact union when contact introduces entries outside that structure.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse
import torch


def _digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _pattern_hash(keys: np.ndarray, n: int) -> str:
    rows = keys // n
    counts = np.bincount(rows, minlength=n)
    crow = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    cols = keys % n
    return _digest(np.asarray([n, n], dtype=np.int64), crow, cols)


def _canonical_csr(matrix: Any, *, shape: tuple[int, int]) -> scipy.sparse.csr_matrix:
    if not scipy.sparse.issparse(matrix):
        raise TypeError("state.collision.hess must be a SciPy sparse matrix")
    result = matrix.tocsr(copy=True)
    if result.shape != shape:
        raise ValueError("collision Hessian shape differs from collision.indices")
    result.sum_duplicates()
    result.sort_indices()
    if not np.isfinite(result.data).all():
        raise ValueError("state.collision.hess contains non-finite values")
    return result


@dataclass(frozen=True)
class _Plan:
    """One GPU union-CSR topology for an IPC pattern and shift structure."""

    contact_pattern_hash: str
    shifted: bool
    keys: np.ndarray
    crow: np.ndarray
    col: np.ndarray
    lower_positions: np.ndarray
    lower_crow: np.ndarray
    lower_col: np.ndarray
    fem_direct_slots: np.ndarray
    contact_direct_slots: np.ndarray
    transpose_positions: np.ndarray
    diagonal_slots: np.ndarray
    pattern_hash: str


@dataclass(frozen=True)
class _CachedGpuPlan:
    """Small CPU receipt for a topology whose large maps stay on CUDA."""

    contact_pattern_hash: str
    shifted: bool
    keys: np.ndarray
    pattern_hash: str
    lower_nnz: int


def _make_plan(
    *,
    n_free: int,
    fem_keys: np.ndarray,
    contact_keys: np.ndarray,
    contact_pattern_hash: str,
    shifted: bool,
) -> _Plan:
    """Create an exact symmetric-union CSR plan from scalar free-space keys."""
    if n_free <= 0:
        raise ValueError("n_free must be positive")
    for name, keys in (("fem", fem_keys), ("contact", contact_keys)):
        if keys.ndim != 1 or (
            keys.size and (keys.min() < 0 or keys.max() >= n_free * n_free)
        ):
            message = f"{name} keys are outside the free matrix"
            raise ValueError(message)
    pieces = [fem_keys, (fem_keys % n_free) * n_free + fem_keys // n_free]
    if contact_keys.size:
        pieces.extend(
            (contact_keys, (contact_keys % n_free) * n_free + contact_keys // n_free)
        )
    # Keep all diagonal slots from the first state.  A later Newton shift is
    # then numeric-only instead of forcing a 24M-entry symbolic rebuild.
    pieces.append(np.arange(n_free, dtype=np.int64) * (n_free + 1))
    keys = np.unique(np.concatenate(pieces)).astype(np.int64, copy=False)
    rows = keys // n_free
    col = (keys % n_free).astype(np.int64, copy=False)
    crow = np.concatenate(
        ([0], np.cumsum(np.bincount(rows, minlength=n_free), dtype=np.int64))
    )
    fem_direct_slots = np.searchsorted(keys, fem_keys)
    contact_direct_slots = np.searchsorted(keys, contact_keys)
    transpose_positions = np.searchsorted(
        keys, (keys % n_free) * n_free + keys // n_free
    )
    if not (
        np.array_equal(keys[fem_direct_slots], fem_keys)
        and np.array_equal(keys[contact_direct_slots], contact_keys)
        and np.array_equal(
            keys[transpose_positions], (keys % n_free) * n_free + keys // n_free
        )
    ):
        raise AssertionError("union CSR plan lost a physical nonzero")
    lower_positions = np.flatnonzero(rows >= col).astype(np.int64, copy=False)
    lower_rows = rows[lower_positions]
    lower_crow = np.concatenate(
        ([0], np.cumsum(np.bincount(lower_rows, minlength=n_free), dtype=np.int64))
    )
    diagonal_keys = np.arange(n_free, dtype=np.int64) * (n_free + 1)
    diagonal_slots = np.searchsorted(keys, diagonal_keys)
    if shifted and not np.array_equal(keys[diagonal_slots], diagonal_keys):
        raise AssertionError("shifted CSR plan lacks a diagonal entry")
    return _Plan(
        contact_pattern_hash=contact_pattern_hash,
        shifted=shifted,
        keys=keys,
        crow=crow,
        col=col,
        lower_positions=lower_positions,
        lower_crow=lower_crow,
        lower_col=col[lower_positions],
        fem_direct_slots=fem_direct_slots.astype(np.int64, copy=False),
        contact_direct_slots=contact_direct_slots.astype(np.int64, copy=False),
        transpose_positions=transpose_positions.astype(np.int64, copy=False),
        diagonal_slots=diagonal_slots.astype(np.int64, copy=False),
        pattern_hash=_pattern_hash(keys, n_free),
    )


def _make_plan_gpu(
    *,
    n_free: int,
    fem_keys: np.ndarray,
    contact_keys: np.ndarray,
    contact_pattern_hash: str,
    device: torch.device,
) -> tuple[_CachedGpuPlan, dict[str, torch.Tensor]]:
    """Build the large union/sort/search maps on CUDA once.

    The CPU receipt retains only sorted keys for later sparse contact remaps;
    CSR row pointers, columns, lower map, and FEM maps never round-trip.
    """
    fem = torch.as_tensor(fem_keys, device=device, dtype=torch.int64)
    contact = torch.as_tensor(contact_keys, device=device, dtype=torch.int64)
    diagonal = torch.arange(n_free, device=device, dtype=torch.int64) * (n_free + 1)
    fem_transpose = fem.remainder(n_free) * n_free + torch.div(
        fem, n_free, rounding_mode="floor"
    )
    pieces = [fem, fem_transpose, diagonal]
    if contact.numel():
        pieces.extend(
            (
                contact,
                contact.remainder(n_free) * n_free
                + torch.div(contact, n_free, rounding_mode="floor"),
            )
        )
    keys = torch.unique(torch.cat(pieces), sorted=True)
    rows = torch.div(keys, n_free, rounding_mode="floor")
    col = keys.remainder(n_free)
    crow = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            torch.cumsum(torch.bincount(rows, minlength=n_free), dim=0),
        )
    )
    fem_direct = torch.searchsorted(keys, fem)
    contact_direct = torch.searchsorted(keys, contact)
    transpose_positions = torch.searchsorted(
        keys,
        col * n_free + rows,
    )
    if not bool(torch.all(keys.index_select(0, fem_direct) == fem)):
        raise AssertionError("GPU union CSR plan lost a FEM physical nonzero")
    if not bool(torch.all(keys.index_select(0, contact_direct) == contact)):
        raise AssertionError("GPU union CSR plan lost an IPC physical nonzero")
    if not bool(
        torch.all(keys.index_select(0, transpose_positions) == col * n_free + rows)
    ):
        raise AssertionError("GPU union CSR plan lost a transpose position")
    lower_positions = torch.nonzero(rows >= col, as_tuple=False).flatten()
    lower_rows = rows.index_select(0, lower_positions)
    lower_crow = torch.cat(
        (
            torch.zeros(1, device=device, dtype=torch.int64),
            torch.cumsum(torch.bincount(lower_rows, minlength=n_free), dim=0),
        )
    )
    lower_col = col.index_select(0, lower_positions)
    diagonal_slots = torch.searchsorted(keys, diagonal)
    if not bool(torch.all(keys.index_select(0, diagonal_slots) == diagonal)):
        raise AssertionError("GPU CSR plan lacks a structural diagonal")
    # Only this sorted vector returns to CPU, for future sparse contact remaps
    # and a reproducible pattern receipt.  All large operational maps remain GPU.
    keys_cpu = keys.cpu().numpy()
    plan = _CachedGpuPlan(
        contact_pattern_hash=contact_pattern_hash,
        shifted=True,
        keys=keys_cpu,
        pattern_hash=_pattern_hash(keys_cpu, n_free),
        lower_nnz=int(lower_positions.numel()),
    )
    return plan, {
        "crow": crow,
        "col": col,
        "lower_crow": lower_crow,
        "lower_col": lower_col,
        "lower_positions": lower_positions,
        "fem_direct": fem_direct,
        "contact_direct": contact_direct,
        "transpose_positions": transpose_positions,
        "diagonal": diagonal_slots,
    }


def _assemble_numpy(
    plan: _Plan,
    *,
    fem_values: np.ndarray,
    contact_values: np.ndarray,
    shift: float,
) -> scipy.sparse.csr_matrix:
    """CPU mirror of the GPU scatter, used only by the focused checker."""
    values = np.zeros(
        len(plan.keys), dtype=np.result_type(fem_values, contact_values, float)
    )
    np.add.at(values, plan.fem_direct_slots, fem_values)
    np.add.at(values, plan.contact_direct_slots, contact_values)
    values = 0.5 * (values + values[plan.transpose_positions])
    if shift:
        values[plan.diagonal_slots] += shift
    n = len(plan.crow) - 1
    return scipy.sparse.csr_matrix((values, plan.col, plan.crow), shape=(n, n))


class GpuFreeSparseHessian:
    """Exact FEM-plus-IPC free Hessian with GPU-resident numeric refresh.

    Public API intentionally matches ``FreeSparseHessian``: ``matrix`` is the
    full symmetric free CSR, ``lower`` its diagonal-inclusive lower triangle,
    and ``setup`` refreshes state-dependent values.  A changed IPC sparsity
    pattern visibly rebuilds the union topology; it is never silently dropped.
    """

    def __init__(self, model: Any, state: Any, fem: Any, shift: float = 0.0) -> None:
        if not state.u.is_cuda:
            raise ValueError("GpuFreeSparseHessian is GPU-only")
        if not np.isfinite(shift):
            raise ValueError("shift must be finite")
        self._model = model
        self._device = state.u.device
        self._dtype = state.u.dtype
        self._n_full = int(model.n_points) * int(model.dim)
        if self._n_full != state.u.numel():
            raise ValueError("model dimensions differ from state.u")
        self._free = (
            model.dof_map.free_indices.detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=True)
        )
        if self._free.ndim != 1 or not len(self._free):
            raise ValueError("model.dof_map.free_indices must be nonempty rank one")
        if (
            self._free.min() < 0
            or self._free.max() >= self._n_full
            or len(np.unique(self._free)) != len(self._free)
        ):
            raise ValueError("free_indices must be unique full-DOF indexes")
        self._n_free = len(self._free)
        self._full_to_free = np.full(self._n_full, -1, dtype=np.int64)
        self._full_to_free[self._free] = np.arange(self._n_free, dtype=np.int64)
        self._fem_source: torch.Tensor | None = None
        self._fem_keys: np.ndarray | None = None
        self._fem_topology_digest: str | None = None
        self._fem_topology_identity: tuple[int, int, int, int] | None = None
        self._local_to_full: np.ndarray | None = None
        self._collision_index_digest: str | None = None
        self._plan: _CachedGpuPlan | None = None
        self._gpu: dict[str, torch.Tensor] = {}
        self._values: torch.Tensor | None = None
        self.matrix: torch.Tensor
        self.lower: torch.Tensor
        self.persistent_bytes = 0
        self.metadata: dict[str, Any] = {
            "method": "exact_gpu_numeric_fem_ipc_free_sparse_csr",
            "exact_physical_hessian": True,
            "cpu_scipy_merge_and_restriction": False,
            "dynamic_contact_pattern_supported": True,
        }
        self.setup(state, fem, shift)

    def _check_state(self, state: Any) -> None:
        if state.u.device != self._device or state.u.dtype != self._dtype:
            raise ValueError("state device or dtype differs from this sparse matrix")
        if state.u.numel() != self._n_full:
            raise ValueError("state size differs from this sparse matrix")

    def _ensure_fem_topology(self, fem: Any) -> tuple[bool, float]:
        matrix = fem.matrix
        if matrix.layout != torch.sparse_bsr:
            raise TypeError("fem.matrix must be a Torch BSR tensor")
        if tuple(matrix.shape) != (self._n_full, self._n_full) or self._n_full % 3:
            raise ValueError("FEM matrix shape differs from full model DOFs")
        started = time.perf_counter()
        crow_tensor = matrix.crow_indices()
        col_tensor = matrix.col_indices()
        identity = (
            crow_tensor.data_ptr(),
            col_tensor.data_ptr(),
            crow_tensor.numel(),
            col_tensor.numel(),
        )
        # AssembledFemHvp refreshes values but retains these two topology
        # tensors.  Avoid even an index-only GPU-to-CPU transfer on that path.
        if identity == self._fem_topology_identity:
            return True, time.perf_counter() - started
        crow = crow_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        col = col_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
        digest = _digest(crow, col, self._free)
        if digest == self._fem_topology_digest:
            self._fem_topology_identity = identity
            return True, time.perf_counter() - started
        n_blocks = len(col)
        block_rows = np.repeat(np.arange(len(crow) - 1, dtype=np.int64), np.diff(crow))
        if len(block_rows) != n_blocks:
            raise AssertionError("FEM BSR crow/column mismatch")
        row_full = np.repeat(3 * block_rows[:, None] + np.arange(3), 3, axis=1).reshape(
            -1
        )
        col_full = np.tile(3 * col[:, None] + np.arange(3), (1, 3)).reshape(-1)
        free_rows = self._full_to_free[row_full]
        free_cols = self._full_to_free[col_full]
        keep = (free_rows >= 0) & (free_cols >= 0)
        source = np.flatnonzero(keep).astype(np.int64, copy=False)
        self._fem_keys = (free_rows[keep] * self._n_free + free_cols[keep]).astype(
            np.int64, copy=False
        )
        self._fem_source = torch.as_tensor(
            source, device=self._device, dtype=torch.int64
        )
        self._fem_topology_digest = digest
        self._fem_topology_identity = identity
        self._plan = None
        return False, time.perf_counter() - started

    def _contact(self, state: Any) -> tuple[np.ndarray, np.ndarray, str, float, bool]:
        started = time.perf_counter()
        collision = getattr(self._model, "collision", None)
        collision_state = getattr(state, "collision", None)
        if collision is None:
            # No contact is an exact empty contribution, rather than a fake
            # collision object. The FEM/free topology remains fully valid.
            empty = np.empty(0, dtype=np.int64)
            return (
                empty,
                np.empty(0, dtype=np.float64),
                "no-contact",
                time.perf_counter() - started,
                False,
            )
        if collision_state is None:
            raise ValueError(
                "GpuFreeSparseHessian requires a collision state when contact is enabled"
            )
        vertices = collision.indices.detach().cpu().numpy().astype(np.int64, copy=False)
        if vertices.ndim != 1 or (
            vertices.size
            and (vertices.min() < 0 or vertices.max() >= int(self._model.n_points))
        ):
            raise ValueError("collision.indices contains a vertex outside the model")
        vertex_digest = _digest(vertices)
        mapping_changed = vertex_digest != self._collision_index_digest
        if mapping_changed:
            self._local_to_full = (3 * vertices[:, None] + np.arange(3)).reshape(-1)
            self._collision_index_digest = vertex_digest
            self._plan = None
        assert self._local_to_full is not None
        local = _canonical_csr(
            getattr(collision_state, "hess", None),
            shape=(len(self._local_to_full), len(self._local_to_full)),
        )
        pattern_digest = _digest(
            np.asarray(local.shape, dtype=np.int64),
            local.indptr,
            local.indices,
            self._local_to_full,
        )
        coo = local.tocoo(copy=False)
        full_row = self._local_to_full[coo.row]
        full_col = self._local_to_full[coo.col]
        free_row = self._full_to_free[full_row]
        free_col = self._full_to_free[full_col]
        keep = (free_row >= 0) & (free_col >= 0)
        keys = (free_row[keep] * self._n_free + free_col[keep]).astype(
            np.int64, copy=False
        )
        values = np.asarray(coo.data[keep])
        return (
            keys,
            values,
            pattern_digest,
            time.perf_counter() - started,
            mapping_changed,
        )

    def _install_plan(self, plan: _CachedGpuPlan, gpu: dict[str, torch.Tensor]) -> None:
        self._plan = plan
        self._gpu = gpu
        self._values = torch.zeros(
            len(plan.keys), device=self._device, dtype=self._dtype
        )

    def _reuse_existing_union(
        self, contact_keys: np.ndarray, contact_pattern_hash: str
    ) -> bool:
        """Retarget contact slots when new contacts fit the existing union CSR."""
        if self._plan is None:
            return False
        keys = self._plan.keys
        direct = np.searchsorted(keys, contact_keys)
        if np.any(direct == len(keys)) or not np.array_equal(
            keys[direct], contact_keys
        ):
            return False
        self._plan = replace(self._plan, contact_pattern_hash=contact_pattern_hash)
        self._gpu["contact_direct"] = torch.as_tensor(
            direct, device=self._device, dtype=torch.int64
        )
        return True

    def setup(self, state: Any, fem: Any, shift: float = 0.0) -> None:
        """Refresh exact FEM/IPC values, rebuilding only changed symbolic maps."""
        self._check_state(state)
        if not np.isfinite(shift):
            raise ValueError("shift must be finite")
        started = time.perf_counter()
        fem_started = time.perf_counter()
        fem.setup(state)
        torch.cuda.synchronize(self._device)
        fem_setup_seconds = time.perf_counter() - fem_started
        fem_topology_cache_hit, fem_topology_seconds = self._ensure_fem_topology(fem)
        assert self._fem_keys is not None
        assert self._fem_source is not None
        (
            contact_keys,
            contact_values,
            contact_pattern_hash,
            contact_pattern_seconds,
            mapping_changed,
        ) = self._contact(state)
        contact_pattern_changed = (
            self._plan is None
            or self._plan.contact_pattern_hash != contact_pattern_hash
        )
        symbolic_rebuilt = False
        union_reused_for_contact = False
        symbolic_seconds = 0.0
        contact_remap_started = time.perf_counter()
        if contact_pattern_changed:
            # Common case: a different IPC active set whose entries already lie
            # in the cached FEM/free graph.  Retarget only sparse contact slots.
            union_reused_for_contact = self._reuse_existing_union(
                contact_keys, contact_pattern_hash
            )
        contact_remap_seconds = time.perf_counter() - contact_remap_started
        if self._plan is None or (
            contact_pattern_changed and not union_reused_for_contact
        ):
            symbolic_rebuilt = True
            symbolic_started = time.perf_counter()
            plan, gpu = _make_plan_gpu(
                n_free=self._n_free,
                fem_keys=self._fem_keys,
                contact_keys=contact_keys,
                contact_pattern_hash=contact_pattern_hash,
                device=self._device,
            )
            self._install_plan(plan, gpu)
            torch.cuda.synchronize(self._device)
            symbolic_seconds = time.perf_counter() - symbolic_started
        assert self._plan is not None
        assert self._values is not None
        numeric_started = time.perf_counter()
        values = self._values
        values.zero_()
        fem_values = fem.matrix.values().reshape(-1).index_select(0, self._fem_source)
        values.index_add_(0, self._gpu["fem_direct"], fem_values)
        contact_upload_started = time.perf_counter()
        contact_gpu = torch.as_tensor(
            contact_values, device=self._device, dtype=self._dtype
        )
        torch.cuda.synchronize(self._device)
        contact_upload_seconds = time.perf_counter() - contact_upload_started
        if contact_gpu.numel() != self._gpu["contact_direct"].numel():
            raise AssertionError("contact value/pattern length changed during setup")
        values.index_add_(0, self._gpu["contact_direct"], contact_gpu)
        transpose_values = values.index_select(0, self._gpu["transpose_positions"])
        skew = values - transpose_values
        denominator = torch.linalg.vector_norm(values).clamp_min(
            torch.finfo(values.dtype).tiny
        )
        relative_asymmetry = float(torch.linalg.vector_norm(skew) / denominator)
        tolerance = 2.0e-11 if values.element_size() >= 8 else 2.0e-5
        if not np.isfinite(relative_asymmetry) or relative_asymmetry > tolerance:
            message = f"assembled free Hessian is nonfinite or materially asymmetric: {relative_asymmetry:.3e}"
            raise ValueError(message)
        values.copy_(0.5 * (values + transpose_values))
        if shift:
            values.index_add_(
                0,
                self._gpu["diagonal"],
                torch.full(
                    (self._n_free,),
                    float(shift),
                    device=self._device,
                    dtype=self._dtype,
                ),
            )
        self.matrix = torch.sparse_csr_tensor(
            self._gpu["crow"],
            self._gpu["col"],
            values,
            size=(self._n_free, self._n_free),
            device=self._device,
            dtype=self._dtype,
        )
        lower_values = values.index_select(0, self._gpu["lower_positions"])
        self.lower = torch.sparse_csr_tensor(
            self._gpu["lower_crow"],
            self._gpu["lower_col"],
            lower_values,
            size=(self._n_free, self._n_free),
            device=self._device,
            dtype=self._dtype,
        )
        explicit_zero_nnz = int((values == 0).sum().item())
        torch.cuda.synchronize(self._device)
        gpu_numeric_seconds = time.perf_counter() - numeric_started
        tensor_bytes = sum(x.numel() * x.element_size() for x in self._gpu.values())
        self.persistent_bytes = (
            tensor_bytes
            + values.numel() * values.element_size()
            + lower_values.numel() * lower_values.element_size()
            + self._fem_source.numel() * self._fem_source.element_size()
        )
        self.metadata.update(
            {
                "n_full": self._n_full,
                "n_free": self._n_free,
                "fem_nnz": int(self._fem_keys.size),
                "collision_nnz_local": int(state.collision.hess.nnz)
                if state.collision is not None
                else 0,
                "collision_nnz_scattered": int(contact_keys.size),
                "free_nnz": len(self._plan.keys),
                "lower_nnz": self._plan.lower_nnz,
                "pattern_hash": self._plan.pattern_hash,
                "physical_hessian_pattern_hash": self._plan.pattern_hash,
                "matrix_pattern_hash": self._plan.pattern_hash,
                "shift": float(shift),
                "relative_asymmetry_before_roundoff_symmetrization": relative_asymmetry,
                "roundoff_symmetrization_tolerance": tolerance,
                "fem_setup_seconds": fem_setup_seconds,
                "fem_topology_seconds": fem_topology_seconds,
                "fem_topology_cache_hit": fem_topology_cache_hit,
                "contact_pattern_seconds": contact_pattern_seconds,
                "contact_remap_seconds": contact_remap_seconds,
                "contact_pattern_changed": contact_pattern_changed,
                "collision_mapping_changed": mapping_changed,
                "union_reused_for_contact": union_reused_for_contact,
                "symbolic_rebuilt": symbolic_rebuilt,
                "symbolic_cache_hit": not symbolic_rebuilt,
                "symbolic_seconds": symbolic_seconds,
                "contact_upload_seconds": contact_upload_seconds,
                "explicit_zero_nnz": explicit_zero_nnz,
                "gpu_numeric_seconds": gpu_numeric_seconds,
                "numeric_setup_seconds": time.perf_counter() - started,
                "persistent_bytes": self.persistent_bytes,
            }
        )
