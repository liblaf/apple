# ruff: noqa: EM101, SLF001, TRY003
"""Per-primal-state exact FEM-with-optional-IPC sparse Hessian cache.

This is an experiment-local adapter.  It owns no solver hooks and never
patches the model or adjoint path.  Call ``invalidate`` after a material or
primal-state transaction, then ``prepare`` once before CG/shift retries.
"""

from __future__ import annotations

from typing import Any

import scipy.sparse
import torch
from assembled_fem_hvp import AssembledFemHvp
from gpu_free_sparse_hessian import GpuFreeSparseHessian


def _state_key(state: Any) -> tuple[int, int, int]:
    u = state.u
    return id(state), id(u), int(u._version)


class HybridHessian:
    """Cache one exact, unshifted free-space Hessian for a primal state."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._fem: Any | None = None
        self._sparse: Any | None = None
        self._valid_key: tuple[int, int, int] | None = None
        self._diagonal_slots: torch.Tensor | None = None
        self.metadata: dict[str, Any] = {
            "method": "hybrid_exact_unshifted_fem_ipc_free_csr",
            "exact_physical_hessian": True,
            "shift_in_operator": 0.0,
            "numeric_refreshes": 0,
            "prepare_cache_hits": 0,
            "invalidations": 0,
            "fem_constructions": 0,
            "sparse_constructions": 0,
            "persistent_bytes": 0,
        }

    def invalidate(self) -> None:
        """Mark numeric values stale after a primal or material transaction."""
        self._valid_key = None
        self._diagonal_slots = None
        self.metadata["invalidations"] += 1

    def _require_gpu_state(self, state: Any) -> None:
        if not getattr(state.u, "is_cuda", False):
            raise ValueError("HybridHessian is GPU-only")

    def _ensure_contact_hessian(self, state: Any) -> None:
        collision = self._model.collision
        if state.collision is None:
            state.collision = collision.state_at(state.u)
        if state.collision.hess is not None:
            return
        vertices = (collision.vertices + state.u[collision.indices]).numpy(force=True)
        state.collision.hess = collision.potential.hessian(
            collisions=state.collision.collisions,
            mesh=collision.collision_mesh,
            X=vertices,
        ).tocsr()
        if not scipy.sparse.issparse(state.collision.hess):
            raise TypeError(
                "owned IPC hessian builder did not return a SciPy sparse matrix"
            )

    def _refresh_metadata(self) -> None:
        assert self._fem is not None
        assert self._sparse is not None
        fem_metadata = dict(getattr(self._fem, "metadata", {}))
        sparse_metadata = dict(getattr(self._sparse, "metadata", {}))
        self.metadata.update(
            {
                "persistent_bytes": int(getattr(self._fem, "persistent_bytes", 0))
                + int(getattr(self._sparse, "persistent_bytes", 0)),
                "fem_topology_cache_hit": fem_metadata.get("topology_cache_hit"),
                "fem_numeric_setup_seconds": fem_metadata.get("numeric_setup_seconds"),
                "contact_pattern_changed": sparse_metadata.get(
                    "contact_pattern_changed"
                ),
                "contact_union_reused": sparse_metadata.get("union_reused_for_contact"),
                "sparse_symbolic_cache_hit": sparse_metadata.get("symbolic_cache_hit"),
                "pattern_hash": sparse_metadata.get("pattern_hash"),
                "free_nnz": sparse_metadata.get("free_nnz"),
                "lower_nnz": sparse_metadata.get("lower_nnz"),
            }
        )

    def prepare(self, state: Any) -> None:
        """Ensure the exact unshifted matrix is numerically current for ``state``."""
        self._require_gpu_state(state)
        key = _state_key(state)
        if self._valid_key == key:
            self.metadata["prepare_cache_hits"] += 1
            return
        if self._model.collision is not None:
            self._ensure_contact_hessian(state)
        if self._fem is None:
            self._fem = AssembledFemHvp(self._model, state)
            self.metadata["fem_constructions"] += 1
        if self._sparse is None:
            self._sparse = GpuFreeSparseHessian(
                self._model, state, self._fem, shift=0.0
            )
            self.metadata["sparse_constructions"] += 1
        else:
            self._sparse.setup(state, self._fem, shift=0.0)
        if float(self._sparse.metadata.get("shift", 0.0)) != 0.0:
            raise AssertionError(
                "HybridHessian must retain an unshifted physical operator"
            )
        self._valid_key = key
        self._diagonal_slots = None
        self.metadata["numeric_refreshes"] += 1
        self._refresh_metadata()

    def _matrix(self, state: Any) -> torch.Tensor:
        self.prepare(state)
        assert self._sparse is not None
        return self._sparse.matrix

    def apply(self, state: Any, free_direction: torch.Tensor) -> torch.Tensor:
        """Apply the cached physical Hessian; callers add a Newton shift."""
        matrix = self._matrix(state)
        if free_direction.ndim != 1 or free_direction.numel() != matrix.shape[1]:
            raise ValueError("free direction shape differs from cached Hessian")
        if (
            free_direction.device != matrix.device
            or free_direction.dtype != matrix.dtype
        ):
            raise ValueError(
                "free direction device or dtype differs from cached Hessian"
            )
        return torch.sparse.mm(matrix, free_direction[:, None])[:, 0]

    def diagonal(self, state: Any) -> torch.Tensor:
        """Return the exact cached free-DOF diagonal in model free-index order."""
        matrix = self._matrix(state)
        if self._diagonal_slots is None:
            # GpuFreeSparseHessian already owns exact diagonal slots for numeric
            # shifts. Reuse them rather than materializing a dense free matrix.
            slots = getattr(self._sparse, "_gpu", {}).get("diagonal")
            if slots is None:
                raise RuntimeError(
                    "GPU sparse Hessian does not expose structural diagonal slots"
                )
            self._diagonal_slots = slots
        return matrix.values().index_select(0, self._diagonal_slots)
