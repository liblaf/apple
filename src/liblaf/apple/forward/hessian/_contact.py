"""GPU products of the exact CPU IPC Hessian, uploaded once per state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class GpuContactHessian:
    """Keep contact matvecs on CUDA without changing contact construction.

    ``collision.update`` must invalidate ``state.hess`` after geometry changes.
    Call ``invalidate`` after changing contact parameters or mappings directly.
    CPU Hessian values must not be modified in place while cached.
    """

    collision: Any
    uploads: int = 0
    products: int = 0
    _state: Any = field(default=None, repr=False)
    _hess: Any = field(default=None, repr=False)
    _matrix: torch.Tensor | None = field(default=None, repr=False)

    def invalidate(self) -> None:
        self._state = self._hess = self._matrix = None

    def hess_prod(
        self, state: Any, u: torch.Tensor, p: torch.Tensor, output: torch.Tensor
    ) -> None:
        assert u.is_cuda
        assert u.device == p.device == output.device
        assert u.shape == p.shape == output.shape
        assert u.shape[-1] == 3
        assert u.dtype == p.dtype == output.dtype
        if state.hess is None:
            positions = (self.collision.vertices + u[self.collision.indices]).numpy(
                force=True
            )
            state.hess = self.collision.potential.hessian(
                collisions=state.collisions,
                mesh=self.collision.collision_mesh,
                X=positions,
            )
        if (
            self._matrix is None
            or self._state is not state
            or self._hess is not state.hess
        ):
            csr = state.hess.tocsr(copy=True)
            csr.sum_duplicates()
            csr.sort_indices()
            assert csr.shape == (self.collision.indices.numel() * 3,) * 2
            assert np.isfinite(csr.data).all()
            self._matrix = torch.sparse_csr_tensor(
                torch.as_tensor(csr.indptr, device=p.device),
                torch.as_tensor(csr.indices, device=p.device),
                torch.as_tensor(csr.data, device=p.device, dtype=p.dtype),
                size=csr.shape,
                device=p.device,
                dtype=p.dtype,
            )
            self._state, self._hess = state, state.hess
            self.uploads += 1
        assert self._matrix.device == p.device
        assert self._matrix.dtype == p.dtype
        local = p[self.collision.indices].reshape(-1, 1)
        product = torch.sparse.mm(self._matrix, local).reshape(-1, 3)
        output.index_add_(0, self.collision.indices, product)
        self.products += 1

    @property
    def persistent_bytes(self) -> int:
        if self._matrix is None:
            return 0
        return sum(
            t.numel() * t.element_size()
            for t in (
                self._matrix.crow_indices(),
                self._matrix.col_indices(),
                self._matrix.values(),
            )
        )
