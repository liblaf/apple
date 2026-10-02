"""Low-memory exact active-strain bulk BSR plus IPC product backend."""

from __future__ import annotations

import time
from typing import Any

import torch
from active_assembled_fem import ActiveAssembledFemHvp

from liblaf.apple.forward.hessian._contact import GpuContactHessian
from liblaf.apple.forward.hessian._problem import HessianProblem


class ActiveBulkContactHessian(HessianProblem):
    """Avoid the free-coordinate CSR union while retaining exact products.

    The reusable BSR matrix stores only the bulk operator in full coordinates.
    Each HVP adds the precomputed IPC stencil product and then restricts to the
    existing free-DOF map.
    """

    def __init__(self, delegate: Any) -> None:
        super().__init__(delegate, "matrix_free")
        assert self.model.collision is not None
        self.contact = GpuContactHessian(self.model.collision)
        self.fem: ActiveAssembledFemHvp | None = None
        self._key: tuple[Any, ...] | None = None
        self.refreshes = 0
        self.products = 0
        self.setup_seconds = 0.0

    def _prepare(self, state: Any) -> None:
        key = (id(state), id(state.u), state.u._version)  # noqa: SLF001
        if key == self._key:
            return
        started = time.perf_counter()
        if self.fem is None:
            self.fem = ActiveAssembledFemHvp(self.model, state)
        else:
            self.fem.setup(state)
        torch.cuda.synchronize(state.u.device)
        self.setup_seconds += time.perf_counter() - started
        self.refreshes += 1
        self._key = key

    def hess_prod(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.products += 1
        self._prepare(state)
        assert self.fem is not None
        assert self.contact is not None
        full = self.model.dof_map.to_full_grad(direction)
        output = self.fem.apply(full)
        self.contact.hess_prod(state.collision, state.u, full, output)
        return self.model.dof_map.to_free_grad(output)

    def report(self) -> dict[str, Any]:
        assert self.fem is not None
        assert self.contact is not None
        return {
            "backend": "experiment_local_active_bulk_bsr_plus_gpu_contact",
            "bulk_storage_bytes": self.fem.persistent_bytes,
            "contact_storage_bytes": self.contact.persistent_bytes,
            "persistent_bytes": self.fem.persistent_bytes
            + self.contact.persistent_bytes,
            "fem_refreshes": self.refreshes,
            "hess_products": self.products,
            "setup_seconds": self.setup_seconds,
            "contact_uploads": self.contact.uploads,
            "fem_metadata": self.fem.metadata,
        }


__all__ = ["ActiveBulkContactHessian"]
