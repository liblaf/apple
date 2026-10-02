"""Solve-local exact GPU Hessian backends for a forward problem."""

from __future__ import annotations

import time
from typing import Any, Literal

import torch

from ._contact import GpuContactHessian

type HessianBackend = Literal["matrix_free", "gpu_contact", "gpu_sparse"]


class HessianProblem:
    """Delegate mechanics while replacing only exact free-coordinate HVPs.

    Own one instance per fixed-material solve. The mesh, free-DOF map, and
    prescribed values must remain fixed. Call ``invalidate`` after external
    material edits; create a new instance for a different mesh or DOF map.
    Energy, gradients, diagonals, CCD and updates always use the original
    problem. Sparse values refresh after every displacement update and are
    shared by all PCG iterations and shift retries at that state.

    No numerical fallback is performed when an explicit GPU backend fails.
    """

    def __init__(self, delegate: Any, backend: HessianBackend = "matrix_free") -> None:
        assert backend in ("matrix_free", "gpu_contact", "gpu_sparse")
        self.delegate = delegate
        self.model = delegate.model
        self.backend = backend
        self.contact = (
            GpuContactHessian(self.model.collision)
            if backend == "gpu_contact"
            else None
        )
        self.fem = None
        self.sparse = None
        self._key = None
        self.refreshes = 0
        self.products = 0
        self.setup_seconds = 0.0
        self.first_setup_seconds = 0.0
        self.symbolic_rebuilds = 0
        self.contact_remaps = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def invalidate(self) -> None:
        self._key = None
        if self.contact is not None:
            self.contact.invalidate()

    def update(self, state: Any, free: torch.Tensor) -> None:
        self.delegate.update(state, free)
        self.invalidate()

    def _prepare(self, state: Any) -> None:
        key = (id(state), id(state.u), state.u._version)  # noqa: SLF001
        if key == self._key:
            return
        assert state.u.is_cuda
        from ._assembled_fem import AssembledFemHvp
        from ._gpu_sparse import GpuFreeSparseHessian

        started = time.perf_counter()
        collision = self.model.collision
        assert collision is not None
        assert state.collision is not None
        if state.collision.hess is None:
            positions = (collision.vertices + state.u[collision.indices]).numpy(
                force=True
            )
            state.collision.hess = collision.potential.hessian(
                collisions=state.collision.collisions,
                mesh=collision.collision_mesh,
                X=positions,
            )
        if self.fem is None:
            self.fem = AssembledFemHvp(self.model, state)
        if self.sparse is None:
            self.sparse = GpuFreeSparseHessian(self.model, state, self.fem, shift=0.0)
        else:
            self.sparse.setup(state, self.fem, shift=0.0)
        torch.cuda.synchronize(state.u.device)
        elapsed = time.perf_counter() - started
        self.setup_seconds += elapsed
        if not self.refreshes:
            self.first_setup_seconds = elapsed
        self.refreshes += 1
        self.symbolic_rebuilds += int(self.sparse.metadata["symbolic_rebuilt"])
        self.contact_remaps += int(self.sparse.metadata["contact_pattern_changed"])
        self._key = key

    def hess_diag(self, state: Any) -> torch.Tensor:
        diagonal = self.delegate.hess_diag(state)
        if self.backend == "gpu_sparse":
            self._prepare(state)
        return diagonal

    def hess_prod(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        self.products += 1
        if self.backend == "matrix_free":
            return self.delegate.hess_prod(state, direction)
        if self.backend == "gpu_sparse":
            self._prepare(state)
            return torch.sparse.mm(self.sparse.matrix, direction[:, None])[:, 0]
        assert self.contact is not None
        full = self.model.dof_map.to_full_grad(direction)
        output = torch.zeros_like(full)
        self.model.warp_model.hess_prod(state.u, full, output)
        self.contact.hess_prod(state.collision, state.u, full, output)
        return self.model.dof_map.to_free_grad(output)

    def hess_quad(self, state: Any, direction: torch.Tensor) -> torch.Tensor:
        return torch.dot(direction, self.hess_prod(state, direction))

    def report(self) -> dict:
        return {
            "backend": self.backend,
            "hess_products": self.products,
            "physical_refreshes": self.refreshes,
            "setup_seconds": self.setup_seconds,
            "first_setup_seconds": self.first_setup_seconds,
            "symbolic_rebuilds": self.symbolic_rebuilds,
            "contact_pattern_changes": self.contact_remaps,
            "persistent_bytes": (
                0
                if self.sparse is None
                else self.sparse.persistent_bytes + self.fem.persistent_bytes
            )
            + (0 if self.contact is None else self.contact.persistent_bytes),
            "contact_uploads": 0 if self.contact is None else self.contact.uploads,
            "shift_in_assembled_matrix": 0.0,
        }
