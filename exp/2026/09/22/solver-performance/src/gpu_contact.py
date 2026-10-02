# ruff: noqa: SLF001
"""Opt-in CUDA sparse application of an owned IPC contact Hessian.

The default collision implementation remains untouched.  This adapter assembles
the exact IPC Hessian on CPU once for each owned contact state, uploads its CSR
representation once, and uses CUDA sparse matvec for later products at that
same state.  Any contact update sets ``state.hess = None`` and therefore forces
fresh CPU assembly and a fresh upload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse
import torch


def _model(value: Any) -> Any:
    if hasattr(value, "collision") and hasattr(value, "hess_prod"):
        return value
    for path in (("runtime", "forward", "model"), ("forward", "model")):
        current = value
        for name in path:
            current = getattr(current, name, None)
        if current is not None and hasattr(current, "collision"):
            return current
    message = "expected a Model or an object owning runtime.forward.model"
    raise TypeError(message)


@dataclass
class GpuContactHessian:
    collision: Any
    original_hess_prod: Any
    _state: Any | None = None
    _hess: Any | None = None
    _device: torch.device | None = None
    _dtype: torch.dtype | None = None
    _matrix: torch.Tensor | None = None
    _indices: torch.Tensor | None = None
    uploads: int = 0
    products: int = 0

    def invalidate(self) -> None:
        self._state = self._hess = None
        self._device = self._dtype = self._matrix = self._indices = None

    def _assemble(self, state: Any, u: torch.Tensor) -> scipy.sparse.csr_matrix:
        vertices = (self.collision.vertices + u[self.collision.indices]).numpy(
            force=True
        )
        hess = self.collision.potential.hessian(
            collisions=state.collisions,
            mesh=self.collision.collision_mesh,
            X=vertices,
        ).tocsr()
        assert hess.shape[0] == hess.shape[1]
        assert np.isfinite(hess.data).all()
        state.hess = hess
        return hess

    def _upload(
        self, state: Any, u: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> None:
        hess = state.hess if state.hess is not None else self._assemble(state, u)
        assert isinstance(hess, scipy.sparse.spmatrix)
        csr = hess.tocsr()
        assert csr.shape == (self.collision.indices.numel() * 3,) * 2
        self._matrix = torch.sparse_csr_tensor(
            torch.as_tensor(csr.indptr, device=device),
            torch.as_tensor(csr.indices, device=device),
            torch.as_tensor(csr.data, device=device, dtype=dtype),
            size=csr.shape,
            device=device,
            dtype=dtype,
        )
        self._indices = self.collision.indices.to(device=device)
        self._state, self._hess = state, state.hess
        self._device, self._dtype = device, dtype
        self.uploads += 1

    def hess_prod(
        self, state: Any, u: torch.Tensor, p: torch.Tensor, output: torch.Tensor
    ) -> None:
        assert u.ndim == p.ndim == output.ndim == 2
        assert u.shape == p.shape == output.shape
        assert u.shape[-1] == 3
        assert u.dtype == p.dtype == output.dtype
        assert u.device == p.device == output.device
        # Preserve the reviewed CPU route, including its exact CPU CSR product.
        if not (p.is_cuda and output.is_cuda):
            self.original_hess_prod(state, u, p, output)
            return
        if (
            self._matrix is None
            or self._state is not state
            or self._hess is not state.hess
            or self._device != p.device
            or self._dtype != p.dtype
        ):
            self._upload(state, u, p.device, p.dtype)
        assert self._matrix is not None
        assert self._indices is not None
        assert self._indices.ndim == 1
        assert self._indices.numel() * 3 == self._matrix.shape[0]
        local = p[self._indices].reshape(-1, 1)
        product = torch.sparse.mm(self._matrix, local).reshape(-1, 3)
        output.index_add_(0, self._indices, product)
        self.products += 1


_ADAPTERS: list[tuple[Any, GpuContactHessian]] = []


def _lookup(collision: Any) -> GpuContactHessian:
    for registered, adapter in _ADAPTERS:
        if registered is collision:
            return adapter
    message = "GPU contact adapter is not registered for this collision"
    raise RuntimeError(message)


@dataclass
class InstalledGpuContact:
    adapter: GpuContactHessian
    collision: Any
    _original_class: type[Any] = field(repr=False)

    def uninstall(self) -> None:
        self.collision.__class__ = self._original_class
        _ADAPTERS[:] = [
            (item, value) for item, value in _ADAPTERS if item is not self.collision
        ]
        self.adapter.invalidate()


@dataclass
class InstalledAdjointGpuContact:
    """Install contact CUDA only for one implicit-adjoint linear solve."""

    solver: Any
    physics_or_model: Any
    _original_solve: Any = field(repr=False)
    uploads: int = 0
    products: int = 0

    def uninstall(self) -> None:
        self.solver.solve = self._original_solve


def install_adjoint_gpu_contact(
    runtime: Any, physics_or_model: Any
) -> InstalledAdjointGpuContact:
    """Wrap only ``runtime.solver.solve`` with temporary CUDA contact HVPs.

    The primal forward solver is never wrapped.  Cleanup is unconditional so a
    failed adjoint solve cannot leave the collision object's class patched.
    """
    solver = runtime.solver
    original_solve = solver.solve
    installed = InstalledAdjointGpuContact(solver, physics_or_model, original_solve)

    def solve(*args: Any, **kwargs: Any) -> Any:
        handle = install_gpu_contact(installed.physics_or_model)
        try:
            return installed._original_solve(*args, **kwargs)
        finally:
            installed.uploads += handle.adapter.uploads
            installed.products += handle.adapter.products
            handle.uninstall()

    solver.solve = solve
    return installed


def install_gpu_contact(physics_or_model: Any) -> InstalledGpuContact:
    """Install an opt-in slotted collision subclass on one model.

    The dynamically derived class has no additional layout, so it remains
    compatible with attrs-slotted ``OwnedContact`` while retaining the exact
    original object and all its owned state.
    """
    model = _model(physics_or_model)
    collision = model.collision
    assert collision is not None
    original_class = type(collision)
    original = collision.hess_prod
    adapter = GpuContactHessian(collision, original)

    def hess_prod(
        self: Any, state: Any, u: torch.Tensor, p: torch.Tensor, output: torch.Tensor
    ) -> None:
        _lookup(self).hess_prod(state, u, p, output)

    patched = type(
        f"Gpu{original_class.__name__}",
        (original_class,),
        {"__slots__": (), "hess_prod": hess_prod},
    )
    collision.__class__ = patched
    _ADAPTERS.append((collision, adapter))
    return InstalledGpuContact(adapter, collision, original_class)
