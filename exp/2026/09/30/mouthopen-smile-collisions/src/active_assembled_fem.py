# ruff: noqa: EM101, TRY003
"""Experiment-local assembled active-strain FEM Hessian for contact checks.

This extends the generic tetrahedral bulk assembly already used by the core
backend. It deliberately leaves the core registry and runtime sources intact.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from liblaf.apple.forward.hessian._assembled_fem import (
    _CELL_BATCH,
    AssembledFemHvp,
    _bulk_kernel,
    _cells_token,
    _topology,
)
from liblaf.apple.forward.hessian._gpu_sparse import GpuFreeSparseHessian
from liblaf.apple.forward.hessian._problem import HessianProblem

_SUPPORTED = {"StableNeoHookean", "StableNeoHookeanActive"}


def _registry(model: Any) -> dict[str, Any]:
    wrapped = getattr(getattr(model, "warp_model", None), "__wrapped__", None)
    potentials = getattr(wrapped, "potentials", None)
    if not isinstance(potentials, dict) or not potentials:
        raise TypeError("requires the concrete Warp potential registry")
    unsupported = [
        f"{name}:{type(potential).__name__}"
        for name, potential in potentials.items()
        if type(potential).__name__ not in _SUPPORTED
    ]
    if unsupported:
        raise TypeError("unsupported potential(s): " + ", ".join(sorted(unsupported)))
    return potentials


class ActiveAssembledFemHvp(AssembledFemHvp):
    """Use the core generic ``hess_prod_func`` assembly for active strain."""

    def __init__(self, model: Any, state: Any) -> None:
        if int(model.dim) != 3 or state.u.ndim != 2 or state.u.shape[1] != 3:
            raise ValueError(
                "assembled FEM HVP requires a three-dimensional full state"
            )
        if not state.u.is_cuda:
            raise ValueError("assembled FEM HVP is GPU-only")
        potentials = _registry(model)
        topology_started = time.perf_counter()
        topology, topology_cache_hit = _topology(model, potentials)
        topology_seconds = time.perf_counter() - topology_started
        self._model = model
        self._potentials = potentials
        self._potential_objects = tuple(
            (name, id(potential), _cells_token(potential.cells))
            for name, potential in potentials.items()
        )
        self._topology = topology
        self._state_shape = tuple(state.u.shape)
        self._state_device = state.u.device
        self._state_dtype = state.u.dtype
        self._values = torch.zeros(
            (len(topology.col), 3, 3), device=state.u.device, dtype=state.u.dtype
        )
        self._kernels = {
            spec.name: _bulk_kernel(potentials[spec.name].hess_prod_func)
            for spec in topology.potentials
        }
        self._crow = torch.as_tensor(
            topology.crow, dtype=torch.int64, device=state.u.device
        )
        self._col = torch.as_tensor(
            topology.col, dtype=torch.int64, device=state.u.device
        )
        self.matrix = torch.sparse_bsr_tensor(
            self._crow,
            self._col,
            self._values,
            size=(3 * topology.n_points, 3 * topology.n_points),
            device=state.u.device,
            dtype=state.u.dtype,
        )
        self.n_points = topology.n_points
        self.persistent_bytes = (
            self._values.numel() * self._values.element_size()
            + self._crow.numel() * self._crow.element_size()
            + self._col.numel() * self._col.element_size()
        )
        self.metadata = {
            "method": "experiment_local_exact_active_strain_fem_sparse_bsr",
            "exact_physical_fem_hessian": True,
            "contact_excluded": True,
            "fixed_topology": True,
            "n_points": topology.n_points,
            "block_size": 3,
            "nnz_blocks": len(topology.col),
            "nnz_scalars_stored": int(self._values.numel()),
            "cell_batch_size": _CELL_BATCH,
            "potentials": [
                {"name": item.name, "kind": item.kind, "cells": len(item.cells)}
                for item in topology.potentials
            ],
            "topology_cache_hit": topology_cache_hit,
            "topology_seconds": topology_seconds,
            "persistent_bytes": self.persistent_bytes,
        }
        self.setup(state)

    def _check_topology(self) -> None:
        potentials = _registry(self._model)
        current = tuple(
            (name, id(potential), _cells_token(potential.cells))
            for name, potential in potentials.items()
        )
        if current != self._potential_objects:
            raise ValueError("FEM potential registry or mesh topology changed")


class ActiveSparseHessianProblem(HessianProblem):
    """Use :class:`ActiveAssembledFemHvp` in the existing exact sparse merge."""

    def __init__(self, delegate: Any) -> None:
        super().__init__(delegate, "gpu_sparse")

    def _prepare(self, state: Any) -> None:
        key = (id(state), id(state.u), state.u._version)  # noqa: SLF001
        if key == self._key:
            return
        assert state.u.is_cuda
        started = time.perf_counter()
        collision = self.model.collision
        assert collision is not None
        assert state.collision is not None
        if state.collision.hess is None:
            raise ValueError("prepare the contact Hessian before sparse assembly")
        if self.fem is None:
            self.fem = ActiveAssembledFemHvp(self.model, state)
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


__all__ = ["ActiveAssembledFemHvp", "ActiveSparseHessianProblem"]
