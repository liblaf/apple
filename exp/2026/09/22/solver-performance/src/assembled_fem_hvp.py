# ruff: noqa: EM101, SLF001, TRY003
"""Frozen exact sparse FEM Hessian products for the joint material registry.

This deliberately assembles only the Warp FEM potentials.  IPC/contact is not
included: callers use this operator only when all compared arms share contact
and need to separate the FEM contribution.  Assembly evaluates the existing
element Hessian-product algebra on each local Cartesian basis vector and sums
the resulting 3-by-3 vertex blocks into a GPU BSR matrix.  No dense global
matrix is formed.
"""

from __future__ import annotations

import sys
import time
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from liblaf.apple.warp.fem import func

_SUPPORTED = {
    "StableNeoHookeanStress",
    "StableNeoHookeanActive",
    "StableNeoHookeanMembrane",
    "StableNeoHookeanActiveMembrane",
}
_CELL_BATCH = 4096
_TOPOLOGIES: dict[int, tuple[weakref.ReferenceType[Any], _Topology]] = {}

floating = Any
vec3 = Any
mat43 = Any
Materials = Any


@dataclass(frozen=True)
class _PotentialTopology:
    cells: np.ndarray
    slots: np.ndarray
    vertices_per_cell: int
    name: str
    kind: str


@dataclass(frozen=True)
class _Topology:
    n_points: int
    keys: np.ndarray
    crow: np.ndarray
    col: np.ndarray
    potentials: tuple[_PotentialTopology, ...]


def _cpu_cells(potential: Any) -> np.ndarray:
    cells = potential.cells
    if torch.is_tensor(cells):
        result = cells.detach().cpu().numpy()
    else:
        result = wp.to_torch(cells).detach().cpu().numpy()
    if result.ndim != 2 or result.shape[1] not in (3, 4):
        raise ValueError("FEM potential cells must be rank-two triangles or tetrahedra")
    return np.asarray(result, dtype=np.int64)


def _registry(model: Any) -> dict[str, Any]:
    wrapped = getattr(getattr(model, "warp_model", None), "__wrapped__", None)
    potentials = getattr(wrapped, "potentials", None)
    if not isinstance(potentials, dict) or not potentials:
        raise TypeError("requires the concrete joint Warp potential registry")
    unsupported = [
        f"{name}:{type(potential).__name__}"
        for name, potential in potentials.items()
        if type(potential).__name__ not in _SUPPORTED
    ]
    if unsupported:
        raise TypeError("unsupported potential(s): " + ", ".join(sorted(unsupported)))
    return potentials


def _topology(model: Any, potentials: dict[str, Any]) -> tuple[_Topology, bool]:
    model_id = id(model)
    cached = _TOPOLOGIES.get(model_id)
    if cached is not None and cached[0]() is model:
        return cached[1], True
    n_points = int(model.n_points)
    raw: list[tuple[str, str, np.ndarray]] = []
    all_keys: list[np.ndarray] = []
    for name, potential in potentials.items():
        cells = _cpu_cells(potential)
        if cells.size and (cells.min() < 0 or cells.max() >= n_points):
            raise ValueError("potential cell indexes a point outside the model")
        vertices = cells.shape[1]
        rows = np.repeat(cells, vertices, axis=1)
        cols = np.tile(cells, (1, vertices))
        all_keys.append((rows * n_points + cols).reshape(-1))
        raw.append((name, type(potential).__name__, cells))
    keys = np.unique(np.concatenate(all_keys))
    rows = keys // n_points
    col = (keys % n_points).astype(np.int64, copy=False)
    counts = np.bincount(rows, minlength=n_points)
    crow = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts))).astype(
        np.int64, copy=False
    )
    potential_topologies: list[_PotentialTopology] = []
    for name, kind, cells in raw:
        vertices = cells.shape[1]
        local_keys = np.repeat(cells, vertices, axis=1) * n_points + np.tile(
            cells, (1, vertices)
        )
        slots = np.searchsorted(keys, local_keys)
        if not np.array_equal(keys[slots], local_keys):  # defensive topology contract
            raise AssertionError("local FEM block missing from global sparse topology")
        potential_topologies.append(
            _PotentialTopology(cells, slots, vertices, name, kind)
        )
    result = _Topology(n_points, keys, crow, col, tuple(potential_topologies))

    def cleanup(_reference: weakref.ReferenceType[Any], *, key: int = model_id) -> None:
        _TOPOLOGIES.pop(key, None)

    _TOPOLOGIES[model_id] = (weakref.ref(model, cleanup), result)
    return result, False


def _bulk_kernel(hess_prod_func: Any) -> wp.Kernel:
    """Create a per-tetrahedron local-Hessian kernel from exact FEM algebra."""

    @wp.kernel(module="unique")
    def kernel(
        u: wp.array1d[vec3],
        cells: wp.array1d[wp.vec4i],
        materials: Materials,
        start: int,
        output: wp.array3d[floating],
    ) -> None:
        local_cell, column = wp.tid()
        cid = start + local_cell
        cell = cells[cid]
        u_cell = func.get_cell_displacements(u, cell)
        basis = wp.matrix(shape=(4, 3), dtype=materials.dV.dtype)
        basis[column // 3, column % 3] = materials.dV.dtype(1.0)
        product = wp.matrix(shape=(4, 3), dtype=materials.dV.dtype)
        # Keep the potential's quadrature sum exact; the joint tetrahedra
        # currently use one point, but this operator must not silently depend
        # on that implementation detail.
        for qid in range(materials.dhdX.shape[1]):
            F = func.deformation_gradient(u_cell, materials.dhdX[cid, qid])
            product += materials.dV[cid, qid] * hess_prod_func(
                F, basis, materials.dhdX[cid, qid], materials, cid
            )
        for vertex in range(4):
            for coordinate in range(3):
                output[local_cell, 3 * vertex + coordinate, column] = product[
                    vertex, coordinate
                ]

    return kernel


def _membrane_kernel(module: Any, *, active_strain: bool = False) -> wp.Kernel:
    """Create the matching local basis kernel for joint_materials membranes."""
    metric = module._metric
    metric_gradient = module.membrane_metric_gradient
    metric_hessian = module.membrane_metric_hessian
    area_weight = module._area_weight

    @wp.kernel(module="unique")
    def kernel(
        u: wp.array1d[vec3],
        cells: wp.array1d[wp.vec3i],
        materials: Materials,
        start: int,
        output: wp.array3d[floating],
    ) -> None:
        local_cell, column = wp.tid()
        cid = start + local_cell
        cell = cells[cid]
        a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
        b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
        p0 = wp.vector(
            materials.fraction.dtype(0.0),
            materials.fraction.dtype(0.0),
            materials.fraction.dtype(0.0),
        )
        p1 = p0
        p2 = p0
        coordinate = column % 3
        if column // 3 == 0:
            p0[coordinate] = materials.fraction.dtype(1.0)
        elif column // 3 == 1:
            p1[coordinate] = materials.fraction.dtype(1.0)
        else:
            p2[coordinate] = materials.fraction.dtype(1.0)
        p_a = p1 - p0
        p_b = p2 - p0
        g = metric(a, b)
        if wp.static(active_strain):
            w = metric_gradient(
                g,
                materials.metric_map[cid],
                materials.activation_inv[cid],
                materials.lmbda[cid],
                materials.mu[cid],
                materials.thickness[cid],
            )
        else:
            w = metric_gradient(
                g,
                materials.metric_map[cid],
                materials.baseline_stress[cid],
                materials.lmbda[cid],
                materials.mu[cid],
                materials.thickness[cid],
            )
        H = metric_hessian(
            g,
            materials.metric_map[cid],
            materials.lmbda[cid],
            materials.mu[cid],
            materials.thickness[cid],
        )
        dg = wp.vector(
            materials.fraction.dtype(2.0) * wp.dot(a, p_a),
            wp.dot(p_a, b) + wp.dot(a, p_b),
            materials.fraction.dtype(2.0) * wp.dot(b, p_b),
        )
        dw = H @ dg
        hp_a = (
            materials.fraction.dtype(2.0) * dw[0] * a
            + materials.fraction.dtype(2.0) * w[0] * p_a
            + dw[1] * b
            + w[1] * p_b
        )
        hp_b = (
            dw[1] * a
            + w[1] * p_a
            + materials.fraction.dtype(2.0) * dw[2] * b
            + materials.fraction.dtype(2.0) * w[2] * p_b
        )
        weight = area_weight(
            materials.fraction[cid], materials.rest_metric_sqrt_det[cid]
        )
        hp0 = -weight * (hp_a + hp_b)
        hp1 = weight * hp_a
        hp2 = weight * hp_b
        for component in range(3):
            output[local_cell, component, column] = hp0[component]
            output[local_cell, 3 + component, column] = hp1[component]
            output[local_cell, 6 + component, column] = hp2[component]

    return kernel


def _wp_u(u: torch.Tensor) -> wp.array:
    scalar = wp.dtype_from_torch(u.dtype)
    return wp.from_torch(u, dtype=wp.types.vector(3, scalar))


class AssembledFemHvp:
    """Exact frozen FEM Hessian action stored as a GPU block sparse matrix."""

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
        self._topology = topology
        self._state_shape = tuple(state.u.shape)
        self._state_device = state.u.device
        self._state_dtype = state.u.dtype
        self._values = torch.zeros(
            (len(topology.col), 3, 3), device=state.u.device, dtype=state.u.dtype
        )
        self._kernels: dict[str, wp.Kernel] = {}
        for spec in topology.potentials:
            potential = potentials[spec.name]
            if spec.kind in {"StableNeoHookeanStress", "StableNeoHookeanActive"}:
                self._kernels[spec.name] = _bulk_kernel(potential.hess_prod_func)
            else:
                module = sys.modules.get(type(potential).__module__)
                if module is None:
                    raise RuntimeError("membrane potential module is not loaded")
                self._kernels[spec.name] = _membrane_kernel(
                    module,
                    active_strain=spec.kind == "StableNeoHookeanActiveMembrane",
                )
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
            "method": "exact_frozen_fem_sparse_bsr",
            "exact_physical_fem_hessian": True,
            "contact_excluded": True,
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

    def setup(self, state: Any) -> None:
        """Refresh numerical FEM blocks at a state with unchanged topology."""
        if tuple(state.u.shape) != self._state_shape:
            raise ValueError("state shape differs from assembled FEM topology")
        if state.u.device != self._state_device or state.u.dtype != self._state_dtype:
            raise ValueError("state device or dtype differs from assembled FEM matrix")
        assembly_started = time.perf_counter()
        self._values.zero_()
        u = _wp_u(state.u)
        stream = wp.stream_from_torch(torch.cuda.current_stream(state.u.device))
        with wp.ScopedStream(stream):
            for spec in self._topology.potentials:
                potential = self._potentials[spec.name]
                kernel = self._kernels[spec.name]
                for start in range(0, len(spec.cells), _CELL_BATCH):
                    stop = min(start + _CELL_BATCH, len(spec.cells))
                    local = torch.empty(
                        (
                            stop - start,
                            3 * spec.vertices_per_cell,
                            3 * spec.vertices_per_cell,
                        ),
                        device=state.u.device,
                        dtype=state.u.dtype,
                    )
                    local_wp = wp.from_torch(
                        local, dtype=wp.dtype_from_torch(local.dtype)
                    )
                    wp.launch(
                        kernel,
                        dim=(stop - start, 3 * spec.vertices_per_cell),
                        inputs=[u, potential.cells, potential.materials, start],
                        outputs=[local_wp],
                        device=potential.cells.device,
                    )
                    blocks = (
                        local.reshape(
                            stop - start,
                            spec.vertices_per_cell,
                            3,
                            spec.vertices_per_cell,
                            3,
                        )
                        .permute(0, 1, 3, 2, 4)
                        .reshape(-1, 3, 3)
                    )
                    slots = torch.as_tensor(
                        spec.slots[start:stop].reshape(-1),
                        device=state.u.device,
                        dtype=torch.long,
                    )
                    self._values.index_add_(0, slots, blocks)
        # Build a fresh sparse view so setup never relies on undocumented
        # aliasing between a BSR tensor and its input values tensor.
        self.matrix = torch.sparse_bsr_tensor(
            self._crow,
            self._col,
            self._values,
            size=(3 * self.n_points, 3 * self.n_points),
            device=state.u.device,
            dtype=state.u.dtype,
        )
        self.metadata["numeric_setup_seconds"] = time.perf_counter() - assembly_started

    def apply(self, direction_full: torch.Tensor) -> torch.Tensor:
        if direction_full.shape != (self.n_points, 3):
            raise ValueError(
                "direction shape differs from the assembled full FEM state"
            )
        if direction_full.device != self.matrix.device:
            raise ValueError("direction device differs from assembled FEM matrix")
        result = torch.sparse.mm(self.matrix, direction_full.reshape(-1, 1))
        return result.reshape(self.n_points, 3)
