# ruff: noqa: EM101, EM102, TRY003
"""Exact reduced vertex-block Hessian preconditioner for the joint model.

Blocks come from existing exact Hessian products. Eigenvalue flooring applies
only to this inverse preconditioner, never to the physical Hessian.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

_SUPPORTED_POTENTIALS = {"StableNeoHookeanStress", "StableNeoHookeanMembrane"}
_EIGH_BATCH_SIZE = 4096


@dataclass
class _Topology:
    adjacency: tuple[frozenset[int], ...]
    fem_edges: int
    colours_by_free: dict[bytes, tuple[np.ndarray, ...]] = field(default_factory=dict)


_TOPOLOGIES: dict[int, tuple[weakref.ReferenceType[Any], _Topology]] = {}


@dataclass(frozen=True)
class VertexBlockPreconditioner:
    """Batched inverse 3-by-3 blocks and a reduced-coordinate lookup."""

    inverse_blocks: torch.Tensor
    free_indices: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    free_mask: torch.Tensor
    relative_eigenvalue_floor: float
    absolute_eigenvalue_floor: float
    setup_metadata: dict[str, Any]

    def apply(self, free_tensor: torch.Tensor) -> torch.Tensor:
        if free_tensor.ndim != 1:
            raise ValueError(
                "vertex-block preconditioner expects a rank-one free vector"
            )
        if free_tensor.numel() != self.setup_metadata["n_free"]:
            raise ValueError("vertex-block preconditioner free-vector size differs")
        free_indices = self.free_indices.to(free_tensor.device)
        local = torch.zeros(
            (self.inverse_blocks.shape[0], 3),
            device=free_tensor.device,
            dtype=free_tensor.dtype,
        )
        local.flatten()[free_indices] = free_tensor
        solved = torch.bmm(
            self.inverse_blocks.to(device=free_tensor.device, dtype=free_tensor.dtype),
            local.unsqueeze(-1),
        ).squeeze(-1)
        return solved.flatten()[free_indices]

    def with_shift(self, shift: float) -> VertexBlockPreconditioner:
        """Return the preconditioner for ``H + shift I`` without new HVPs."""
        if not np.isfinite(shift) or shift < 0:
            raise ValueError("preconditioner shift must be finite and non-negative")
        # The joint model constrains whole vertices. Partial vertex constraints
        # would shift a coordinate subspace and require a new eigendecomposition.
        whole_vertex = torch.all(self.free_mask == self.free_mask[:, :1])
        if not bool(whole_vertex):
            raise ValueError("shifted vertex blocks require whole-vertex constraints")
        shifted = (
            self.eigenvalues + self.free_mask[:, :1].to(self.eigenvalues.dtype) * shift
        )
        floors = torch.clamp(
            shifted.abs().amax(dim=1, keepdim=True) * self.relative_eigenvalue_floor,
            min=self.absolute_eigenvalue_floor,
        )
        inverse = (
            self.eigenvectors * shifted.clamp_min(floors).reciprocal().unsqueeze(1)
        ) @ self.eigenvectors.transpose(1, 2)
        metadata = {
            **self.setup_metadata,
            "shift": {
                "value": shift,
                "physical_hessian_unchanged": True,
                "additional_exact_hessian_product_calls": 0,
                "additional_eigendecompositions": 0,
                "batched_inverse_recomputations": 1,
            },
            "regularization": {
                **self.setup_metadata["regularization"],
                "shift": shift,
                "floored_modes": int((shifted < floors).sum()),
                "minimum_unfloored_eigenvalue": float(shifted.min()),
            },
        }
        return VertexBlockPreconditioner(
            inverse_blocks=inverse,
            free_indices=self.free_indices,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            free_mask=self.free_mask,
            relative_eigenvalue_floor=self.relative_eigenvalue_floor,
            absolute_eigenvalue_floor=self.absolute_eigenvalue_floor,
            setup_metadata=metadata,
        )


def _as_cpu_cells(potential: Any) -> np.ndarray:
    cells = getattr(potential, "cells", None)
    if cells is None:
        raise TypeError(f"{type(potential).__name__} has no cell connectivity")
    if torch.is_tensor(cells):
        result = cells.detach().cpu().numpy()
    else:
        try:
            import warp as wp

            result = wp.to_torch(cells).detach().cpu().numpy()
        except Exception as error:  # pragma: no cover - requires Warp runtime
            raise TypeError(
                "cannot read supported potential cell connectivity"
            ) from error
    if result.ndim != 2 or result.shape[1] not in (3, 4):
        raise ValueError("supported potential cells must be triangles or tetrahedra")
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError("supported potential cells must use integer vertex IDs")
    return result


def _add_cliques(adjacency: list[set[int]], cells: np.ndarray, n_points: int) -> int:
    edges = 0
    for cell in cells:
        vertices = [int(vertex) for vertex in cell]
        if min(vertices) < 0 or max(vertices) >= n_points:
            raise ValueError("potential cell references a vertex outside the model")
        for index, left in enumerate(vertices):
            for right in vertices[index + 1 :]:
                if right not in adjacency[left]:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
                    edges += 1
    return edges


def _base_topology(model: Any, potentials: dict[str, Any]) -> tuple[_Topology, bool]:
    model_id = id(model)
    cached = _TOPOLOGIES.get(model_id)
    if cached is not None and cached[0]() is model:
        return cached[1], True
    adjacency = [set() for _ in range(int(model.n_points))]
    fem_edges = sum(
        _add_cliques(adjacency, _as_cpu_cells(item), int(model.n_points))
        for item in potentials.values()
    )
    topology = _Topology(tuple(frozenset(row) for row in adjacency), fem_edges)

    def _cleanup(
        _reference: weakref.ReferenceType[Any], *, key: int = model_id
    ) -> None:
        _TOPOLOGIES.pop(key, None)

    _TOPOLOGIES[model_id] = (weakref.ref(model, _cleanup), topology)
    return topology, False


def _contact_adjacency(
    model: Any, state: Any, adjacency: list[set[int]]
) -> tuple[int, int]:
    collision = model.collision
    if collision is None:
        return 0, 0
    if state.collision is None:
        raise ValueError("contact model requires a rebuilt collision state")
    if state.collision.hess is None:
        vertices = collision.vertices + state.u[collision.indices]
        state.collision.hess = collision.potential.hessian(
            collisions=state.collision.collisions,
            mesh=collision.collision_mesh,
            X=vertices.detach().cpu().numpy(),
        )
    matrix = state.collision.hess.tocoo()
    local_count = int(collision.indices.numel())
    if matrix.shape != (3 * local_count, 3 * local_count):
        raise ValueError("IPC Hessian shape differs from collision vertex map")
    global_vertices = collision.indices.detach().cpu().numpy()
    edges = 0
    for row, column in zip(matrix.row, matrix.col, strict=True):
        left, right = (
            int(global_vertices[int(row) // 3]),
            int(global_vertices[int(column) // 3]),
        )
        if left != right and right not in adjacency[left]:
            adjacency[left].add(right)
            adjacency[right].add(left)
            edges += 1
    return edges, int(matrix.nnz)


def _colour(adjacency: list[set[int]], active: np.ndarray) -> tuple[np.ndarray, ...]:
    colours: dict[int, int] = {}
    for vertex in sorted(
        active.tolist(), key=lambda item: (-len(adjacency[item]), item)
    ):
        used = {colours[other] for other in adjacency[vertex] if other in colours}
        colours[vertex] = next(
            index for index in range(len(used) + 1) if index not in used
        )
    groups: defaultdict[int, list[int]] = defaultdict(list)
    for vertex, colour in colours.items():
        groups[colour].append(vertex)
    return tuple(
        np.asarray(groups[index], dtype=np.int64) for index in range(len(groups))
    )


def _free_lookup(model: Any) -> tuple[np.ndarray, np.ndarray]:
    free_indices = model.dof_map.free_indices.detach().cpu().numpy().astype(np.int64)
    lookup = np.full(int(model.n_points) * 3, -1, dtype=np.int64)
    lookup[free_indices] = np.arange(len(free_indices), dtype=np.int64)
    return lookup.reshape(-1, 3), free_indices


def _probe_blocks(
    model: Any, state: Any, colours: tuple[np.ndarray, ...], lookup: np.ndarray
) -> tuple[torch.Tensor, int]:
    blocks = torch.zeros(
        (int(model.n_points), 3, 3), dtype=state.u.dtype, device=state.u.device
    )
    probes = 0
    for group in colours:
        group_t = torch.as_tensor(group, device=state.u.device)
        for coordinate in range(3):
            selected_mask = lookup[group, coordinate] >= 0
            if not np.any(selected_mask):
                continue
            direction = torch.zeros_like(state.u)
            direction[
                group_t[torch.as_tensor(selected_mask, device=state.u.device)],
                coordinate,
            ] = 1.0
            product = model.hess_prod(state, direction)
            selected = group_t[torch.as_tensor(selected_mask, device=state.u.device)]
            blocks[selected, :, coordinate] = product[selected]
            probes += 1
    return blocks, probes


def _inverse_blocks(
    blocks: torch.Tensor, lookup: np.ndarray, relative: float, absolute: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, float]:
    free = torch.as_tensor(lookup >= 0, device=blocks.device)
    block_mask = free.unsqueeze(1) & free.unsqueeze(2)
    blocks = blocks * block_mask + torch.diag_embed((~free).to(blocks.dtype))
    symmetric = 0.5 * (blocks + blocks.transpose(1, 2))
    values = torch.empty(blocks.shape[:2], dtype=blocks.dtype, device=blocks.device)
    vectors = torch.empty_like(blocks)
    inverse = torch.empty_like(blocks)
    floored = 0
    minimum = float("inf")
    # Full-face models contain hundreds of thousands of independent 3-by-3
    # blocks. CUDA's batched eigensolver reserves enormous hidden workspace for
    # that full batch, so factor bounded independent batches instead.
    for start in range(0, len(blocks), _EIGH_BATCH_SIZE):
        stop = min(start + _EIGH_BATCH_SIZE, len(blocks))
        batch_values, batch_vectors = torch.linalg.eigh(symmetric[start:stop])
        floors = torch.clamp(
            batch_values.abs().amax(dim=1, keepdim=True) * relative,
            min=absolute,
        )
        values[start:stop] = batch_values
        vectors[start:stop] = batch_vectors
        inverse[start:stop] = (
            batch_vectors * batch_values.clamp_min(floors).reciprocal().unsqueeze(1)
        ) @ batch_vectors.transpose(1, 2)
        floored += int((batch_values < floors).sum())
        minimum = min(minimum, float(batch_values.min()))
    return inverse, values, vectors, free, floored, minimum


def build_vertex_preconditioner(
    model: Any,
    state: Any,
    *,
    relative_eigenvalue_floor: float = 1.0e-8,
    absolute_eigenvalue_floor: float = 1.0e-12,
) -> VertexBlockPreconditioner:
    """Build exact blocks using three HVPs per vertex-graph colour.

    FEM topology and the collision-free colouring are cached per model. IPC
    sparse Hessian structure is rebuilt and recoloured for every contact state.
    """
    if not 0 < relative_eigenvalue_floor < 1 or absolute_eigenvalue_floor <= 0:
        raise ValueError("invalid preconditioner eigenvalue floors")
    wrapped = getattr(getattr(model, "warp_model", None), "__wrapped__", None)
    potentials = getattr(wrapped, "potentials", None)
    if not isinstance(potentials, dict) or not potentials:
        raise TypeError("requires the joint model's concrete Warp potential registry")
    unsupported = sorted(
        name
        for name, item in potentials.items()
        if type(item).__name__ not in _SUPPORTED_POTENTIALS
    )
    if unsupported:
        raise TypeError(
            f"unsupported potentials for vertex blocks: {', '.join(unsupported)}"
        )
    if int(model.dim) != 3:
        raise ValueError("vertex blocks require a three-dimensional model")
    topology, topology_cache_hit = _base_topology(model, potentials)
    lookup, free_indices = _free_lookup(model)
    active = np.flatnonzero(np.any(lookup >= 0, axis=1))
    if not len(active):
        raise ValueError("model has no free degrees of freedom")
    if model.collision is None:
        key = free_indices.tobytes()
        colours = topology.colours_by_free.get(key)
        colour_cache_hit = colours is not None
        if colours is None:
            colours = _colour([set(row) for row in topology.adjacency], active)
            topology.colours_by_free[key] = colours
        contact_edges, contact_nnz = 0, 0
    else:
        adjacency = [set(row) for row in topology.adjacency]
        contact_edges, contact_nnz = _contact_adjacency(model, state, adjacency)
        colours = _colour(adjacency, active)
        colour_cache_hit = False
    blocks, probes = _probe_blocks(model, state, colours, lookup)
    inverse, values, vectors, free_mask, floored, minimum = _inverse_blocks(
        blocks, lookup, relative_eigenvalue_floor, absolute_eigenvalue_floor
    )
    return VertexBlockPreconditioner(
        inverse_blocks=inverse,
        free_indices=torch.as_tensor(
            free_indices, dtype=torch.long, device=state.u.device
        ),
        eigenvalues=values,
        eigenvectors=vectors,
        free_mask=free_mask,
        relative_eigenvalue_floor=relative_eigenvalue_floor,
        absolute_eigenvalue_floor=absolute_eigenvalue_floor,
        setup_metadata={
            "method": "exact_hessian_product_vertex_blocks",
            "exact_hessian_unchanged": True,
            "n_free": len(free_indices),
            "active_vertices": len(active),
            "colour_count": len(colours),
            "exact_hessian_product_calls": probes,
            "eigenvalue_factorization_batch_size": _EIGH_BATCH_SIZE,
            "fem_graph_edges": topology.fem_edges,
            "ipc_graph_edges": contact_edges,
            "ipc_hessian_nnz": contact_nnz,
            "static_topology_cache_hit": topology_cache_hit,
            "collision_free_colour_cache_hit": colour_cache_hit,
            "shift": {
                "value": 0.0,
                "physical_hessian_unchanged": True,
                "additional_exact_hessian_product_calls": 0,
                "additional_eigendecompositions": 0,
                "batched_inverse_recomputations": 0,
            },
            "regularization": {
                "kind": "eigenvalue_floor_for_preconditioner_only",
                "relative_floor": relative_eigenvalue_floor,
                "absolute_floor": absolute_eigenvalue_floor,
                "floored_modes": floored,
                "minimum_unfloored_eigenvalue": minimum,
            },
        },
    )
