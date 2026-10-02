# ruff: noqa: SLF001
"""Frozen-state exact FEM Hessian products for the joint material model.

This module deliberately owns only the two experiment-local FEM potentials.  It
does not include IPC/contact curvature and does not change the constitutive
law: ``setup`` evaluates state-dependent quantities once and ``apply`` uses
the same analytical Hessian product as :mod:`joint_materials`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
from joint_materials import (
    StableNeoHookeanMembrane,
    StableNeoHookeanStress,
    membrane_metric_gradient,
    membrane_metric_hessian,
)

from liblaf.apple.warp.fem import func

_FLOAT = Any
_MAT33 = Any
_MAT43 = Any
_VEC3 = Any
_VEC3I = Any


def _vec3(tensor: torch.Tensor) -> wp.array:
    return wp.from_torch(
        tensor, dtype=wp.types.vector(3, wp.dtype_from_torch(tensor.dtype))
    )


def _array(tensor: torch.Tensor, dtype: Any) -> wp.array:
    return wp.from_torch(tensor, dtype=dtype)


@wp.kernel(module="unique")
def _bulk_invariants(
    F: wp.array2d[_MAT33],
    J: wp.array2d[_FLOAT],
    cofactor: wp.array2d[_MAT33],
) -> None:
    cid, qid = wp.tid()
    J[cid, qid] = func.I3(F[cid, qid])
    cofactor[cid, qid] = func.g3(F[cid, qid])


@wp.kernel(module="unique")
def _bulk_apply(
    direction: wp.array1d[_VEC3],
    cells: wp.array1d[wp.vec4i],
    dhdX: wp.array2d[_MAT43],
    dV: wp.array2d[_FLOAT],
    lmbda: wp.array1d[_FLOAT],
    mu: wp.array1d[_FLOAT],
    active_stress: wp.array1d[_MAT33],
    F: wp.array2d[_MAT33],
    J: wp.array2d[_FLOAT],
    cofactor: wp.array2d[_MAT33],
    output: wp.array1d[_VEC3],
) -> None:
    cid, qid = wp.tid()
    cell = cells[cid]
    p = func.get_cell_displacements(direction, cell)
    dhdX_cell = dhdX[cid, qid]
    F_cell = F[cid, qid]
    passive = (
        lmbda[cid] * func.h3_prod(p, dhdX_cell, cofactor[cid, qid])
        + F_cell.dtype(0.5) * mu[cid] * func.h5_prod(p, dhdX_cell)
        + (-mu[cid] + lmbda[cid] * (J[cid, qid] - F_cell.dtype(1.0)))
        * func.h6_prod(p, dhdX_cell, F_cell)
    )
    dF = func.deformation_gradient_jvp(dhdX_cell, p)
    product = dV[cid, qid] * (
        passive + func.deformation_gradient_vjp(dhdX_cell, dF @ active_stress[cid])
    )
    for index in range(4):
        wp.atomic_add(output, cell[index], product[index])


@wp.func
def _metric(a: _VEC3, b: _VEC3) -> _VEC3:
    return wp.vector(wp.dot(a, a), wp.dot(a, b), wp.dot(b, b))


@wp.func
def _area_weight(fraction: _FLOAT, rest_metric_sqrt_det: _FLOAT) -> _FLOAT:
    return fraction * rest_metric_sqrt_det / fraction.dtype(2.0)


@wp.kernel(module="unique")
def _membrane_setup(
    displacement: wp.array1d[_VEC3],
    cells: wp.array1d[_VEC3I],
    rest_edge_01: wp.array1d[_VEC3],
    rest_edge_02: wp.array1d[_VEC3],
    metric_map: wp.array1d[_MAT33],
    baseline_stress: wp.array1d[Any],
    lmbda: wp.array1d[_FLOAT],
    mu: wp.array1d[_FLOAT],
    thickness: wp.array1d[_FLOAT],
    fraction: wp.array1d[_FLOAT],
    rest_metric_sqrt_det: wp.array1d[_FLOAT],
    a_cache: wp.array1d[_VEC3],
    b_cache: wp.array1d[_VEC3],
    w_cache: wp.array1d[_VEC3],
    H_cache: wp.array1d[_MAT33],
    weight_cache: wp.array1d[_FLOAT],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = rest_edge_01[cid] + displacement[cell[1]] - displacement[cell[0]]
    b = rest_edge_02[cid] + displacement[cell[2]] - displacement[cell[0]]
    g = _metric(a, b)
    a_cache[cid] = a
    b_cache[cid] = b
    w_cache[cid] = membrane_metric_gradient(
        g, metric_map[cid], baseline_stress[cid], lmbda[cid], mu[cid], thickness[cid]
    )
    H_cache[cid] = membrane_metric_hessian(
        g, metric_map[cid], lmbda[cid], mu[cid], thickness[cid]
    )
    weight_cache[cid] = _area_weight(fraction[cid], rest_metric_sqrt_det[cid])


@wp.kernel(module="unique")
def _membrane_apply(
    direction: wp.array1d[_VEC3],
    cells: wp.array1d[_VEC3I],
    a_cache: wp.array1d[_VEC3],
    b_cache: wp.array1d[_VEC3],
    w_cache: wp.array1d[_VEC3],
    H_cache: wp.array1d[_MAT33],
    weight_cache: wp.array1d[_FLOAT],
    output: wp.array1d[_VEC3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a, b, w, H = a_cache[cid], b_cache[cid], w_cache[cid], H_cache[cid]
    p_a = direction[cell[1]] - direction[cell[0]]
    p_b = direction[cell[2]] - direction[cell[0]]
    dg = wp.vector(
        a.dtype(2.0) * wp.dot(a, p_a),
        wp.dot(p_a, b) + wp.dot(a, p_b),
        b.dtype(2.0) * wp.dot(b, p_b),
    )
    dw = H @ dg
    hp_a = a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b
    hp_b = dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b
    hp_a = weight_cache[cid] * hp_a
    hp_b = weight_cache[cid] * hp_b
    wp.atomic_add(output, cell[0], -(hp_a + hp_b))
    wp.atomic_add(output, cell[1], hp_a)
    wp.atomic_add(output, cell[2], hp_b)


def _clone_wp(value: wp.array) -> tuple[torch.Tensor, wp.array]:
    tensor = wp.to_torch(value).detach().clone().contiguous()
    return tensor, wp.from_torch(tensor, dtype=value.dtype)


@dataclass(slots=True)
class _BulkCache:
    potential: StableNeoHookeanStress
    fields: dict[str, wp.array]
    tensors: tuple[torch.Tensor, ...]
    F: wp.array
    J: wp.array
    cofactor: wp.array
    cache_tensors: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class _MembraneCache:
    potential: StableNeoHookeanMembrane
    fields: dict[str, wp.array]
    tensors: tuple[torch.Tensor, ...]
    a: wp.array
    b: wp.array
    w: wp.array
    H: wp.array
    weight: wp.array
    cache_tensors: tuple[torch.Tensor, ...]


class CachedFemHvp:
    """Exact frozen FEM HVP for the joint bulk and skin potentials only.

    ``setup`` snapshots all material fields used by the Hessian and rejects a
    changed source displacement in ``apply``.  The returned full vector has no
    contact contribution; callers must add that separately when required.
    """

    def __init__(self, model: Any, state: Any) -> None:
        self.model = model
        self._state: Any = None
        self._state_key: tuple[int, int, int] | None = None
        self._bulk: list[_BulkCache] = []
        self._membrane: list[_MembraneCache] = []
        self.setup(state)

    @staticmethod
    def _potentials(model: Any) -> list[Any]:
        try:
            potentials = model.warp_model.__wrapped__.potentials
        except AttributeError as error:
            message = (
                "expected model.warp_model.__wrapped__.potentials from WarpModelAdapter"
            )
            raise TypeError(message) from error
        unsupported = {
            name: type(potential).__name__
            for name, potential in potentials.items()
            if type(potential) not in {StableNeoHookeanStress, StableNeoHookeanMembrane}
        }
        if unsupported:
            message = f"unsupported FEM potentials: {unsupported}"
            raise TypeError(message)
        return list(potentials.values())

    @staticmethod
    def _key(state: Any) -> tuple[int, int, int]:
        displacement = state.u
        return id(state), displacement.data_ptr(), displacement._version

    @staticmethod
    def _stream(tensor: torch.Tensor) -> contextlib.AbstractContextManager[Any]:
        if tensor.is_cuda:
            return wp.ScopedStream(
                wp.stream_from_torch(torch.cuda.current_stream(tensor.device))
            )
        return contextlib.nullcontext()

    def setup(self, state: Any) -> None:
        """Build state/material caches; excluded from repeated HVP timing."""
        self._potentials(self.model)
        displacement = state.u
        if displacement.ndim != 2 or displacement.shape[1] != 3:
            message = "state.u must have shape (points, 3)"
            raise ValueError(message)
        self._bulk.clear()
        self._membrane.clear()
        with self._stream(displacement):
            u_wp = _vec3(displacement)
            for potential in self._potentials(self.model):
                if type(potential) is StableNeoHookeanStress:
                    self._bulk.append(self._make_bulk(potential, u_wp, displacement))
                else:
                    self._membrane.append(
                        self._make_membrane(potential, u_wp, displacement)
                    )
        self._state = state
        self._state_key = self._key(state)

    def _make_bulk(
        self,
        potential: StableNeoHookeanStress,
        displacement: wp.array,
        tensor: torch.Tensor,
    ) -> _BulkCache:
        names = ("dhdX", "dV", "lmbda", "mu", "active_stress")
        clones = [_clone_wp(getattr(potential.materials, name)) for name in names]
        owned, fields = zip(*clones, strict=True)
        count, quadrature = potential.launch_dim
        F_tensor = torch.empty(
            (count, quadrature, 3, 3), dtype=tensor.dtype, device=tensor.device
        )
        J_tensor = torch.empty(
            (count, quadrature), dtype=tensor.dtype, device=tensor.device
        )
        cof_tensor = torch.empty_like(F_tensor)
        F = _array(F_tensor, wp.types.matrix((3, 3), wp.dtype_from_torch(tensor.dtype)))
        J = _array(J_tensor, wp.dtype_from_torch(tensor.dtype))
        cofactor = _array(
            cof_tensor, wp.types.matrix((3, 3), wp.dtype_from_torch(tensor.dtype))
        )
        potential.deformation_gradient(displacement, F)
        wp.launch(
            _bulk_invariants,
            dim=potential.launch_dim,
            inputs=[F, J, cofactor],
            device=potential.cells.device,
        )
        return _BulkCache(
            potential,
            dict(zip(names, fields, strict=True)),
            owned,
            F,
            J,
            cofactor,
            (F_tensor, J_tensor, cof_tensor),
        )

    def _make_membrane(
        self,
        potential: StableNeoHookeanMembrane,
        displacement: wp.array,
        tensor: torch.Tensor,
    ) -> _MembraneCache:
        names = (
            "rest_edge_01",
            "rest_edge_02",
            "metric_map",
            "baseline_stress",
            "lmbda",
            "mu",
            "thickness",
            "fraction",
            "rest_metric_sqrt_det",
        )
        clones = [_clone_wp(getattr(potential.materials, name)) for name in names]
        owned, fields = zip(*clones, strict=True)
        count = potential.launch_dim
        dtype = wp.dtype_from_torch(tensor.dtype)
        a_tensor = torch.empty((count, 3), dtype=tensor.dtype, device=tensor.device)
        b_tensor, w_tensor = torch.empty_like(a_tensor), torch.empty_like(a_tensor)
        H_tensor = torch.empty((count, 3, 3), dtype=tensor.dtype, device=tensor.device)
        weight_tensor = torch.empty((count,), dtype=tensor.dtype, device=tensor.device)
        a, b, w = (_vec3(a_tensor), _vec3(b_tensor), _vec3(w_tensor))
        H = _array(H_tensor, wp.types.matrix((3, 3), dtype))
        weight = _array(weight_tensor, dtype)
        copied = dict(zip(names, fields, strict=True))
        wp.launch(
            _membrane_setup,
            dim=count,
            inputs=[
                displacement,
                potential.cells,
                *(copied[name] for name in names),
                a,
                b,
                w,
                H,
                weight,
            ],
            device=potential.cells.device,
        )
        return _MembraneCache(
            potential,
            copied,
            owned,
            a,
            b,
            w,
            H,
            weight,
            (a_tensor, b_tensor, w_tensor, H_tensor, weight_tensor),
        )

    def apply(self, direction_full: torch.Tensor) -> torch.Tensor:
        """Return the frozen exact bulk-plus-membrane HVP for ``direction_full``."""
        if self._state is None or self._state_key != self._key(self._state):
            message = (
                "cached FEM HVP is stale; call setup(state) after displacement mutation"
            )
            raise RuntimeError(message)
        if direction_full.shape != self._state.u.shape:
            message = "direction_full shape must match the frozen displacement"
            raise ValueError(message)
        if (
            direction_full.dtype != self._state.u.dtype
            or direction_full.device != self._state.u.device
        ):
            message = (
                "direction_full dtype and device must match the frozen displacement"
            )
            raise ValueError(message)
        output = torch.zeros_like(direction_full)
        with self._stream(direction_full):
            direction, result = _vec3(direction_full), _vec3(output)
            for cache in self._bulk:
                fields = cache.fields
                wp.launch(
                    _bulk_apply,
                    dim=cache.potential.launch_dim,
                    inputs=[
                        direction,
                        cache.potential.cells,
                        fields["dhdX"],
                        fields["dV"],
                        fields["lmbda"],
                        fields["mu"],
                        fields["active_stress"],
                        cache.F,
                        cache.J,
                        cache.cofactor,
                        result,
                    ],
                    device=cache.potential.cells.device,
                )
            for cache in self._membrane:
                wp.launch(
                    _membrane_apply,
                    dim=cache.potential.launch_dim,
                    inputs=[
                        direction,
                        cache.potential.cells,
                        cache.a,
                        cache.b,
                        cache.w,
                        cache.H,
                        cache.weight,
                        result,
                    ],
                    device=cache.potential.cells.device,
                )
        return output

    @property
    def state_key(self) -> tuple[int, int, int] | None:
        return self._state_key

    @property
    def persistent_bytes(self) -> int:
        caches = [*self._bulk, *self._membrane]
        return sum(
            tensor.numel() * tensor.element_size()
            for cache in caches
            for tensor in (*cache.tensors, *cache.cache_tensors)
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "potential_container": "model.warp_model.__wrapped__.potentials",
            "bulk_potentials": len(self._bulk),
            "membrane_potentials": len(self._membrane),
            "persistent_bytes": self.persistent_bytes,
            "includes_contact": False,
            "state_key": self.state_key,
            "cached_bulk": ("F", "J", "cofactor"),
            "cached_membrane": ("a", "b", "w", "H", "weight"),
        }


__all__ = ["CachedFemHvp"]
