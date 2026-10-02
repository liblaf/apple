"""Corrected active-strain membrane used by the new-neutral replay.

The tangential active metric is ``B=A_inv``.  Only the distortional norm uses
``B``; the plane-stress thickness stretch and the Jacobian retain the physical
surface-area ratio.  Thus changing ``B`` never changes how much membrane
material or physical volume the triangle represents.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, cast, no_type_check, override

import attrs
import numpy as np
import torch
import warp as wp
from joint_materials import (
    THICKNESS,
    _area_gradient,
    _area_ratio,
    _area_weight,
    _diag_component,
    _get_fraction,
    _get_metric_map,
    _get_rest_edge,
    _get_rest_sqrt_det,
    _metric,
    _metric_to_tangent,
    _normal_stretch,
    _tangent_metric_hessian,
)

from liblaf.apple.common import FRACTION, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.model import MaterialField, Struct, WarpPotential, make_struct
from liblaf.apple.warp.utils import warp_default_dtype

ACTIVATION_INV = "activation_inv"

floating = Any
mat22 = Any
mat33 = Any
vec3 = Any
vec3i = Any
Materials = Any


def _get_activation_inv(region: Region, annotation: Any) -> wp.array:
    """Read a finite SPD tangential ``ActivationInv`` matrix, defaulting to I."""
    values = region.cell_data.get("ActivationInv")
    if values is None:
        values = np.broadcast_to(np.eye(2), (region.n_cells, 2, 2)).copy()
    values = np.asarray(values)
    if values.shape == (region.n_cells, 4):
        values = values.reshape(region.n_cells, 2, 2)
    if values.shape != (region.n_cells, 2, 2):
        msg = f"ActivationInv must have shape ({region.n_cells}, 2, 2), got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "ActivationInv must contain only finite values"
        raise ValueError(msg)
    if not np.allclose(values, np.swapaxes(values, -1, -2), rtol=0.0, atol=1e-12):
        msg = "ActivationInv must be symmetric"
        raise ValueError(msg)
    if np.any(np.linalg.eigvalsh(values) <= 0.0):
        msg = "ActivationInv must be SPD"
        raise ValueError(msg)
    return wp.from_numpy(np.ascontiguousarray(values), dtype=annotation.dtype)


@wp.func
@no_type_check
def membrane_energy_density(
    g: vec3,
    metric_map: mat33,
    activation_inv: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> floating:
    """Exact plane-stress SNH with a tangential multiplicative active metric."""
    c = _metric_to_tangent(g, metric_map)
    area = _area_ratio(c)
    z = _normal_stretch(area, la, mu)
    J = area * z
    # Expanded form of tr(B.T C B), with S=B B.T.
    s00 = (
        activation_inv[0, 0] * activation_inv[0, 0]
        + activation_inv[0, 1] * activation_inv[0, 1]
    )
    s01 = (
        activation_inv[0, 0] * activation_inv[1, 0]
        + activation_inv[0, 1] * activation_inv[1, 1]
    )
    s11 = (
        activation_inv[1, 0] * activation_inv[1, 0]
        + activation_inv[1, 1] * activation_inv[1, 1]
    )
    return thickness * (
        g.dtype(0.5)
        * mu
        * (c[0] * s00 + g.dtype(2.0) * c[1] * s01 + c[2] * s11 + z * z - g.dtype(3.0))
        - mu * (J - g.dtype(1.0))
        + g.dtype(0.5) * la * (J - g.dtype(1.0)) * (J - g.dtype(1.0))
    )


@wp.func
@no_type_check
def _tangent_metric_gradient(
    c: vec3,
    activation_inv: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> vec3:
    area = _area_ratio(c)
    z = _normal_stretch(area, la, mu)
    k = -mu + la * (area * z - c.dtype(1.0))
    # Derivative of the physical-J part.  It is intentionally independent of B.
    da = _area_gradient(c, area)
    s00 = (
        activation_inv[0, 0] * activation_inv[0, 0]
        + activation_inv[0, 1] * activation_inv[0, 1]
    )
    s01 = (
        activation_inv[0, 0] * activation_inv[1, 0]
        + activation_inv[0, 1] * activation_inv[1, 1]
    )
    s11 = (
        activation_inv[1, 0] * activation_inv[1, 0]
        + activation_inv[1, 1] * activation_inv[1, 1]
    )
    return (
        wp.vector(
            thickness * c.dtype(0.5) * mu * s00,
            thickness * mu * s01,
            thickness * c.dtype(0.5) * mu * s11,
        )
        + thickness * z * k * da
    )


@wp.func
@no_type_check
def membrane_metric_gradient(
    g: vec3,
    metric_map: mat33,
    activation_inv: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> vec3:
    c = _metric_to_tangent(g, metric_map)
    return wp.transpose(metric_map) @ _tangent_metric_gradient(
        c, activation_inv, la, mu, thickness
    )


@wp.func
@no_type_check
def membrane_metric_hessian(
    g: vec3,
    metric_map: mat33,
    la: floating,
    mu: floating,
    thickness: floating,
) -> mat33:
    # B enters the energy linearly in C, therefore it adds no metric Hessian.
    c = _metric_to_tangent(g, metric_map)
    H = _tangent_metric_hessian(c, la, mu, thickness)
    return wp.transpose(metric_map) @ H @ metric_map


@wp.func
def _edge_grad(a: vec3, b: vec3, materials: Materials, cid: int) -> tuple[vec3, vec3]:
    w = membrane_metric_gradient(
        _metric(a, b),
        materials.metric_map[cid],
        materials.activation_inv[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    return (
        w[0] * a.dtype(2.0) * a + w[1] * b,
        w[1] * a + w[2] * b.dtype(2.0) * b,
    )


@wp.kernel(module="unique")
@no_type_check
def _fun_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[floating],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    density = membrane_energy_density(
        _metric(a, b),
        materials.metric_map[cid],
        materials.activation_inv[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    wp.atomic_add(
        output,
        0,
        _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
        * density,
    )


@wp.kernel(module="unique")
@no_type_check
def _grad_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    grad_a, grad_b = _edge_grad(a, b, materials, cid)
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    wp.atomic_add(output, cell[0], -weight * (grad_a + grad_b))
    wp.atomic_add(output, cell[1], weight * grad_a)
    wp.atomic_add(output, cell[2], weight * grad_b)


@wp.func
def _hvp_edges(
    a: vec3, b: vec3, p_a: vec3, p_b: vec3, materials: Materials, cid: int
) -> tuple[vec3, vec3]:
    g = _metric(a, b)
    w = membrane_metric_gradient(
        g,
        materials.metric_map[cid],
        materials.activation_inv[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    H = membrane_metric_hessian(
        g,
        materials.metric_map[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    dg = wp.vector(
        a.dtype(2.0) * wp.dot(a, p_a),
        wp.dot(p_a, b) + wp.dot(a, p_b),
        b.dtype(2.0) * wp.dot(b, p_b),
    )
    dw = H @ dg
    return (
        a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b,
        dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b,
    )


@wp.kernel(module="unique")
@no_type_check
def _hess_prod_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    hp_a, hp_b = _hvp_edges(
        a, b, p[cell[1]] - p[cell[0]], p[cell[2]] - p[cell[0]], materials, cid
    )
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    wp.atomic_add(output, cell[0], -weight * (hp_a + hp_b))
    wp.atomic_add(output, cell[1], weight * hp_a)
    wp.atomic_add(output, cell[2], weight * hp_b)


@wp.kernel(module="unique")
@no_type_check
def _hess_quad_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[floating],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    p_a = p[cell[1]] - p[cell[0]]
    p_b = p[cell[2]] - p[cell[0]]
    hp_a, hp_b = _hvp_edges(a, b, p_a, p_b, materials, cid)
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    wp.atomic_add(
        output,
        0,
        wp.max(weight * (wp.dot(p_a, hp_a) + wp.dot(p_b, hp_b)), weight.dtype(0.0)),
    )


@wp.kernel(module="unique")
@no_type_check
def _hess_diag_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    g = _metric(a, b)
    w = membrane_metric_gradient(
        g,
        materials.metric_map[cid],
        materials.activation_inv[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    H = membrane_metric_hessian(
        g,
        materials.metric_map[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    for vertex in range(3):
        diagonal = wp.vector(
            _diag_component(H, w, a[0], b[0], vertex),
            _diag_component(H, w, a[1], b[1], vertex),
            _diag_component(H, w, a[2], b[2], vertex),
        )
        wp.atomic_add(output, cell[vertex], weight * diagonal)


@attrs.define
class StableNeoHookeanActiveMembrane(WarpPotential):
    """Exact plane-stress SNH with tangential active strain and physical area."""

    class Materials(WarpPotential.Materials):
        activation_inv: wp.array
        fraction: wp.array
        lmbda: wp.array
        metric_map: wp.array
        mu: wp.array
        rest_edge_01: wp.array
        rest_edge_02: wp.array
        rest_metric_sqrt_det: wp.array
        thickness: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        ACTIVATION_INV: MaterialField(
            name=ACTIVATION_INV,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((2, 2), dtype)),
            factory=_get_activation_inv,
        ),
        FRACTION.value: MaterialField(
            name=FRACTION.value,
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_get_fraction,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        "metric_map": MaterialField(
            name="metric_map",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((3, 3), dtype)),
            factory=_get_metric_map,
        ),
        MU.value: MaterialField.CELL.floating(MU.value),
        "rest_edge_01": MaterialField(
            name="rest_edge_01",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=lambda region, annotation: _get_rest_edge(region, annotation, 0),
        ),
        "rest_edge_02": MaterialField(
            name="rest_edge_02",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=lambda region, annotation: _get_rest_edge(region, annotation, 1),
        ),
        "rest_metric_sqrt_det": MaterialField(
            name="rest_metric_sqrt_det",
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_get_rest_sqrt_det,
        ),
        THICKNESS: MaterialField.CELL.floating(THICKNESS),
    }
    fun_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _fun_kernel)
    grad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _grad_kernel)
    hess_diag_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_diag_kernel)
    hess_prod_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_prod_kernel)
    hess_quad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_quad_kernel)

    cells: wp.array
    thickness: float | np.ndarray = attrs.field(default=1.0, kw_only=True)

    @functools.cached_property
    def material_struct(self) -> Struct:
        fields = tuple(
            field.make(warp_default_dtype()) for field in self.MATERIAL_FIELDS.values()
        )
        return make_struct(
            fields, module=self.__module__, qualname=self.__class__.__qualname__
        )

    def _material_from_region(
        self, region: Region, requires_grad: Sequence[str]
    ) -> Materials:
        thickness = np.asarray(self.thickness)
        if thickness.ndim == 0:
            thickness = np.full(region.n_cells, thickness.item())
        if (
            thickness.shape != (region.n_cells,)
            or not np.all(np.isfinite(thickness))
            or np.any(thickness <= 0)
        ):
            msg = "thickness must be finite, positive, and scalar or per triangle"
            raise ValueError(msg)
        unknown = set(requires_grad) - self.MATERIAL_FIELDS.keys()
        if unknown:
            raise ValueError(
                "unknown active membrane material fields: " + ", ".join(sorted(unknown))
            )
        materials = self.material_struct()
        with wp.ScopedDevice(self.cells.device):
            for field in self.material_vars.values():
                value = (
                    wp.from_numpy(
                        np.ascontiguousarray(thickness), dtype=field.annotation.dtype
                    )
                    if field.name == THICKNESS
                    else field.from_region(region)
                )
                if field.name in requires_grad:
                    value = wp.clone(value, requires_grad=True)
                setattr(materials, field.name, value)
        return cast("Materials", materials)

    @classmethod
    @override
    def from_region(
        cls, region: Region, requires_grad: Sequence[str] = (), **kwargs: Any
    ) -> Self:
        cells = region.cells_global.to(torch.int32).contiguous()
        if cells.ndim != 2 or cells.shape[1] != 3:
            msg = "StableNeoHookeanActiveMembrane expects triangle cells"
            raise ValueError(msg)
        self = cls(cells=wp.from_torch(cells, dtype=wp.vec3i), **kwargs)
        self.materials = self._material_from_region(region, requires_grad)
        return self

    @property
    def launch_dim(self) -> int:
        return self.cells.shape[0]

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.fun_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials],
            outputs=[output],
            device=self.cells.device,
        )

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.grad_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials],
            outputs=[output],
            device=self.cells.device,
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_diag_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials],
            outputs=[output],
            device=self.cells.device,
        )

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.cells, self.materials],
            outputs=[output],
            device=self.cells.device,
        )

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_quad_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.cells, self.materials],
            outputs=[output],
            device=self.cells.device,
        )


__all__ = [
    "ACTIVATION_INV",
    "StableNeoHookeanActiveMembrane",
    "membrane_energy_density",
    "membrane_metric_gradient",
    "membrane_metric_hessian",
]
