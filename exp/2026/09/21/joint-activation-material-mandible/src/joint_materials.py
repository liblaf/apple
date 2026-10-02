"""Experiment-local signed-stress bulk and membrane materials.

The volume material uses the polynomial Stable Neo-Hookean energy plus a signed
symmetric second-Piola stress in the neutral material frame.  The membrane uses
the exact plane-stress reduction of the same polynomial and a signed symmetric
stress resultant in a frozen orthonormal neutral tangent frame.

Callers supply the Stable Neo-Hookean ``lmbda`` code parameter.  For a desired
small-strain ``(E, nu)``, this experiment uses
``lmbda = E*nu/((1+nu)*(1-2*nu)) + mu``.  Neither class performs that conversion.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, cast, no_type_check, override

import attrs
import numpy as np
import torch
import warp as wp
from torch import Tensor

from liblaf.apple.common import FRACTION, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math as warp_math
from liblaf.apple.warp.fem import WarpPotentialFem, func
from liblaf.apple.warp.model import MaterialField, Struct, WarpPotential, make_struct
from liblaf.apple.warp.utils import warp_default_dtype

ACTIVE_STRESS = "active_stress"
BASELINE_STRESS = "baseline_stress"
THICKNESS = "thickness"

floating = Any
mat22 = Any
mat33 = Any
mat43 = Any
vec3 = Any
vec3i = Any
Materials = Any


def _validate_symmetric(
    values: np.ndarray, expected: tuple[int, int, int], name: str
) -> np.ndarray:
    values = np.asarray(values)
    flat_shape = (expected[0], expected[1] * expected[2])
    if values.shape == flat_shape:
        values = values.reshape(expected)
    if values.shape != expected:
        msg = f"{name} must have shape {expected}, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    if not np.allclose(values, np.swapaxes(values, -1, -2), rtol=0.0, atol=1.0e-12):
        msg = f"{name} must be symmetric"
        raise ValueError(msg)
    return np.ascontiguousarray(values)


def _get_active_stress(region: Region, annotation: Any) -> wp.array:
    vtk_name = "ActiveStress"
    values = region.cell_data.get(vtk_name)
    if values is None:
        return wp.zeros((region.n_cells,), dtype=annotation.dtype)
    values = _validate_symmetric(np.asarray(values), (region.n_cells, 3, 3), vtk_name)
    return wp.from_numpy(values, dtype=annotation.dtype)


@wp.func
def bulk_energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    """Return polynomial SNH plus signed symmetric additive stress."""
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    Q = materials.active_stress[cid]
    J = func.I3(F)
    passive = (
        F.dtype(0.5) * mu * (func.I2(F) - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * warp_math.square(J - F.dtype(1.0))
    )
    C = wp.transpose(F) @ F
    active = F.dtype(0.5) * wp.ddot(Q, C - wp.identity(3, dtype=F.dtype))
    return passive + active


@wp.func
def bulk_first_piola(F: mat33, materials: Materials, cid: int) -> mat33:
    """Return ``P_SNH(F) + F Q`` under the signed-symmetric-Q contract."""
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    Q = materials.active_stress[cid]
    J = func.I3(F)
    passive = F.dtype(0.5) * mu * func.g2(F) + (
        -mu + la * (J - F.dtype(1.0))
    ) * func.g3(F)
    return passive + F @ Q


@wp.func
@no_type_check
def bulk_hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat43:
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    Q = materials.active_stress[cid]
    J = func.I3(F)
    g3 = func.g3(F)
    passive = (
        la * func.h3_diag(dhdX, g3)
        + F.dtype(0.5) * mu * func.h5_diag(dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_diag(dhdX, F)
    )
    q0 = wp.dot(dhdX[0], Q @ dhdX[0])
    q1 = wp.dot(dhdX[1], Q @ dhdX[1])
    q2 = wp.dot(dhdX[2], Q @ dhdX[2])
    q3 = wp.dot(dhdX[3], Q @ dhdX[3])
    active = wp.matrix_from_rows(
        wp.vector(q0, q0, q0),
        wp.vector(q1, q1, q1),
        wp.vector(q2, q2, q2),
        wp.vector(q3, q3, q3),
    )
    return passive + active


@wp.func
def bulk_hess_prod(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> mat43:
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    Q = materials.active_stress[cid]
    J = func.I3(F)
    g3 = func.g3(F)
    passive = (
        la * func.h3_prod(p, dhdX, g3)
        + F.dtype(0.5) * mu * func.h5_prod(p, dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_prod(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)
    return passive + func.deformation_gradient_vjp(dhdX, dF @ Q)


@wp.func
def bulk_hess_quad(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> floating:
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    Q = materials.active_stress[cid]
    J = func.I3(F)
    g3 = func.g3(F)
    passive = (
        la * func.h3_quad(p, dhdX, g3)
        + F.dtype(0.5) * mu * func.h5_quad(p, dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_quad(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)
    return passive + wp.ddot(dF, dF @ Q)


@attrs.define
class StableNeoHookeanStress(WarpPotentialFem):
    """Polynomial Stable Neo-Hookean material with signed additive stress.

    ``active_stress`` is a signed symmetric 3x3 second-Piola tensor per cell in
    the neutral reference frame.  The class intentionally does not project it;
    baseline admissibility and activation PSD projection belong to the caller.
    """

    class Materials(WarpPotentialFem.Materials):
        active_stress: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVE_STRESS: MaterialField(
            name=ACTIVE_STRESS,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((3, 3), dtype)),
            factory=_get_active_stress,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
    }

    energy_density_func: ClassVar[wp.Function] = cast(
        "wp.Function", bulk_energy_density
    )
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", bulk_first_piola
    )
    hess_diag_func: ClassVar[wp.Function] = cast("wp.Function", bulk_hess_diag)
    hess_prod_func: ClassVar[wp.Function] = cast("wp.Function", bulk_hess_prod)
    hess_quad_func: ClassVar[wp.Function] = cast("wp.Function", bulk_hess_quad)

    energy_density_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_energy_density_kernel(energy_density_func)
    )
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_first_piola_kirchhoff_kernel(first_piola_kirchhoff_func)
    )
    fun_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_fun_kernel(
        energy_density_func
    )
    grad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_grad_kernel(
        first_piola_kirchhoff_func
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_diag_kernel(
        hess_diag_func
    )
    hess_prod_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_prod_kernel(
        hess_prod_func
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_quad_kernel(
        hess_quad_func
    )

    @override
    def material_from_region(
        self, region: Region, requires_grad: Sequence[str] = ()
    ) -> Materials:
        unknown = set(requires_grad) - self.MATERIAL_FIELDS.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            msg = f"unknown bulk material fields: {names}"
            raise ValueError(msg)
        materials = self.material_struct()
        with wp.ScopedDevice(self.cells.device):
            for field in self.material_vars.values():
                value = field.from_region(region)
                if field.name in requires_grad:
                    value = wp.clone(value, requires_grad=True)
                setattr(materials, field.name, value)
        return cast("Materials", materials)


def rest_tangent_frames(obj: Region | Any) -> Tensor:
    """Return frozen right-handed orthonormal triangle frames ``(cells, 3, 2)``."""
    region = obj if isinstance(obj, Region) else Region.from_pyvista(obj)
    points = region.points
    cells = region.cells_local
    edge_0 = points[cells[:, 1]] - points[cells[:, 0]]
    edge_1 = points[cells[:, 2]] - points[cells[:, 0]]
    norm_0 = torch.linalg.vector_norm(edge_0, dim=1)
    normals = torch.linalg.cross(edge_0, edge_1, dim=1)
    norm_n = torch.linalg.vector_norm(normals, dim=1)
    if torch.any(norm_0 <= 0) or torch.any(norm_n <= 0):
        msg = "membrane rest triangles must have positive area and first-edge length"
        raise ValueError(msg)
    tangent_0 = edge_0 / norm_0[:, None]
    normal = normals / norm_n[:, None]
    tangent_1 = torch.linalg.cross(normal, tangent_0, dim=1)
    return torch.stack((tangent_0, tangent_1), dim=2).contiguous()


def _rest_terms(
    region: Region,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    points = region.points
    cells = region.cells_local
    edge_0 = points[cells[:, 1]] - points[cells[:, 0]]
    edge_1 = points[cells[:, 2]] - points[cells[:, 0]]
    frames = rest_tangent_frames(region)
    edges = torch.stack((edge_0, edge_1), dim=2)
    coordinates = torch.bmm(frames.transpose(1, 2), edges)
    coordinate_inv = torch.linalg.inv(coordinates)

    basis = torch.zeros((3, 2, 2), dtype=points.dtype, device=points.device)
    basis[0, 0, 0] = 1.0
    basis[1, 0, 1] = 1.0
    basis[1, 1, 0] = 1.0
    basis[2, 1, 1] = 1.0
    pulled = torch.einsum("cai,kab,cbj->ckij", coordinate_inv, basis, coordinate_inv)
    metric_map = torch.stack(
        (pulled[..., 0, 0], pulled[..., 0, 1], pulled[..., 1, 1]), dim=1
    ).contiguous()

    metric_00 = torch.sum(edge_0 * edge_0, dim=1)
    metric_01 = torch.sum(edge_0 * edge_1, dim=1)
    metric_11 = torch.sum(edge_1 * edge_1, dim=1)
    sqrt_det = torch.sqrt(metric_00 * metric_11 - metric_01.square()).contiguous()
    return edge_0.contiguous(), edge_1.contiguous(), metric_map, sqrt_det, frames


def _get_rest_edge(region: Region, annotation: Any, index: int) -> wp.array:
    terms = _rest_terms(region)
    return wp.from_torch(terms[index], dtype=annotation.dtype)


def _get_metric_map(region: Region, annotation: Any) -> wp.array:
    return wp.from_torch(_rest_terms(region)[2], dtype=annotation.dtype)


def _get_rest_sqrt_det(region: Region, annotation: Any) -> wp.array:
    return wp.from_torch(_rest_terms(region)[3], dtype=annotation.dtype)


def _get_fraction(region: Region, annotation: Any) -> wp.array:
    values = region.cell_data.get(FRACTION.vtk)
    if values is None:
        values = np.ones(region.n_cells)
    values = np.asarray(values)
    if values.shape != (region.n_cells,) or not np.all(np.isfinite(values)):
        msg = f"{FRACTION.vtk} must be a finite scalar per triangle"
        raise ValueError(msg)
    return wp.from_numpy(np.ascontiguousarray(values), dtype=annotation.dtype)


def _get_baseline_stress(region: Region, annotation: Any) -> wp.array:
    vtk_name = "BaselineStress"
    values = region.cell_data.get(vtk_name)
    if values is None:
        return wp.zeros((region.n_cells,), dtype=annotation.dtype)
    values = _validate_symmetric(np.asarray(values), (region.n_cells, 2, 2), vtk_name)
    return wp.from_numpy(values, dtype=annotation.dtype)


@wp.func
@no_type_check
def _metric(a: vec3, b: vec3) -> vec3:
    return wp.vector(wp.dot(a, a), wp.dot(a, b), wp.dot(b, b))


@wp.func
@no_type_check
def _metric_to_tangent(g: vec3, metric_map: mat33) -> vec3:
    return metric_map @ g


@wp.func
@no_type_check
def _area_ratio(c: vec3) -> floating:
    return wp.sqrt(c[0] * c[2] - c[1] * c[1])


@wp.func
@no_type_check
def _area_gradient(c: vec3, area: floating) -> vec3:
    return wp.vector(
        c[2] / (area.dtype(2.0) * area),
        -c[1] / area,
        c[0] / (area.dtype(2.0) * area),
    )


@wp.func
@no_type_check
def _area_hessian(c: vec3, area: floating) -> mat33:
    d = wp.vector(c[2], -area.dtype(2.0) * c[1], c[0])
    determinant_hessian = wp.matrix_from_rows(
        wp.vector(area.dtype(0.0), area.dtype(0.0), area.dtype(1.0)),
        wp.vector(area.dtype(0.0), -area.dtype(2.0), area.dtype(0.0)),
        wp.vector(area.dtype(1.0), area.dtype(0.0), area.dtype(0.0)),
    )
    return area.dtype(0.5) / area * determinant_hessian - area.dtype(0.25) / (
        area * area * area
    ) * wp.outer(d, d)


@wp.func
@no_type_check
def _normal_stretch(area: floating, la: floating, mu: floating) -> floating:
    return area * (la + mu) / (la * area * area + mu)


@wp.func
@no_type_check
def membrane_energy_density(
    g: vec3,
    metric_map: mat33,
    baseline: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> floating:
    """Return exact plane-stress SNH plus tangential stress per neutral area."""
    c = _metric_to_tangent(g, metric_map)
    area = _area_ratio(c)
    z = _normal_stretch(area, la, mu)
    J = area * z
    passive = thickness * (
        g.dtype(0.5) * mu * (c[0] + c[2] + z * z - g.dtype(3.0))
        - mu * (J - g.dtype(1.0))
        + g.dtype(0.5) * la * (J - g.dtype(1.0)) * (J - g.dtype(1.0))
    )
    active = g.dtype(0.5) * (
        baseline[0, 0] * (c[0] - g.dtype(1.0))
        + g.dtype(2.0) * baseline[0, 1] * c[1]
        + baseline[1, 1] * (c[2] - g.dtype(1.0))
    )
    return passive + active


@wp.func
@no_type_check
def _tangent_metric_gradient(
    c: vec3,
    baseline: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> vec3:
    area = _area_ratio(c)
    z = _normal_stretch(area, la, mu)
    k = -mu + la * (area * z - c.dtype(1.0))
    dpsi_darea = thickness * z * k
    da = _area_gradient(c, area)
    return (
        wp.vector(
            thickness * c.dtype(0.5) * mu + c.dtype(0.5) * baseline[0, 0],
            baseline[0, 1],
            thickness * c.dtype(0.5) * mu + c.dtype(0.5) * baseline[1, 1],
        )
        + dpsi_darea * da
    )


@wp.func
@no_type_check
def _tangent_metric_hessian(
    c: vec3, la: floating, mu: floating, thickness: floating
) -> mat33:
    area = _area_ratio(c)
    denominator = la * area * area + mu
    z = _normal_stretch(area, la, mu)
    dz = (la + mu) * (mu - la * area * area) / (denominator * denominator)
    k = -mu + la * (area * z - c.dtype(1.0))
    dpsi_darea = thickness * z * k
    d2psi_darea2 = thickness * (dz * k + z * la * (z + area * dz))
    da = _area_gradient(c, area)
    return d2psi_darea2 * wp.outer(da, da) + dpsi_darea * _area_hessian(c, area)


@wp.func
@no_type_check
def membrane_metric_gradient(
    g: vec3,
    metric_map: mat33,
    baseline: mat22,
    la: floating,
    mu: floating,
    thickness: floating,
) -> vec3:
    c = _metric_to_tangent(g, metric_map)
    return wp.transpose(metric_map) @ _tangent_metric_gradient(
        c, baseline, la, mu, thickness
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
    c = _metric_to_tangent(g, metric_map)
    H = _tangent_metric_hessian(c, la, mu, thickness)
    return wp.transpose(metric_map) @ H @ metric_map


@wp.func
def _quad(H: mat33, a: vec3, b: vec3) -> floating:
    return wp.dot(a, H @ b)


@wp.func
@no_type_check
def _diag_component(
    H: mat33, w: vec3, a: floating, b: floating, which: int
) -> floating:
    zero = a.dtype(0.0)
    ja = wp.vector(a.dtype(2.0) * a, b, zero)
    jb = wp.vector(zero, a, a.dtype(2.0) * b)
    Haa = _quad(H, ja, ja) + a.dtype(2.0) * w[0]
    Hbb = _quad(H, jb, jb) + a.dtype(2.0) * w[2]
    Hab = _quad(H, ja, jb) + w[1]
    if which == 0:
        return Haa + Hbb + a.dtype(2.0) * Hab
    if which == 1:
        return Haa
    return Hbb


@wp.func
@no_type_check
def _area_weight(fraction: floating, rest_metric_sqrt_det: floating) -> floating:
    return fraction * rest_metric_sqrt_det / fraction.dtype(2.0)


@wp.func
def _membrane_edge_grad(
    a: vec3, b: vec3, materials: Materials, cid: int
) -> tuple[vec3, vec3]:
    g = _metric(a, b)
    w = membrane_metric_gradient(
        g,
        materials.metric_map[cid],
        materials.baseline_stress[cid],
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
def _membrane_fun_kernel(
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
        materials.baseline_stress[cid],
        materials.lmbda[cid],
        materials.mu[cid],
        materials.thickness[cid],
    )
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    wp.atomic_add(output, 0, weight * density)


@wp.kernel(module="unique")
@no_type_check
def _membrane_grad_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    grad_a, grad_b = _membrane_edge_grad(a, b, materials, cid)
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    grad_a = weight * grad_a
    grad_b = weight * grad_b
    wp.atomic_add(output, cell[0], -(grad_a + grad_b))
    wp.atomic_add(output, cell[1], grad_a)
    wp.atomic_add(output, cell[2], grad_b)


@wp.kernel(module="unique")
@no_type_check
def _membrane_hess_diag_kernel(
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
        materials.baseline_stress[cid],
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
    diag_0 = weight * wp.vector(
        _diag_component(H, w, a[0], b[0], 0),
        _diag_component(H, w, a[1], b[1], 0),
        _diag_component(H, w, a[2], b[2], 0),
    )
    diag_1 = weight * wp.vector(
        _diag_component(H, w, a[0], b[0], 1),
        _diag_component(H, w, a[1], b[1], 1),
        _diag_component(H, w, a[2], b[2], 1),
    )
    diag_2 = weight * wp.vector(
        _diag_component(H, w, a[0], b[0], 2),
        _diag_component(H, w, a[1], b[1], 2),
        _diag_component(H, w, a[2], b[2], 2),
    )
    wp.atomic_add(output, cell[0], diag_0)
    wp.atomic_add(output, cell[1], diag_1)
    wp.atomic_add(output, cell[2], diag_2)


@wp.kernel(module="unique")
@no_type_check
def _membrane_hess_prod_kernel(
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
    p_a = p[cell[1]] - p[cell[0]]
    p_b = p[cell[2]] - p[cell[0]]
    g = _metric(a, b)
    w = membrane_metric_gradient(
        g,
        materials.metric_map[cid],
        materials.baseline_stress[cid],
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
    Hp_a = a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b
    Hp_b = dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    Hp_a = weight * Hp_a
    Hp_b = weight * Hp_b
    wp.atomic_add(output, cell[0], -(Hp_a + Hp_b))
    wp.atomic_add(output, cell[1], Hp_a)
    wp.atomic_add(output, cell[2], Hp_b)


@wp.kernel(module="unique")
@no_type_check
def _membrane_hess_quad_kernel(
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
    g = _metric(a, b)
    w = membrane_metric_gradient(
        g,
        materials.metric_map[cid],
        materials.baseline_stress[cid],
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
    Hp_a = a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b
    Hp_b = dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b
    weight = _area_weight(materials.fraction[cid], materials.rest_metric_sqrt_det[cid])
    h_quad = weight * (wp.dot(p_a, Hp_a) + wp.dot(p_b, Hp_b))
    # Only the PNCG step estimate is clamped; hess_prod remains exact.
    wp.atomic_add(output, 0, wp.max(h_quad, h_quad.dtype(0.0)))


@attrs.define
class StableNeoHookeanMembrane(WarpPotential):
    """Exact plane-stress polynomial SNH membrane with signed stress resultant.

    ``baseline_stress`` is a symmetric 2x2 tensor in the frozen tangent frame
    returned by :func:`rest_tangent_frames`.  With geometry measured in metres
    and bulk moduli in MPa, its internal unit is MPa*m.  Multiply by ``1e6`` to
    report N/m.  ``thickness`` is the fixed reference thickness in metres.
    """

    class Materials(WarpPotential.Materials):
        baseline_stress: wp.array
        fraction: wp.array
        lmbda: wp.array
        metric_map: wp.array
        mu: wp.array
        rest_edge_01: wp.array
        rest_edge_02: wp.array
        rest_metric_sqrt_det: wp.array
        thickness: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        BASELINE_STRESS: MaterialField(
            name=BASELINE_STRESS,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((2, 2), dtype)),
            factory=_get_baseline_stress,
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

    fun_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _membrane_fun_kernel)
    grad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _membrane_grad_kernel)
    hess_diag_kernel: ClassVar[wp.Kernel] = cast(
        "wp.Kernel", _membrane_hess_diag_kernel
    )
    hess_prod_kernel: ClassVar[wp.Kernel] = cast(
        "wp.Kernel", _membrane_hess_prod_kernel
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = cast(
        "wp.Kernel", _membrane_hess_quad_kernel
    )

    cells: wp.array
    thickness: float | np.ndarray = attrs.field(default=1.0, kw_only=True)

    @functools.cached_property
    def material_struct(self) -> Struct:
        return _membrane_material_struct(warp_default_dtype())

    def _material_from_region(
        self, region: Region, requires_grad: Sequence[str]
    ) -> Materials:
        thickness = np.asarray(self.thickness)
        if thickness.ndim == 0:
            thickness = np.full(region.n_cells, thickness.item())
        expected = (region.n_cells,)
        if thickness.shape != expected:
            msg = (
                f"thickness must be a scalar or have shape {expected}, "
                f"got {thickness.shape}"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(thickness)) or np.any(thickness <= 0):
            msg = "thickness values must be finite and positive"
            raise ValueError(msg)

        unknown = set(requires_grad) - self.MATERIAL_FIELDS.keys()
        if unknown:
            names = ", ".join(sorted(unknown))
            msg = f"unknown membrane material fields: {names}"
            raise ValueError(msg)

        materials = self.material_struct()
        with wp.ScopedDevice(self.cells.device):
            for field in self.material_vars.values():
                if field.name == THICKNESS:
                    value = wp.from_numpy(
                        np.ascontiguousarray(thickness), dtype=field.annotation.dtype
                    )
                else:
                    value = field.from_region(region)
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
            msg = "StableNeoHookeanMembrane expects triangle cells"
            raise ValueError(msg)
        self = cls(cells=wp.from_torch(cells, dtype=wp.vec3i), **kwargs)
        self.materials = self._material_from_region(region, requires_grad=requires_grad)
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


@functools.cache
def _membrane_material_struct(dtype: Any) -> Struct:
    material_vars = tuple(
        field.make(dtype) for field in StableNeoHookeanMembrane.MATERIAL_FIELDS.values()
    )
    return make_struct(
        material_vars,
        module=StableNeoHookeanMembrane.__module__,
        qualname=StableNeoHookeanMembrane.__qualname__,
    )


__all__ = [
    "ACTIVE_STRESS",
    "BASELINE_STRESS",
    "THICKNESS",
    "StableNeoHookeanMembrane",
    "StableNeoHookeanStress",
    "bulk_energy_density",
    "bulk_first_piola",
    "bulk_hess_diag",
    "bulk_hess_prod",
    "bulk_hess_quad",
    "membrane_energy_density",
    "membrane_metric_gradient",
    "membrane_metric_hessian",
    "rest_tangent_frames",
]
