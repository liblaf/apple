from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, cast, no_type_check, override

import attrs
import numpy as np
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FRACTION, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp.model import ArrayAnnotation, MaterialField, WarpPotential

floating = Any
mat22 = Any
mat33 = Any
vec3 = Any
vec3i = Any
Materials = Any


@wp.func
@no_type_check
def _make_activation_mat22(activation: vec3) -> mat22:
    return wp.identity(2, activation.dtype) + wp.matrix_from_rows(
        wp.vector(activation[0], activation[2]),
        wp.vector(activation[2], activation[1]),
    )


@wp.func
@no_type_check
def _metric(a: vec3, b: vec3) -> vec3:
    return wp.vector(wp.dot(a, a), wp.dot(a, b), wp.dot(b, b))


@wp.func
def _metric_inv(materials: Materials, cid: int) -> mat22:
    A_inv = _make_activation_mat22(materials.activation_inv[cid])
    return A_inv @ materials.rest_metric_inv[cid] @ wp.transpose(A_inv)


@wp.func
def _energy_weight(materials: Materials, cid: int, thickness: floating) -> floating:
    fraction = materials.fraction[cid]
    h = fraction.dtype(thickness)
    return h * fraction * materials.rest_metric_sqrt_det[cid] / fraction.dtype(8.0)


@wp.func
def _metric_energy_density(g: vec3, S: mat22, la: floating, mu: floating) -> floating:
    m00 = S[0, 0] * g[0] + S[0, 1] * g[1] - g.dtype(1.0)
    m01 = S[0, 0] * g[1] + S[0, 1] * g[2]
    m10 = S[1, 0] * g[0] + S[1, 1] * g[1]
    m11 = S[1, 0] * g[1] + S[1, 1] * g[2] - g.dtype(1.0)
    tr = m00 + m11
    tr_m2 = m00 * m00 + g.dtype(2.0) * m01 * m10 + m11 * m11
    return g.dtype(0.5) * la * tr * tr + mu * tr_m2


@wp.func
@no_type_check
def _metric_gradient(g: vec3, S: mat22, la: floating, mu: floating) -> vec3:
    s00 = S[0, 0]
    s01 = S[0, 1]
    s11 = S[1, 1]
    m00 = s00 * g[0] + s01 * g[1] - g.dtype(1.0)
    m01 = s00 * g[1] + s01 * g[2]
    m10 = s01 * g[0] + s11 * g[1]
    m11 = s01 * g[1] + s11 * g[2] - g.dtype(1.0)
    tr = m00 + m11
    return wp.vector(
        la * tr * s00 + mu * g.dtype(2.0) * (m00 * s00 + m01 * s01),
        la * tr * g.dtype(2.0) * s01
        + mu * g.dtype(2.0) * (m00 * s01 + m10 * s00 + m01 * s11 + m11 * s01),
        la * tr * s11 + mu * g.dtype(2.0) * (m10 * s01 + m11 * s11),
    )


@wp.func
@no_type_check
def _metric_hessian(S: mat22, la: floating, mu: floating) -> mat33:
    s00 = S[0, 0]
    s01 = S[0, 1]
    s11 = S[1, 1]
    tr_grad = wp.vector(s00, s01 + s01, s11)
    m00_grad = wp.vector(s00, s01, S.dtype(0.0))
    m01_grad = wp.vector(S.dtype(0.0), s00, s01)
    m10_grad = wp.vector(s01, s11, S.dtype(0.0))
    m11_grad = wp.vector(S.dtype(0.0), s01, s11)
    return la * wp.outer(tr_grad, tr_grad) + mu * S.dtype(2.0) * (
        wp.outer(m00_grad, m00_grad)
        + wp.outer(m01_grad, m10_grad)
        + wp.outer(m10_grad, m01_grad)
        + wp.outer(m11_grad, m11_grad)
    )


@wp.func
def _quad(H: mat33, a: vec3, b: vec3) -> floating:
    return wp.dot(a, H @ b)


@wp.func
@no_type_check
def _diag_component(
    H: mat33,
    w: vec3,
    a: floating,
    b: floating,
    which: int,
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
def _edge_grad(a: vec3, b: vec3, materials: Materials, cid: int) -> tuple[vec3, vec3]:
    S = _metric_inv(materials, cid)
    g = _metric(a, b)
    w = _metric_gradient(g, S, materials.lmbda[cid], materials.mu[cid])
    grad_a = w[0] * a.dtype(2.0) * a + w[1] * b
    grad_b = w[1] * a + w[2] * b.dtype(2.0) * b
    return grad_a, grad_b


@wp.kernel(module="unique")
@no_type_check
def _fun_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    thickness: floating,
    output: wp.array1d[floating],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    S = _metric_inv(materials, cid)
    g = _metric(a, b)
    W = _metric_energy_density(g, S, materials.lmbda[cid], materials.mu[cid])
    wp.atomic_add(output, 0, _energy_weight(materials, cid, thickness) * W)


@wp.kernel(module="unique")
@no_type_check
def _grad_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    thickness: floating,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    weight = _energy_weight(materials, cid, thickness)
    grad_a, grad_b = _edge_grad(a, b, materials, cid)
    grad_a = weight * grad_a
    grad_b = weight * grad_b
    wp.atomic_add(output, cell[0], -(grad_a + grad_b))
    wp.atomic_add(output, cell[1], grad_a)
    wp.atomic_add(output, cell[2], grad_b)


@wp.kernel(module="unique")
@no_type_check
def _hess_diag_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    thickness: floating,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    S = _metric_inv(materials, cid)
    g = _metric(a, b)
    w = _metric_gradient(g, S, materials.lmbda[cid], materials.mu[cid])
    H = _metric_hessian(S, materials.lmbda[cid], materials.mu[cid])
    weight = _energy_weight(materials, cid, thickness)
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
def _hess_prod_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    thickness: floating,
    output: wp.array1d[vec3],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    p_a = p[cell[1]] - p[cell[0]]
    p_b = p[cell[2]] - p[cell[0]]
    S = _metric_inv(materials, cid)
    g = _metric(a, b)
    w = _metric_gradient(g, S, materials.lmbda[cid], materials.mu[cid])
    H = _metric_hessian(S, materials.lmbda[cid], materials.mu[cid])
    dg = wp.vector(
        a.dtype(2.0) * wp.dot(a, p_a),
        wp.dot(p_a, b) + wp.dot(a, p_b),
        b.dtype(2.0) * wp.dot(b, p_b),
    )
    dw = H @ dg
    weight = _energy_weight(materials, cid, thickness)
    Hp_a = weight * (
        a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b
    )
    Hp_b = weight * (
        dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b
    )
    wp.atomic_add(output, cell[0], -(Hp_a + Hp_b))
    wp.atomic_add(output, cell[1], Hp_a)
    wp.atomic_add(output, cell[2], Hp_b)


@wp.kernel(module="unique")
@no_type_check
def _hess_quad_kernel(
    u: wp.array1d[vec3],
    p: wp.array1d[vec3],
    cells: wp.array1d[vec3i],
    materials: Materials,
    thickness: floating,
    output: wp.array1d[floating],
) -> None:
    cid = wp.tid()
    cell = cells[cid]
    a = materials.rest_edge_01[cid] + u[cell[1]] - u[cell[0]]
    b = materials.rest_edge_02[cid] + u[cell[2]] - u[cell[0]]
    p_a = p[cell[1]] - p[cell[0]]
    p_b = p[cell[2]] - p[cell[0]]
    S = _metric_inv(materials, cid)
    g = _metric(a, b)
    w = _metric_gradient(g, S, materials.lmbda[cid], materials.mu[cid])
    H = _metric_hessian(S, materials.lmbda[cid], materials.mu[cid])
    dg = wp.vector(
        a.dtype(2.0) * wp.dot(a, p_a),
        wp.dot(p_a, b) + wp.dot(a, p_b),
        b.dtype(2.0) * wp.dot(b, p_b),
    )
    dw = H @ dg
    Hp_a = a.dtype(2.0) * dw[0] * a + a.dtype(2.0) * w[0] * p_a + dw[1] * b + w[1] * p_b
    Hp_b = dw[1] * a + w[1] * p_a + b.dtype(2.0) * dw[2] * b + b.dtype(2.0) * w[2] * p_b
    h_quad = _energy_weight(materials, cid, thickness) * (
        wp.dot(p_a, Hp_a) + wp.dot(p_b, Hp_b)
    )
    wp.atomic_add(output, 0, h_quad)


def _get_activation_inv2(region: Region, annotation: ArrayAnnotation) -> wp.array:
    activation_inv: Float[np.ndarray, "c 3"] | None = region.cell_data.get(
        ACTIVATION_INV.vtk
    )
    if activation_inv is not None:
        activation_inv = np.asarray(activation_inv)
        _check_cell_vec3(region, activation_inv, ACTIVATION_INV.vtk)
        return wp.from_numpy(activation_inv, annotation.dtype)

    activation: Float[np.ndarray, "c 3"] | None = region.cell_data.get(ACTIVATION.vtk)
    if activation is not None:
        activation = np.asarray(activation)
        _check_cell_vec3(region, activation, ACTIVATION.vtk)
        A: Float[np.ndarray, "c 2 2"] = np.zeros((region.n_cells, 2, 2))
        A[:, 0, 0] = 1.0 + activation[:, 0]
        A[:, 1, 1] = 1.0 + activation[:, 1]
        A[:, 0, 1] = A[:, 1, 0] = activation[:, 2]
        A_inv: Float[np.ndarray, "c 2 2"] = np.linalg.inv(A)
        activation_inv = np.zeros((region.n_cells, 3))
        activation_inv[:, 0] = A_inv[:, 0, 0] - 1.0
        activation_inv[:, 1] = A_inv[:, 1, 1] - 1.0
        activation_inv[:, 2] = A_inv[:, 0, 1]
        return wp.from_numpy(activation_inv, annotation.dtype)

    return wp.zeros((region.n_cells,), annotation.dtype)


def _get_fraction(region: Region, annotation: ArrayAnnotation) -> wp.array:
    fraction: Float[np.ndarray, " c"] | None = region.cell_data.get(FRACTION.vtk)
    if fraction is None:
        fraction = np.ones(region.n_cells)
    return wp.from_numpy(np.asarray(fraction), annotation.dtype)


def _get_rest_edges(
    region: Region, annotation: ArrayAnnotation
) -> tuple[wp.array, wp.array]:
    points: Float[Tensor, "p 3"] = region.points
    cells: Tensor = region.cells_local
    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    return (
        wp.from_torch((p1 - p0).contiguous(), dtype=annotation.dtype),
        wp.from_torch((p2 - p0).contiguous(), dtype=annotation.dtype),
    )


def _rest_metric_terms(
    region: Region,
) -> tuple[Float[Tensor, "c 2 2"], Float[Tensor, " c"]]:
    points: Float[Tensor, "p 3"] = region.points
    cells: Tensor = region.cells_local
    p0 = points[cells[:, 0]]
    a = points[cells[:, 1]] - p0
    b = points[cells[:, 2]] - p0
    g00 = torch.sum(a * a, dim=1)
    g01 = torch.sum(a * b, dim=1)
    g11 = torch.sum(b * b, dim=1)
    det = g00 * g11 - g01.square()
    if torch.any(det <= 0):
        msg = "Koiter triangle rest metric must have positive determinant"
        raise ValueError(msg)
    metric = torch.stack(
        [
            torch.stack([g00, g01], dim=1),
            torch.stack([g01, g11], dim=1),
        ],
        dim=1,
    )
    metric_inv = torch.linalg.inv(metric).contiguous()
    metric_sqrt_det = torch.sqrt(det).contiguous()
    return metric_inv, metric_sqrt_det


def _get_rest_metric_inv(region: Region, annotation: ArrayAnnotation) -> wp.array:
    metric_inv, _ = _rest_metric_terms(region)
    return wp.from_torch(metric_inv, dtype=annotation.dtype)


def _get_rest_metric_sqrt_det(region: Region, annotation: ArrayAnnotation) -> wp.array:
    _, metric_sqrt_det = _rest_metric_terms(region)
    return wp.from_torch(metric_sqrt_det, dtype=annotation.dtype)


def _check_cell_vec3(region: Region, values: np.ndarray, name: str) -> None:
    expected = (region.n_cells, 3)
    if values.shape != expected:
        msg = f"{name} must have shape {expected}, got {values.shape}"
        raise ValueError(msg)


@attrs.define
class Koiter(WarpPotential):
    """Metric membrane with caller-supplied effective in-plane Lamé moduli.

    ``Lambda`` and ``Mu`` are not reduced internally. When they originate from
    3D Young's modulus and Poisson's ratio for a thin membrane, callers should
    apply a plane-stress conversion before constructing this potential.

    ``ActivationInv`` changes the stress-free metric while the energy remains
    integrated over the original triangle reference area. It does not change
    the amount of membrane material represented by a triangle.
    """

    class Materials(WarpPotential.Materials):
        activation_inv: wp.array
        fraction: wp.array
        lmbda: wp.array
        mu: wp.array
        rest_edge_01: wp.array
        rest_edge_02: wp.array
        rest_metric_inv: wp.array
        rest_metric_sqrt_det: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        ACTIVATION_INV.value: MaterialField(
            name=ACTIVATION_INV.value,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=_get_activation_inv2,
        ),
        FRACTION.value: MaterialField(
            name=FRACTION.value,
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_get_fraction,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
        "rest_edge_01": MaterialField(
            name="rest_edge_01",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=lambda region, annotation: _get_rest_edges(region, annotation)[0],
        ),
        "rest_edge_02": MaterialField(
            name="rest_edge_02",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=lambda region, annotation: _get_rest_edges(region, annotation)[1],
        ),
        "rest_metric_inv": MaterialField(
            name="rest_metric_inv",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((2, 2), dtype)),
            factory=_get_rest_metric_inv,
        ),
        "rest_metric_sqrt_det": MaterialField(
            name="rest_metric_sqrt_det",
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=_get_rest_metric_sqrt_det,
        ),
    }

    fun_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _fun_kernel)
    grad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _grad_kernel)
    hess_diag_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_diag_kernel)
    hess_prod_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_prod_kernel)
    hess_quad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _hess_quad_kernel)

    cells: wp.array
    thickness: float = attrs.field(default=1.0, kw_only=True)

    @classmethod
    @override
    def from_region(
        cls, region: Region, requires_grad: Sequence[str] = (), **kwargs
    ) -> Self:
        cells: Tensor = region.cells_global.to(torch.int32).contiguous()
        if cells.shape[1] != 3:
            msg = (
                f"Koiter expects triangle cells, got {cells.shape[1]} vertices per cell"
            )
            raise ValueError(msg)
        self: Self = cls(cells=wp.from_torch(cells, dtype=wp.vec3i), **kwargs)
        self.materials = self.material_from_region(region, requires_grad=requires_grad)
        return self

    @property
    def launch_dim(self) -> int:
        return self.cells.shape[0]

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.fun_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials, self.thickness],
            outputs=[output],
        )

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.grad_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials, self.thickness],
            outputs=[output],
        )

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_diag_kernel,
            dim=self.launch_dim,
            inputs=[u, self.cells, self.materials, self.thickness],
            outputs=[output],
        )

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_prod_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.cells, self.materials, self.thickness],
            outputs=[output],
        )

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        wp.launch(
            self.hess_quad_kernel,
            dim=self.launch_dim,
            inputs=[u, p, self.cells, self.materials, self.thickness],
            outputs=[output],
        )
