from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast, no_type_check, override

import attrs
import warp as wp

from liblaf.apple.common import ACTIVATION_INV, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math
from liblaf.apple.warp.model import MaterialField

from . import func, utils
from ._base import WarpPotentialFem

floating = Any
mat33 = Any
mat43 = Any
Materials = Any
vec3 = Any
vec4i = Any


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])  # mat33
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    G = F @ A_inv  # mat33
    I2 = func.I2(G)  # float
    J = func.I3(F)  # float
    return (
        F.dtype(0.5) * mu * (I2 - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * math.square(J - F.dtype(1.0))
    )


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])  # mat33
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    G = F @ A_inv  # mat33
    J = func.I3(F)  # float
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    g2 = func.g2(G)  # mat33
    g3 = func.g3(F)  # mat33
    return dPsi_dI2 * g2 @ wp.transpose(A_inv) + dPsi_dI3 * g3


@wp.func
def hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat33:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])  # mat33
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    # h2_diag = func.h2_diag(dhdX, g2)  # mat43
    dhdX_A = dhdX @ A_inv  # mat43
    h3_diag = func.h3_diag(dhdX, g3)  # mat43
    h5_diag = func.h5_diag(dhdX_A)  # mat43
    h6_diag = func.h6_diag(dhdX, F)  # mat43
    return (
        # d2Psi_dI22 * h2_diag
        d2Psi_dI32 * h3_diag + dPsi_dI2 * h5_diag + dPsi_dI3 * h6_diag
    )


@wp.func
def hess_prod(F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int) -> mat33:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])  # mat33
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    dhdX_A = dhdX @ A_inv  # mat43
    h3_prod = func.h3_prod(p, dhdX, g3)  # mat43
    h5_prod = func.h5_prod(p, dhdX_A)  # mat43
    h6_prod = func.h6_prod(p, dhdX, F)  # mat43
    return d2Psi_dI32 * h3_prod + dPsi_dI2 * h5_prod + dPsi_dI3 * h6_prod


@wp.func
def hess_quad(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> floating:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])  # mat33
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    dhdX_A = dhdX @ A_inv  # mat43
    h3_quad = func.h3_quad(p, dhdX, g3)  # float
    h5_quad = func.h5_quad(p, dhdX_A)  # float
    h6_quad = func.h6_quad(p, dhdX, F)  # float
    return d2Psi_dI32 * h3_quad + dPsi_dI2 * h5_quad + dPsi_dI3 * h6_quad


@wp.kernel(module="unique")
@no_type_check
def _fun_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec4i],
    materials: Materials,
    output: wp.array1d[floating],
) -> None:
    cid, qid = wp.tid()
    cell = cells[cid]
    u_cell = func.get_cell_displacements(u, cell)
    F = func.deformation_gradient(u_cell, materials.dhdX[cid, qid])
    wp.atomic_add(output, 0, energy_density(F, materials, cid) * materials.dV[cid, qid])


@wp.kernel(module="unique")
@no_type_check
def _grad_kernel(
    u: wp.array1d[vec3],
    cells: wp.array1d[vec4i],
    materials: Materials,
    output: wp.array1d[vec3],
) -> None:
    cid, qid = wp.tid()
    cell = cells[cid]
    u_cell = func.get_cell_displacements(u, cell)
    dhdX = materials.dhdX[cid, qid]
    F = func.deformation_gradient(u_cell, dhdX)
    grad_cell = (
        func.deformation_gradient_vjp(dhdX, first_piola_kirchhoff(F, materials, cid))
        * materials.dV[cid, qid]
    )
    for i in range(4):
        wp.atomic_add(output, cell[i], grad_cell[i])


@attrs.define
class StableNeoHookeanActive(WarpPotentialFem):
    r"""Stable Neo-Hookean active strain with physical-volume regularization.

    With ``B = A_inv``, the activated norm is ``||F B||²`` while both
    determinant terms use the physical deformation ``J = det(F)``. Therefore
    activation changes the directional elastic response without redefining the
    volume measured by the volumetric penalty.
    """

    class Materials(WarpPotentialFem.Materials):
        activation_inv: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVATION_INV.value: MaterialField(
            name=ACTIVATION_INV.value,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(6, dtype)),
            factory=utils.get_activation_inv,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
    }

    energy_density_func: ClassVar[wp.Function] = cast("wp.Function", energy_density)
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", first_piola_kirchhoff
    )
    hess_diag_func: ClassVar[wp.Function] = cast("wp.Function", hess_diag)
    hess_prod_func: ClassVar[wp.Function] = cast("wp.Function", hess_prod)
    hess_quad_func: ClassVar[wp.Function] = cast("wp.Function", hess_quad)

    energy_density_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_energy_density_kernel(energy_density_func)
    )
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_first_piola_kirchhoff_kernel(first_piola_kirchhoff_func)
    )

    # Keep these direct kernels: the callback factories do not propagate
    # material-array adjoints through their constitutive-function argument.
    fun_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _fun_kernel)
    grad_kernel: ClassVar[wp.Kernel] = cast("wp.Kernel", _grad_kernel)
    hess_prod_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_prod_kernel(
        hess_prod_func
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_diag_kernel(
        hess_diag_func
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_quad_kernel(
        hess_quad_func
    )

    @override
    def material_from_region(
        self, region: Region, requires_grad: Sequence[str] = ()
    ) -> Any:
        materials = self.material_struct()
        with wp.ScopedDevice(self.cells.device):
            for field in self.material_vars.values():
                value = field.from_region(region)
                value.requires_grad = field.name in requires_grad
                setattr(materials, field.name, value)
        return materials

    @override
    def energy_density(self, u: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().energy_density(u, output)

    @override
    def first_piola_kirchhoff(self, u: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().first_piola_kirchhoff(u, output)

    @override
    def fun(self, u: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().fun(u, output)

    @override
    def grad(self, u: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().grad(u, output)

    @override
    def hess_diag(self, u: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().hess_diag(u, output)

    @override
    def hess_prod(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().hess_prod(u, p, output)

    @override
    def hess_quad(self, u: wp.array, p: wp.array, output: wp.array) -> None:
        with wp.ScopedDevice(self.cells.device):
            super().hess_quad(u, p, output)
