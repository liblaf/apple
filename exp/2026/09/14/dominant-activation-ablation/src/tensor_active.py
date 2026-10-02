"""Stable Neo-Hookean elasticity with additive symmetric tensor active stress.

The caller supplies a symmetric positive-semidefinite ``active_stress`` tensor
``Q`` per cell.  Projection to that cone and any norm cap belong to the caller;
this material intentionally trusts that contract.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, cast, no_type_check

import attrs
import warp as wp

from liblaf.apple.common import LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math as warp_math
from liblaf.apple.warp.fem import WarpPotentialFem, func
from liblaf.apple.warp.model import MaterialField

ACTIVE_STRESS = "active_stress"

floating = Any
mat33 = Any
mat43 = Any
Materials = Any


def get_active_stress(region: Region, annotation: Any) -> wp.array:
    """Return the zero-active-stress default for every cell."""
    return wp.zeros((region.mesh.n_cells,), dtype=annotation.dtype)


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    r"""Return ``W_stable(F) + 1/2 Q:(F^T F - I)``."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    Q = materials.active_stress[cid]  # mat33
    J = func.I3(F)  # float
    passive = (
        F.dtype(0.5) * mu * (func.I2(F) - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * warp_math.square(J - F.dtype(1.0))
    )
    C = wp.transpose(F) @ F  # mat33
    active = F.dtype(0.5) * wp.ddot(Q, C - wp.identity(3, dtype=F.dtype))
    return passive + active


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    r"""Return ``P_stable(F) + F Q`` under the symmetric-``Q`` contract."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    Q = materials.active_stress[cid]  # mat33
    J = func.I3(F)  # float
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = dPsi_dI2 * func.g2(F) + dPsi_dI3 * func.g3(F)  # mat33
    return passive + F @ Q


@wp.func
@no_type_check
def hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat43:
    """Return the exact nodal Hessian diagonal before kernel clamping."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    Q = materials.active_stress[cid]  # mat33
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_diag(dhdX, g3)
        + dPsi_dI2 * func.h5_diag(dhdX)
        + dPsi_dI3 * func.h6_diag(dhdX, F)
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
def hess_prod(
    F: mat33,
    p: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int,
) -> mat43:
    """Apply the exact assembled tangent to one nodal direction."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    Q = materials.active_stress[cid]  # mat33
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_prod(p, dhdX, g3)
        + dPsi_dI2 * func.h5_prod(p, dhdX)
        + dPsi_dI3 * func.h6_prod(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)  # mat33
    active_dP = dF @ Q  # mat33
    active = func.deformation_gradient_vjp(dhdX, active_dP)  # mat43
    return passive + active


@wp.func
def hess_quad(
    F: mat33,
    p: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int,
) -> floating:
    """Return the exact assembled tangent quadratic for one direction."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    Q = materials.active_stress[cid]  # mat33
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_quad(p, dhdX, g3)
        + dPsi_dI2 * func.h5_quad(p, dhdX)
        + dPsi_dI3 * func.h6_quad(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)  # mat33
    return passive + wp.ddot(dF, dF @ Q)


@attrs.define
class StableNeoHookeanTensorActive(WarpPotentialFem):
    """Stable Neo-Hookean material with additive tensor active stress.

    ``Q=active_stress`` must be symmetric positive semidefinite.  The class
    exposes it as a cellwise ``mat33`` material so Torch callers may pass
    ``materials[name]["active_stress"]`` with shape ``(cells, 3, 3)``.
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
            factory=get_active_stress,
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
    fun_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_fun_kernel(
        energy_density_func
    )
    grad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_grad_kernel(
        first_piola_kirchhoff_func
    )
    hess_prod_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_prod_kernel(
        hess_prod_func
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_diag_kernel(
        hess_diag_func
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_quad_kernel(
        hess_quad_func
    )


__all__ = [
    "ACTIVE_STRESS",
    "StableNeoHookeanTensorActive",
    "energy_density",
    "first_piola_kirchhoff",
    "hess_diag",
    "hess_prod",
    "hess_quad",
]
