from typing import no_type_check, override

import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, sim, struct, utils
from liblaf.apple.typed.warp import mat33, mat43

from .elastic import Elastic


@struct.pytree
class PhaceStatic(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    @property
    def lambda_(self) -> Float[jax.Array, " cells"]:
        return self.actor.cell_data["lambda"]

    @property
    def mu(self) -> Float[jax.Array, " cells"]:
        return self.actor.cell_data["mu"]

    @property
    def params(self) -> Float[jax.Array, "cells 2"]:
        return jnp.stack((self.lambda_, self.mu), axis=-1)

    @override
    @utils.jit_method(inline=True)
    def energy_density(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        Psi: Float[jax.Array, " cq"]
        (Psi,) = phace_static_energy_density_warp(F, self.params)
        Psi: Float[jax.Array, "c q"] = region.unsqueeze_cq(Psi)
        return Psi

    @override
    @utils.jit_method(inline=True)
    def first_piola_kirchhoff_stress(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q J J"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        PK1: Float[jax.Array, "cq J J"]
        (PK1,) = phace_static_first_piola_kirchhoff_stress_warp(F, self.params)
        PK1: Float[jax.Array, "c q J J"] = region.unsqueeze_cq(PK1)
        return PK1

    @override
    @utils.jit_method(inline=True)
    def energy_density_hess_diag(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q a J"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_diag: Float[jax.Array, "cq a J"]
        (hess_diag,) = phace_static_energy_density_hess_diag_warp(F, self.params, dhdX)
        hess_diag: Float[jax.Array, "c q a J"] = region.unsqueeze_cq(hess_diag)
        return hess_diag

    @override
    @utils.jit_method(inline=True)
    def energy_density_hess_quad(
        self, field: sim.Field, p: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_quad: Float[jax.Array, " cq"]
        (hess_quad,) = phace_static_energy_density_hess_quad_warp(
            F, region.scatter(p.values), self.params, dhdX
        )
        hess_quad: Float[jax.Array, "c q"] = region.unsqueeze_cq(hess_quad)
        return hess_quad


@no_type_check
@utils.jax_kernel
def phace_static_energy_density_warp(
    F: wp.array(dtype=mat33),
    params: wp.array(dtype=wp.vec2),
    Psi: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    params_struct = func.elastic.PhaceStaticParams(
        lambda_=params[tid][0], mu=params[tid][1]
    )
    Psi[tid] = func.elastic.phace_static_energy_density(F=F[tid], params=params_struct)


@no_type_check
@utils.jax_kernel
def phace_static_first_piola_kirchhoff_stress_warp(
    F: wp.array(dtype=mat33),
    params: wp.array(dtype=wp.vec2),
    PK1: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    params_struct = func.elastic.PhaceStaticParams(
        lambda_=params[tid][0], mu=params[tid][1]
    )
    PK1[tid] = func.elastic.phace_static_first_piola_kirchhoff_stress(
        F=F[tid], params=params_struct
    )


@no_type_check
@utils.jax_kernel
def phace_static_energy_density_hess_diag_warp(
    F: wp.array(dtype=mat33),
    params: wp.array(dtype=wp.vec2),
    dh_dX: wp.array(dtype=mat43),
    hess_diag: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    params_struct = func.elastic.PhaceStaticParams(
        lambda_=params[tid][0], mu=params[tid][1]
    )
    hess_diag[tid] = func.elastic.phace_static_energy_density_hess_diag(
        F=F[tid], params=params_struct, dh_dX=dh_dX[tid]
    )


@no_type_check
@utils.jax_kernel
def phace_static_energy_density_hess_quad_warp(
    F: wp.array(dtype=mat33),
    p: wp.array(dtype=mat43),
    params: wp.array(dtype=wp.vec2),
    dh_dX: wp.array(dtype=mat43),
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    params_struct = func.elastic.PhaceStaticParams(
        lambda_=params[tid][0], mu=params[tid][1]
    )
    hess_quad[tid] = func.elastic.phace_static_energy_density_hess_quad(
        F=F[tid], p=p[tid], params=params_struct, dh_dX=dh_dX[tid]
    )
