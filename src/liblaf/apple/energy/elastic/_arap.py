from typing import no_type_check, override

import flax.struct
import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, physics, utils
from liblaf.apple.typed.warp import mat33, mat43

from ._elastic import Elastic


class ARAP(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    id: str = flax.struct.field(pytree_node=False, default="ARAP")

    mu: Float[jax.Array, " cells"] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0)
    )

    @override
    @utils.jit
    def energy_density(self, field: physics.Field) -> Float[jax.Array, " cells"]:
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        Psi: Float[jax.Array, " cells"]
        (Psi,) = arap_energy_density_warp(field.deformation_gradient, mu)
        return Psi

    @override
    @utils.jit
    def first_piola_kirchhoff_stress(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 3 3"]:
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        PK1: Float[jax.Array, " cells"]
        (PK1,) = arap_first_piola_kirchhoff_stress_warp(field.deformation_gradient, mu)
        return PK1

    @override
    @utils.jit
    def energy_density_hess_diag(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 4 3"]:
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        hess_diag: Float[jax.Array, "cells 4 3"]
        (hess_diag,) = arap_energy_density_hess_diag_warp(
            field.deformation_gradient, mu, field.dh_dX
        )
        return hess_diag

    @override
    @utils.jit
    def energy_density_hess_quad(
        self, field: physics.Field, p: physics.Field
    ) -> Float[jax.Array, " cells"]:
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        hess_quad: Float[jax.Array, " cells"]
        (hess_quad,) = arap_energy_density_hess_quad_warp(
            field.deformation_gradient, p.values[p.cells], mu, field.dh_dX
        )
        return hess_quad


@no_type_check
@utils.jax_kernel
def arap_energy_density_warp(
    F: wp.array(dtype=mat33), mu: wp.array(dtype=float), Psi: wp.array(dtype=float)
) -> None:
    tid = wp.tid()
    Psi[tid] = func.elastic.arap_energy_density(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_first_piola_kirchhoff_stress_warp(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    PK1: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    PK1[tid] = func.elastic.arap_first_piola_kirchhoff_stress(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_diag_warp(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_diag: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    hess_diag[tid] = func.elastic.arap_energy_density_hess_diag(
        F=F[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_quad_warp(
    F: wp.array(dtype=mat33),
    p: wp.array(dtype=mat43),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    hess_quad[tid] = func.elastic.arap_energy_density_hess_quad(
        F=F[tid], p=p[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )
