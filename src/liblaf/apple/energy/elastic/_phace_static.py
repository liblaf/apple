from typing import no_type_check, override

import flax.struct
import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, physics, utils
from liblaf.apple.typed.warp import mat33, mat43

from ._elastic import Elastic


class PhaceStatic(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    id: str = flax.struct.field(pytree_node=False, default="phace-static", kw_only=True)

    mu: Float[jax.Array, " cells"] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0)
    )
    lambda_: Float[jax.Array, " cells"] = flax.struct.field(
        default_factory=lambda: jnp.asarray(3.0)
    )

    @override
    @utils.jit
    def energy_density(self, field: physics.Field) -> Float[jax.Array, " cells"]:
        params: Float[jax.Array, " cells 2"] = self.make_params(field)
        Psi: Float[jax.Array, " cells"]
        (Psi,) = phace_static_energy_density_warp(field.deformation_gradient, params)
        return Psi

    @override
    @utils.jit
    def first_piola_kirchhoff_stress(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 3 3"]:
        params: Float[jax.Array, " cells 2"] = self.make_params(field)
        PK1: Float[jax.Array, " cells"]
        (PK1,) = phace_static_first_piola_kirchhoff_stress_warp(
            field.deformation_gradient, params
        )
        return PK1

    @override
    @utils.jit
    def energy_density_hess_diag(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 4 3"]:
        params: Float[jax.Array, " cells 2"] = self.make_params(field)
        hess_diag: Float[jax.Array, "cells 4 3"]
        (hess_diag,) = phace_static_energy_density_hess_diag_warp(
            field.deformation_gradient, params, field.dh_dX
        )
        return hess_diag

    @override
    @utils.jit
    def energy_density_hess_quad(
        self, field: physics.Field, p: physics.Field
    ) -> Float[jax.Array, " cells"]:
        params: Float[jax.Array, " cells 2"] = self.make_params(field)
        hess_quad: Float[jax.Array, " cells"]
        (hess_quad,) = phace_static_energy_density_hess_quad_warp(
            field.deformation_gradient, p.values[p.cells], params, field.dh_dX
        )
        return hess_quad

    def make_params(self, field: physics.Field) -> Float[jax.Array, "cells 2"]:
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        lambda_: Float[jax.Array, " cells"] = jnp.broadcast_to(
            self.lambda_, (field.n_cells,)
        )
        return jnp.stack((mu, lambda_), axis=-1)


@no_type_check
@utils.jax_kernel
def phace_static_energy_density_warp(
    F: wp.array(dtype=mat33),
    params: wp.array(dtype=wp.vec2),
    Psi: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    Psi[tid] = func.elastic.phace_static_energy_density(F=F[tid], params=params[tid])


@no_type_check
@utils.jax_kernel
def phace_static_first_piola_kirchhoff_stress_warp(
    F: wp.array(dtype=mat33),
    params: wp.array(dtype=wp.vec2),
    PK1: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    PK1[tid] = func.elastic.phace_static_first_piola_kirchhoff_stress(
        F=F[tid], params=params[tid]
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
    hess_diag[tid] = func.elastic.phace_static_energy_density_hess_diag(
        F=F[tid], params=params[tid], dh_dX=dh_dX[tid]
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
    hess_quad[tid] = func.elastic.phace_static_energy_density_hess_quad(
        F=F[tid], p=p[tid], params=params[tid], dh_dX=dh_dX[tid]
    )
