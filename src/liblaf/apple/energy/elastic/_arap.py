from typing import no_type_check, override

import flax.struct
import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, physics, utils

from ._elastic import Elastic


class ARAP(Elastic):
    mu: Float[jax.Array, " cells"] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0)
    )

    @override
    @utils.jit
    def energy_density(
        self, field: physics.Field, F: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, " cells"]:
        F = jnp.asarray(F)
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        Psi: Float[jax.Array, " cells"]
        (Psi,) = arap_energy_density_warp(F, mu)
        return Psi

    @override
    @utils.jit
    def first_piola_kirchhoff_stress(
        self, field: physics.Field, F: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells 3 3"]:
        F = jnp.asarray(F)
        mu: Float[jax.Array, " cells"] = jnp.broadcast_to(self.mu, (field.n_cells,))
        PK1: Float[jax.Array, " cells"]
        (PK1,) = arap_first_piola_kirchhoff_stress_warp(F, mu)
        return PK1


@no_type_check
@utils.jax_kernel
def arap_energy_density_warp(
    F: wp.array(dtype=wp.mat33), mu: wp.array(dtype=float), Psi: wp.array(dtype=float)
) -> None:
    tid = wp.tid()
    Psi[tid] = func.elastic.arap_energy_density(F[tid], mu[tid])


@no_type_check
@utils.jax_kernel
def arap_first_piola_kirchhoff_stress_warp(
    F: wp.array(dtype=wp.mat33),
    mu: wp.array(dtype=float),
    PK1: wp.array(dtype=wp.mat33),
) -> None:
    tid = wp.tid()
    PK1[tid] = func.elastic.arap_first_piola_kirchhoff_stress(F[tid], mu[tid])
