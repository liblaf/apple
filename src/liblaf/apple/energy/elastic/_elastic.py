from typing import override

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import elem, physics, utils


class Elastic(physics.Energy):
    hess_diag_filter: bool = flax.struct.field(
        pytree_node=False, default=True, kw_only=True
    )
    hess_quad_filter: bool = flax.struct.field(
        pytree_node=False, default=True, kw_only=True
    )

    @override
    @utils.jit
    def fun(self, field: physics.Field) -> Float[jax.Array, ""]:
        Psi: Float[jax.Array, " cells"] = self.energy_density(field)
        return jnp.dot(Psi, field.dV)

    @override
    @utils.jit
    def jac(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        dV: Float[jax.Array, " cells"] = field.dV
        dPsidx: Float[jax.Array, " cells 4 3"] = self.energy_density_jac(field)
        jac: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            dPsidx * dV[:, None, None], field.cells, n_points=field.n_points
        )
        return jac.ravel()

    @override
    @utils.jit
    def hess_diag(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        dV: Float[jax.Array, " cells"] = field.dV
        hess_diag: Float[jax.Array, "cells 4 3"] = self.energy_density_hess_diag(field)
        if self.hess_diag_filter:
            hess_diag = jnp.clip(hess_diag, min=0.0)
        hess_diag: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            hess_diag * dV[:, None, None], field.cells, n_points=field.n_points
        )
        # hess_diag = jnp.ones_like(hess_diag)
        return hess_diag.ravel()

    @override
    @utils.jit
    def hess_quad(self, field: physics.Field, p: physics.Field) -> Float[jax.Array, ""]:
        dV: Float[jax.Array, " cells"] = field.dV
        hess_quad: Float[jax.Array, " cells"] = self.energy_density_hess_quad(field, p)
        if self.hess_quad_filter:
            hess_quad = jnp.clip(hess_quad, min=0.0)
        hess_quad: Float[jax.Array, ""] = jnp.dot(hess_quad, dV)
        return hess_quad

    @override
    @utils.jit
    def fun_and_jac(
        self, field: physics.Field
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        dV: Float[jax.Array, " cells"] = field.dV
        Psi: Float[jax.Array, " cells"] = self.energy_density(field)
        dPsidx: Float[jax.Array, " cells 4 3"] = self.energy_density_jac(field)
        fun: Float[jax.Array, ""] = jnp.dot(Psi, dV)
        jac: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            dPsidx * dV[:, None, None], field.cells, n_points=field.n_points
        )
        return fun, jac.ravel()

    @override
    @utils.jit
    def jac_and_hess_diag(
        self, field: physics.Field
    ) -> tuple[Float[jax.Array, " DoF"], Float[jax.Array, " DoF"]]:
        return self.jac(field), self.hess_diag(field)

    def energy_density(self, field: physics.Field) -> Float[jax.Array, " cells"]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 3 3"]:
        raise NotImplementedError

    @utils.jit
    def energy_density_jac(
        self, field: physics.Field
    ) -> Float[jax.Array, " cells 4 3"]:
        PK1: Float[jax.Array, "cells 3 3"] = self.first_piola_kirchhoff_stress(field)
        dPsidx: Float[jax.Array, "cells 4 3"] = elem.tetra.deformation_gradient_vjp(
            field.dh_dX, PK1
        )
        return dPsidx

    def energy_density_hess_diag(
        self, field: physics.Field
    ) -> Float[jax.Array, "cells 4 3"]:
        raise NotImplementedError

    def energy_density_hess_quad(
        self, field: physics.Field, p: physics.Field
    ) -> Float[jax.Array, " cells"]:
        raise NotImplementedError
