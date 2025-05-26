import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import elem, physics, utils


class Elastic(physics.Energy):
    @utils.jit
    def fun(self, field: physics.Field) -> Float[jax.Array, ""]:
        F: Float[jax.Array, "cells 3 3"] = field.deformation_gradient
        Psi: Float[jax.Array, " cells"] = self.energy_density(field, F)
        return jnp.dot(Psi, field.dV)

    @utils.jit
    def jac(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        F: Float[jax.Array, "cells 3 3"] = field.deformation_gradient
        dV: Float[jax.Array, " cells"] = field.dV
        dPsidx: Float[jax.Array, " cells 4 3"] = self.energy_density_jac(field, F)
        jac: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            dPsidx * dV[:, None, None], field.cells, n_points=field.n_points
        )
        return jac.ravel()

    @utils.jit
    def fun_jac(
        self, field: physics.Field
    ) -> tuple[Float[jax.Array, ""], Float[jax.Array, " DoF"]]:
        F: Float[jax.Array, "cells 3 3"] = field.deformation_gradient
        dV: Float[jax.Array, " cells"] = field.dV
        Psi: Float[jax.Array, " cells"] = self.energy_density(field, F)
        dPsidx: Float[jax.Array, " cells 4 3"] = self.energy_density_jac(field, F)
        fun: Float[jax.Array, ""] = jnp.dot(Psi, dV)
        jac: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            dPsidx * dV[:, None, None], field.cells, n_points=field.n_points
        )
        return fun, jac.ravel()

    def energy_density(
        self, field: physics.Field, F: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, " cells"]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, field: physics.Field, F: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells 3 3"]:
        raise NotImplementedError

    @utils.jit
    def energy_density_jac(
        self, field: physics.Field, F: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, " cells 4 3"]:
        PK1: Float[jax.Array, "cells 3 3"] = self.first_piola_kirchhoff_stress(field, F)
        dPsidx: Float[jax.Array, "cells 4 3"] = elem.tetra.deformation_gradient_vjp(
            field.dh_dX, PK1
        )
        return dPsidx
