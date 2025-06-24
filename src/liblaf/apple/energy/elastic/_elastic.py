from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import elem, sim, struct, utils


class Elastic(sim.Energy):
    hess_diag_filter: bool = struct.static(default=True, kw_only=True)
    hess_quad_filter: bool = struct.static(default=True, kw_only=True)
    obj: sim.Actor = struct.data()

    @property
    def deps(self) -> struct.FrozenDict:
        return struct.FrozenDict(self.obj)

    def with_deps(self, deps: struct.MappingLike) -> Self:
        deps = struct.FrozenDict(deps)
        return self.evolve(obj=deps[self.obj.id])

    def make_field(self, x: struct.DictArray, /) -> sim.Field:
        x: Float[jax.Array, "points dim"] = x[self.obj.id]
        return self.obj.displacement.with_values(x)

    @override
    @utils.jit
    def fun(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        field: sim.Field = self.make_field(x)
        Psi: Float[jax.Array, " cells"] = self.energy_density(field)
        return jnp.dot(Psi, field.dV)

    @override
    @utils.jit
    def jac(self, x: struct.DictArray, /, params: sim.GlobalParams) -> struct.DictArray:
        field: sim.Field = self.make_field(x)
        jac: Float[jax.Array, "cells 4 3"] = self.energy_density_jac(field)
        jac: Float[jax.Array, "points 3"] = field.region.gather(jac * field.dV[:, None])
        return struct.DictArray({self.obj.id: jac})

    @override
    @utils.jit
    def hess_diag(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> struct.DictArray:
        field: sim.Field = self.make_field(x)
        hess_diag: Float[jax.Array, "cells 4 3"] = self.energy_density_hess_diag(field)
        if self.hess_diag_filter:
            hess_diag = jnp.clip(hess_diag, min=0.0)
        hess_diag: Float[jax.Array, "points 3"] = field.region.gather(
            hess_diag * field.dV[:, None]
        )
        return struct.DictArray({self.obj.id: hess_diag})

    @override
    @utils.jit
    def hess_quad(
        self, x: struct.DictArray, p: struct.DictArray, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        field: sim.Field = self.make_field(x)
        field_p: sim.Field = self.make_field(p)
        hess_quad: Float[jax.Array, " cells"] = self.energy_density_hess_quad(
            field, field_p
        )
        if self.hess_quad_filter:
            hess_quad = jnp.clip(hess_quad, min=0.0)
        hess_quad: Float[jax.Array, ""] = jnp.vdot(hess_quad, field.dV)
        return hess_quad

    @override
    @utils.jit
    def fun_and_jac(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> tuple[Float[jax.Array, ""], struct.DictArray]:
        field: sim.Field = self.make_field(x)
        dV: Float[jax.Array, " cells"] = field.dV
        Psi: Float[jax.Array, " cells"] = self.energy_density(field)
        dPsidx: Float[jax.Array, " cells 4 3"] = self.energy_density_jac(field)
        fun: Float[jax.Array, ""] = jnp.dot(Psi, dV)
        jac: Float[jax.Array, "points 3"] = elem.tetra.segment_sum(
            dPsidx * dV[:, None, None], field.cells, n_points=field.n_points
        )
        return fun, struct.DictArray({self.obj.id: jac})

    @utils.jit
    @override
    def jac_and_hess_diag(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> tuple[struct.DictArray, struct.DictArray]:
        return self.jac(x, params), self.hess_diag(x, params)

    def energy_density(self, field: sim.Field) -> Float[jax.Array, " cells"]:
        raise NotImplementedError

    def first_piola_kirchhoff_stress(
        self, field: sim.Field
    ) -> Float[jax.Array, "cells 3 3"]:
        raise NotImplementedError

    @utils.jit
    def energy_density_jac(self, field: sim.Field) -> Float[jax.Array, " cells 4 3"]:
        PK1: Float[jax.Array, "cells 3 3"] = self.first_piola_kirchhoff_stress(field)
        dPsidx: Float[jax.Array, "cells 4 3"] = elem.tetra.deformation_gradient_vjp(
            field.dhdX.squeeze(), PK1
        )
        return dPsidx

    def energy_density_hess_diag(
        self, field: sim.Field
    ) -> Float[jax.Array, "cells 4 3"]:
        raise NotImplementedError

    def energy_density_hess_quad(
        self, field: sim.Field, p: sim.Field
    ) -> Float[jax.Array, " cells"]:
        raise NotImplementedError
