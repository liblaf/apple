from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import sim, struct, utils


class Inertia(sim.Energy):
    obj: sim.Object = struct.data()

    # region Computational Graph

    @property
    @override
    def deps(self) -> struct.FrozenDict:
        return struct.FrozenDict(self.obj)

    @override
    def with_deps(self, nodes: struct.MappingLike, /) -> Self:
        nodes = struct.FrozenDict(nodes)
        obj: sim.Object = nodes[self.obj]
        return self.evolve(obj=obj)

    # endregion Computational Graph

    @property
    def displacement_prev(self) -> sim.Field:
        return self.obj.fields["displacement_prev"]

    @property
    def velocity(self) -> sim.Field:
        return self.obj.fields["velocity"]

    @property
    def force(self) -> sim.Field:
        return self.obj.fields["force"]

    @property
    def mass(self) -> sim.Field:
        return self.obj.fields["mass"]

    @override
    @utils.jit
    def fun(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = x[self.obj.id]
        x_tilde: sim.Field = (
            self.displacement_prev
            + params.time_step * self.velocity
            + params.time_step**2 * self.force / self.mass
        )
        fun: Float[jax.Array, ""] = 0.5 * jnp.sum(
            jnp.asarray(self.mass * (x - x_tilde) ** 2)
        )
        fun /= params.time_step**2
        return fun

    @override
    @utils.jit
    def jac(self, x: struct.DictArray, /, params: sim.GlobalParams) -> struct.DictArray:
        x: Float[jax.Array, "points dim"] = x[self.obj.id]
        x_tilde: sim.Field = (
            self.displacement_prev
            + params.time_step * self.velocity
            + params.time_step**2 * self.force / self.mass
        )
        jac: sim.Field = self.mass * (x - x_tilde)
        jac /= params.time_step**2
        return struct.DictArray({self.obj.id: jac.values})

    @override
    @utils.jit
    def hess_diag(
        self, x: struct.DictArray, /, params: sim.GlobalParams
    ) -> struct.DictArray:
        hess_diag: sim.Field = self.mass / params.time_step**2
        return struct.DictArray({self.obj.id: hess_diag.values})

    @override
    @utils.jit
    def hess_quad(
        self,
        x: struct.DictArray,
        p: struct.DictArray,
        /,
        params: sim.GlobalParams,
    ) -> Float[jax.Array, ""]:
        p: Float[jax.Array, "points dim"] = p[self.obj.id]
        hess_quad: Float[jax.Array, ""] = jnp.sum(jnp.asarray(self.mass * p**2))
        hess_quad /= params.time_step**2
        return hess_quad
