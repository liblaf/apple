from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import sim, struct, utils


class Inertia(sim.Energy):
    obj: sim.Object = struct.data()

    # region Node

    @property
    @override
    def deps(self) -> struct.NodeCollection[sim.Object]:
        return struct.NodeCollection(self.obj)

    @override
    def with_deps(self, nodes: struct.NodesLike, /) -> Self:
        nodes = struct.NodeCollection(nodes)
        obj: sim.Object = nodes[self.obj]
        return self.evolve(obj=obj)

    # endregion Node

    @property
    def displacement_prev(self) -> sim.Field:
        return self.obj.displacement_prev

    @property
    def velocity(self) -> sim.Field:
        return self.obj.velocity

    @property
    def force(self) -> sim.Field:
        return self.obj.force

    @property
    def mass(self) -> sim.Field:
        return self.displacement_prev.with_values(self.obj.mass)

    @override
    @utils.jit
    def fun(
        self, x: sim.FieldCollection, /, params: sim.GlobalParams
    ) -> Float[jax.Array, ""]:
        x: sim.Field = x[self.obj.id]
        x_tilde: Float[jax.Array, "points dim"] = (
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
    def jac(
        self, x: sim.FieldCollection, /, params: sim.GlobalParams
    ) -> sim.FieldCollection:
        x: sim.Field = x[self.obj.id]
        x_tilde: Float[jax.Array, "points dim"] = (
            self.displacement_prev
            + params.time_step * self.velocity
            + params.time_step**2 * self.force / self.mass
        )
        jac: Float[jax.Array, "points dim"] = self.mass * (x - x_tilde)
        jac /= params.time_step**2
        return sim.FieldCollection({self.obj.id: x.with_values(jac)})

    @override
    @utils.jit
    def hess_diag(
        self, x: sim.FieldCollection, /, params: sim.GlobalParams
    ) -> sim.FieldCollection:
        x: sim.Field = x[self.obj.id]
        hess_diag: Float[jax.Array, "points dim"] = self.mass / params.time_step**2
        return sim.FieldCollection({self.obj.id: x.with_values(hess_diag)})

    @override
    @utils.jit
    def hess_quad(
        self,
        x: sim.FieldCollection,
        p: sim.FieldCollection,
        /,
        params: sim.GlobalParams,
    ) -> Float[jax.Array, ""]:
        p: sim.Field = p[self.obj.id]
        hess_quad: Float[jax.Array, ""] = jnp.sum(jnp.asarray(self.mass * p**2))
        hess_quad /= params.time_step**2
        return hess_quad
