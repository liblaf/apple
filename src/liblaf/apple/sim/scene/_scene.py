from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct, utils
from liblaf.apple.sim.abc import Energy, Object


class Scene(struct.DerivativeMixin, struct.PyTree):
    bases: struct.NodeCollection[Object] = struct.data(factory=struct.NodeCollection)
    energies: struct.NodeCollection[Energy] = struct.data(factory=struct.NodeCollection)
    nodes: struct.NodeCollection = struct.data(factory=struct.NodeCollection)
    topological_sort: list[str] = struct.static(factory=list)

    # region Shape

    @property
    def n_dof(self) -> int:
        raise NotImplementedError

    # endregion Shape

    # region Optimization

    @utils.jit
    @override
    def fun(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, ""]:
        x = jnp.asarray(x)
        return jnp.sum(
            jnp.asarray([energy.fun(energy.dof_index(x)) for energy in self.energies])
        )

    @utils.jit
    @override
    def jac(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        x = jnp.asarray(x)
        jac: Float[jax.Array, " DoF"] = jnp.zeros((self.n_dof,))
        for energy in self.energies:
            jac = energy.dof_index.add(jac, energy.jac(energy.dof_index(x)))
        return jac

    @utils.jit
    @override
    def hessp(
        self,
        x: Float[ArrayLike, " DoF"],
        p: Float[ArrayLike, " DoF"],
        /,
    ) -> Float[jax.Array, " DoF"]:
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        hessp: Float[jax.Array, " DoF"] = jnp.zeros((self.n_dof,))
        for energy in self.energies:
            hessp = energy.dof_index.add(hessp, energy.hessp(energy.dof_index(x), p))
        return hessp

    @utils.jit
    @override
    def hess_diag(self, x: Float[ArrayLike, " DoF"], /) -> Float[jax.Array, " DoF"]:
        x = jnp.asarray(x)
        hess_diag: Float[jax.Array, " DoF"] = jnp.zeros((self.n_dof,))
        for energy in self.energies:
            hess_diag = energy.dof_index.add(
                hess_diag, energy.hess_diag(energy.dof_index(x))
            )
        return hess_diag

    # endregion Optimization

    # region State Update

    def update(self, bases: struct.CollectionLike | None = None) -> Self:
        bases: struct.NodeCollection = struct.NodeCollection(bases)
        if not bases:
            return self
        bases = self.bases.update(bases)
        nodes: struct.NodeCollection = self.nodes.update(bases)
        for node_id in self.topological_sort:
            node: struct.Node = nodes[node_id]
            nodes = nodes.add(node.update(nodes))
        return self.evolve(
            bases=bases,
            energies=nodes.select(self.energies),
            nodes=nodes.select(self.nodes),
        )

    def with_displacement(
        self, displacement: Float[ArrayLike, " DoF"] | None, /
    ) -> Self:
        if displacement is None:
            return self
        displacement = jnp.asarray(displacement)
        leaves: struct.NodeCollection = self.nodes
        offset: int = 0
        for obj in self.nodes:
            if obj.n_dof > 0:
                obj = obj.with_displacement(displacement[offset : offset + obj.n_dof])
                leaves = leaves.update(obj)
            offset += obj.n_dof
        return self.evolve(objects=leaves)

    # endregion State Update
