from collections.abc import Sequence
from typing import Self

import jax.numpy as jnp
import networkx as nx
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim.abc import Energy, Object


class Scene(struct.PyTree):
    energies: struct.NodeCollection[Energy] = struct.data(factory=struct.NodeCollection)
    leaves: struct.NodeCollection[Object] = struct.data(factory=struct.NodeCollection)
    objects: struct.NodeCollection[Object] = struct.data(factory=struct.NodeCollection)
    topological_sort: Sequence[str] = struct.static(factory=list)

    # region Underlying

    # endregion Underlying

    # region Shape

    @property
    def n_dof(self) -> int:
        raise NotImplementedError

    # endregion Shape

    def add_energy(self, energy: Energy) -> Self:
        return self.evolve(energies=self.energies.add(energy))

    def build(self) -> Self:
        raise NotImplementedError

    def step(
        self,
        displacement: Float[ArrayLike, " DoF"] | None = None,
        velocity: Float[ArrayLike, " DoF"] | None = None,
        force: Float[ArrayLike, " DoF"] | None = None,
    ) -> Self:
        displacement = jnp.asarray(displacement)
        objects: struct.NodeCollection[Object] = self.objects
        offset: int = 0
        for obj in self.objects:
            if obj.n_dof > 0:
                obj = obj.step(displacement[offset : offset + obj.n_dof])
        raise NotImplementedError


class SceneBuilder(struct.PyTree):
    energies: struct.NodeCollection[Energy] = struct.data(factory=struct.NodeCollection)

    def build(self) -> Scene:
        graph: nx.DiGraph = self.energies.add_to_graph()
        nodes: struct.NodeCollection = self.energies.nodes
        for node_id in nx.lexicographical_topological_sort(graph):
            node: struct.Node = nodes[node_id]
            nodes.add(node.update(nodes))
        raise NotImplementedError

    def add_energy(self, energy: Energy) -> Self:
        return self.evolve(energies=self.energies.add(energy))
