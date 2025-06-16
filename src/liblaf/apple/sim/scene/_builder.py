from collections.abc import Generator, Iterator
from typing import override

import attrs

from liblaf.apple import struct
from liblaf.apple.sim.abc import Dirichlet, Energy, GlobalParams, Object

from ._scene import Scene


@attrs.define
class SceneBuilder(struct.NodeCollectionMixin):
    params: GlobalParams = attrs.field(factory=GlobalParams)
    _graph: struct.Graph = attrs.field(factory=struct.Graph, init=False)
    _energy_keys: list[str] = attrs.field(factory=list, init=False)

    # region NodeCollectionMixin

    @override
    def __getitem__(self, key: struct.KeyLike, /) -> struct.Node:
        return self.graph[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.graph

    @override
    def __len__(self) -> int:
        return len(self.graph)

    # endregion NodeCollectionMixin

    @property
    def bases(self) -> struct.NodeCollection[Object]:
        return self.graph.bases

    @property
    def dirichlet(self) -> Dirichlet:
        return Dirichlet.concat(*(obj.dirichlet for obj in self.bases.values()))

    @property
    def energies(self) -> struct.NodeCollection[Energy]:
        return self.graph.select(self._energy_keys)

    @property
    def graph(self) -> struct.Graph:
        return self._graph

    @property
    def n_dof(self) -> int:
        return sum(obj.n_dof for obj in self.bases.values())

    @property
    def nodes(self) -> struct.NodeCollection:
        return struct.NodeCollection(self)

    @property
    def topological(self) -> Generator[str]:
        return self.graph.topological

    def add(self, node: struct.Node, /) -> None:
        self.graph.add(node)

    def add_energy(self, energy: Energy, /) -> None:
        self._energy_keys.append(energy.id)
        self.add(energy)

    def assign_dof[T: Object](self, obj: T) -> T:
        obj = obj.evolve(dof_index=struct.make_index(obj.shape, start=self.n_dof))
        self.add(obj)
        return obj

    def build(self) -> Scene:
        return Scene(
            nodes=self.nodes,
            params=self.params,
            topological=tuple(self.topological),
            _base_keys=self.bases.keys(),
            _energy_keys=self._energy_keys,
        )
