from collections.abc import Generator, Iterator
from typing import cast, override

import attrs

from liblaf.apple import struct
from liblaf.apple.sim.abc import Energy, GlobalParams, Object

from ._scene import Scene


@attrs.define
class SceneBuilder(struct.MappingTrait):
    params: GlobalParams = attrs.field(factory=GlobalParams)
    _graph: struct.Graph = attrs.field(factory=struct.Graph, init=False)
    _energy_keys: list[str] = attrs.field(factory=list, init=False)

    # region NodeCollectionMixin

    @override
    def __getitem__(self, key: struct.KeyLike, /) -> struct.GraphNode:
        return self.graph[key]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self.graph

    @override
    def __len__(self) -> int:
        return len(self.graph)

    # endregion NodeCollectionMixin

    @property
    def bases(self) -> struct.PyTreeDict[Object]:
        return cast("struct.PyTreeDict[Object]", self._graph.select(self.graph.bases))

    @property
    def energies(self) -> struct.PyTreeDict[Energy]:
        return cast("struct.PyTreeDict[Energy]", self.graph.select(self._energy_keys))

    @property
    def graph(self) -> struct.Graph:
        return self._graph

    @property
    def n_dof(self) -> int:
        return sum(obj.n_dof for obj in self.bases.values())

    @property
    def nodes(self) -> struct.PyTreeDict:
        return struct.PyTreeDict(self)

    @property
    def topological(self) -> Generator[str]:
        return self.graph.topological

    def add(self, node: struct.GraphNode, /) -> None:
        self.graph.add(node)

    def add_energy(self, energy: Energy, /) -> None:
        self._energy_keys.append(energy.id)
        self.add(energy)

    def assign_dof[T: Object](self, obj: T) -> T:
        obj = obj.replace(dof_map=struct.make_dof_map(obj.shape, offset=self.n_dof))
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
