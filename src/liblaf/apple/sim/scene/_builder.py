from typing import Self

from liblaf.apple import struct
from liblaf.apple.sim.abc import Energy

from ._scene import Scene


class SceneBuilder(struct.PyTree):
    energies: struct.NodeCollection[Energy] = struct.data(factory=struct.NodeCollection)

    def add_energy(self, energy: Energy) -> Self:
        return self.evolve(energies=self.energies.add(energy))

    def build(self) -> Scene:
        raise NotImplementedError
