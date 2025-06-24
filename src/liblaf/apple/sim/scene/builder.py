import attrs

from liblaf.apple import struct
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import make_dofs
from liblaf.apple.sim.energy import Energy
from liblaf.apple.sim.params import GlobalParams

from .scene import Scene


@attrs.define
class SceneBuilder:
    actors: struct.NodeContainer[Actor] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )
    energies: struct.NodeContainer[Energy] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )
    params: GlobalParams = attrs.field(factory=lambda: GlobalParams())

    @property
    def dirichlet(self) -> Dirichlet:
        return Dirichlet.union(
            *(
                actor.dirichlet
                for actor in self.actors.values()
                if actor.dirichlet is not None
            )
        )

    @property
    def actors_needed(self) -> struct.NodeContainer[Actor]:
        actors: struct.NodeContainer[Actor] = struct.NodeContainer()
        for energy in self.energies.values():
            actors = actors.update(energy.actors)
        return actors

    @property
    def n_dofs(self) -> int:
        return sum(actor.n_dofs for actor in self.actors.values())

    def add_energy(self, energy: Energy) -> Energy:
        self.energies = self.energies.add(energy)
        return energy

    def assign_dofs(self, actor: Actor) -> Actor:
        actor = actor.with_dofs(
            make_dofs((actor.n_points, actor.dim), offset=self.n_dofs)
        )
        self.actors = self.actors.add(actor)
        return actor

    def finish(self) -> Scene:
        return Scene(
            actors=self.actors_needed,
            dirichlet=self.dirichlet,
            energies=self.energies,
            params=self.params,
            n_dofs=self.n_dofs,
        )
