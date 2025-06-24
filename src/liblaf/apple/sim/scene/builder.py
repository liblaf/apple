import attrs

from liblaf.apple import struct
from liblaf.apple.sim.actor import Actor
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import make_dofs

from .scene import Scene


@attrs.define
class SceneBuilder:
    actors: struct.NodeContainer[Actor] = attrs.field(
        converter=struct.NodeContainer, factory=struct.NodeContainer
    )

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
    def n_dofs(self) -> int:
        return sum(actor.n_dofs for actor in self.actors.values())

    def assign_dofs(self, actor: Actor) -> Actor:
        actor = actor.with_dofs(
            make_dofs((actor.n_points, actor.dim), offset=self.n_dofs)
        )
        self.actors = self.actors.add(actor)
        return actor

    def finish(self) -> Scene:
        return Scene(dirichlet=self.dirichlet)
