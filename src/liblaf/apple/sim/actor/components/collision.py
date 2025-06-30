from typing import Self

import warp as wp

from liblaf.apple import struct
from liblaf.apple.sim.actor import Actor

from .component import Component


@struct.pytree
class ComponentCollision(Component):
    mesh: wp.Mesh = struct.static(default=None)

    def register[T: Actor](self, actor: T) -> tuple[Self, T]:
        component: Self = self
        mesh: wp.Mesh = actor.to_warp()
        component = component.evolve(mesh=mesh)
        actor = actor.evolve(
            collision_mesh=mesh, components=[*actor.components, component]
        )
        return self, actor

    def pre_optim_iter[T: Actor](self, actor: T) -> tuple[Self, T]:
        self.mesh.points = wp.from_jax(actor.positions, dtype=wp.vec3)
        self.mesh.refit()
        return self, actor
