from typing import Self

from liblaf.apple import struct
from liblaf.apple.sim.actor import Actor


class Component(struct.PyTree):
    def register[T: Actor](self, actor: T) -> tuple[Self, T]:
        actor = actor.replace(components=[*actor.components, self])
        return self, actor

    def pre_time_step[T: Actor](self, actor: T) -> tuple[Self, T]:
        return self, actor

    def pre_optim_iter[T: Actor](self, actor: T) -> tuple[Self, T]:
        return self, actor
