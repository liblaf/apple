from typing import Self

from liblaf.apple import struct


@struct.pytree
class ActorPlugin(struct.PyTreeMixin):
    def pre_optim_iter[T](self, actor: T) -> tuple[Self, T]: ...
    def pre_time_step[T](self, actor: T) -> tuple[Self, T]: ...
    def register[T](self, actor: T) -> tuple[Self, T]: ...
